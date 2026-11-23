"""
DistilBERT fine-tuning baseline.

GPU-heavy but provides a strong neural reference point.
~30 min on a T4, ~10 min on an A100.

Used to compare accuracy and latency against the BiLSTM-Attention model.

Requires: transformers
  pip install transformers

Skip with --skip if you don't have time.
"""

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
except ImportError:
    print("ERROR: transformers not installed. Run: pip install transformers")
    raise SystemExit(1)

from data_utils import (
    LABEL_TO_IDX,
    LABELS,
    NUM_CLASSES,
    build_text_input,
    class_weights_from_train,
    load_split,
)
from evaluate_common import count_parameters, save_results

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256  # DistilBERT max is 512; 256 covers most requests, faster
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42


class HFTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = []
        self.labels = []
        for _, row in df.iterrows():
            method = str(row.get("method", "GET"))
            path = str(row.get("path", "/"))
            query = str(row.get("query_string", ""))
            body = str(row.get("body", ""))
            if query == "nan":
                query = ""
            if body == "nan":
                body = ""
            self.texts.append(build_text_input(method, path, query, body))
            self.labels.append(LABEL_TO_IDX[row["label"]])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def measure_latency_hf(model, dataset, device, n_samples=500):
    """DistilBERT latency. Smaller n=500 because it's slow."""
    model.eval()
    latencies = []
    # warmup
    item = dataset[0]
    input_ids = item["input_ids"].unsqueeze(0).to(device)
    attn = item["attention_mask"].unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(20):
            _ = model(input_ids=input_ids, attention_mask=attn)
    # measure
    n = min(n_samples, len(dataset))
    with torch.no_grad():
        for i in range(n):
            item = dataset[i]
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            attn = item["attention_mask"].unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_ids=input_ids, attention_mask=attn)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(latencies)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
        "n": int(len(arr)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: DistilBERT on CPU is very slow. Aborting if you proceed without GPU is recommended.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    train_ds = HFTextDataset(train_df, tokenizer, max_len=args.max_len)
    val_ds = HFTextDataset(val_df, tokenizer, max_len=args.max_len)
    test_ds = HFTextDataset(test_df, tokenizer, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES
    ).to(device)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")

    # Class weights
    weights = class_weights_from_train(train_df).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0
    best_state = None
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        n_correct = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            # Apply class weights manually (HF default is unweighted)
            loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * labels.size(0)
            n += labels.size(0)
            n_correct += (out.logits.argmax(dim=1) == labels).sum().item()

        # Val
        model.eval()
        val_correct = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                out = model(input_ids=input_ids, attention_mask=attn)
                val_correct += (out.logits.argmax(dim=1) == labels).sum().item()
                val_n += labels.size(0)
        val_acc = val_correct / val_n
        print(f"Epoch {epoch} | train_loss={total_loss/n:.4f} train_acc={n_correct/n:.4f}"
              f" | val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    t_train = time.perf_counter() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test predictions
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"]
            out = model(input_ids=input_ids, attention_mask=attn)
            all_preds.extend(out.logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    print("Measuring latency (n=500)...")
    latency = measure_latency_hf(model, test_ds, device, n_samples=500)

    save_results(
        model_name="distilbert",
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        latency=latency,
        n_params=n_params,
        training_time_sec=t_train,
        epochs_trained=args.epochs,
        best_val_accuracy=best_val_acc,
        extra={
            "model_name": MODEL_NAME,
            "max_len": args.max_len,
            "epochs": args.epochs,
            "batch_size": args.batch,
        },
    )


if __name__ == "__main__":
    main()
