"""
Train one of the neural baselines on identical splits + hyperparameters.

Usage:
  python train_neural.py --model cnn
  python train_neural.py --model lstm_plain
  python train_neural.py --model bilstm_meanpool
  python train_neural.py --model bilstm_nometa
  python train_neural.py --model bilstm_full

Hyperparameters mirror the training setup of training/train_bilstm.py:
  - Adam, lr=1e-3, weight_decay=1e-5
  - batch_size=128
  - up to 25 epochs, early stop patience=4 on val accuracy
  - class-weighted cross-entropy
  - ReduceLROnPlateau (factor=0.5, patience=2)
  - gradient clipping at norm 1.0

This makes the comparison fair across all variants — only the architecture changes.
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data_utils import (
    CharTokenizer,
    IWAFDataset,
    NUM_CLASSES,
    META_DIM,
    class_weights_from_train,
    load_split,
)
from evaluate_common import (
    count_parameters,
    evaluate_neural,
    measure_latency_neural,
    save_results,
)
from models import MODEL_REGISTRY


# ============================================================
# Hyperparameters — IDENTICAL across all variants
# ============================================================
EPOCHS = 25
PATIENCE = 4
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
SEED = 42


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    for tokens, meta, labels in loader:
        tokens = tokens.to(device)
        meta = meta.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(tokens, meta)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        n_correct += (logits.argmax(dim=1) == labels).sum().item()
        n_total += labels.size(0)
    return total_loss / n_total, n_correct / n_total


def eval_loader(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for tokens, meta, labels in loader:
            tokens = tokens.to(device)
            meta = meta.to(device)
            labels = labels.to(device)
            logits = model(tokens, meta)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            n_correct += (logits.argmax(dim=1) == labels).sum().item()
            n_total += labels.size(0)
    return total_loss / n_total, n_correct / n_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    # Data
    tokenizer = CharTokenizer()
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    train_ds = IWAFDataset(train_df, tokenizer)
    val_ds = IWAFDataset(val_df, tokenizer)
    test_ds = IWAFDataset(test_df, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    # Model
    ModelCls = MODEL_REGISTRY[args.model]
    if args.model == "bilstm_nometa":
        model = ModelCls(
            vocab_size=tokenizer.vocab_size,
            num_classes=NUM_CLASSES,
            pad_idx=tokenizer.pad_idx,
        ).to(device)
    else:
        model = ModelCls(
            vocab_size=tokenizer.vocab_size,
            num_classes=NUM_CLASSES,
            meta_dim=META_DIM,
            pad_idx=tokenizer.pad_idx,
        ).to(device)

    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")

    # Loss + optimizer
    weights = class_weights_from_train(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # Train
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_loader(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
              f" | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best val_acc={best_val_acc:.4f})")
                break
    t_train = time.perf_counter() - t_start

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, f"{args.model}.pt") 

    # Test eval
    print("\nEvaluating on test set...")
    y_true, y_pred = evaluate_neural(model, test_loader, device)

    # Latency (single-sample, mirrors ml-service/benchmark_latency.py)
    print("Measuring latency (n=2000)...")
    test_loader_lat = DataLoader(test_ds, batch_size=args.batch, shuffle=False)
    latency = measure_latency_neural(model, test_loader_lat, device, n_samples=2000)

    # Save
    save_results(
        model_name=args.model,
        y_true=y_true,
        y_pred=y_pred,
        latency=latency,
        n_params=n_params,
        training_time_sec=t_train,
        epochs_trained=epoch,
        best_val_accuracy=best_val_acc,
    )


if __name__ == "__main__":
    main()
