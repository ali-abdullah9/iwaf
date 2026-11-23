"""
Run the external bypass set through the trained BiLSTM-Attention model.

Usage:
  # Uses the checkpoint written by `train_neural.py --model bilstm_full`
  # (./bilstm_full.pt) OR pass --checkpoint <path>
  python eval_external_bypass.py --checkpoint /path/to/bilstm_checkpoint.pt

Outputs:
  results/external_bypass.json  -- accuracy, per-class metrics, per-source
                                   accuracy, full per-row predictions
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Reuse the EXACT preprocessing from data_utils so behaviour matches training.
# Run from training/baselines/ so data_utils.py and models.py are importable.
from data_utils import (
    LABELS,
    LABEL_TO_IDX,
    META_DIM,
    NUM_CLASSES,
    CharTokenizer,
    build_meta_features,
    build_text_input,
)
from models import BiLSTMAttnFull


# ---------------------------------------------------------------------------
# Dataset wrapping the external CSV
# ---------------------------------------------------------------------------
class ExternalBypassDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.tokenizer = tokenizer
        self.rows = []          # raw rows for reporting
        self.texts = []
        self.metas = []
        self.labels = []

        with open(csv_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                method = r.get("method") or "GET"
                path = r.get("path") or "/"
                query = r.get("query_string") or ""
                body = r.get("body") or ""
                lbl = r["label"]

                text = build_text_input(method, path, query, body)
                meta = build_meta_features(method, path, query, body, text)

                self.rows.append(r)
                self.texts.append(text)
                self.metas.append(meta)
                self.labels.append(LABEL_TO_IDX[lbl])

        self.metas = np.asarray(self.metas, dtype=np.float32)
        self.labels = np.asarray(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return (
            torch.tensor(self.tokenizer.tokenize(self.texts[i]), dtype=torch.long),
            torch.tensor(self.metas[i], dtype=torch.float32),
            torch.tensor(self.labels[i], dtype=torch.long),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="external_bypass_set.csv")
    ap.add_argument("--checkpoint", default="bilstm_full.pt",
                    help="Path to a state_dict .pt produced by train_neural.py --model bilstm_full")
    ap.add_argument("--out", default="results/external_bypass.json")
    args = ap.parse_args()

    # ---- model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = CharTokenizer()
    model = BiLSTMAttnFull(
        vocab_size=tokenizer.vocab_size,
        num_classes=NUM_CLASSES,
        meta_dim=META_DIM,
        pad_idx=tokenizer.pad_idx,
    ).to(device)

    if not os.path.exists(args.checkpoint):
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "After running train_neural.py --model bilstm_full, save the best state with:\n"
            "  torch.save(best_state, 'bilstm_full.pt')\n"
            "(or pass --checkpoint <path>)"
        )
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ---- data ----
    ds = ExternalBypassDataset(args.csv, tokenizer)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    print(f"Loaded {len(ds)} payloads from {args.csv}")

    # ---- inference ----
    all_pred = []
    all_conf = []
    all_probs = []
    with torch.no_grad():
        for tokens, meta, _ in loader:
            tokens = tokens.to(device)
            meta = meta.to(device)
            logits = model(tokens, meta)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            confs = probs.max(axis=1)
            all_pred.extend(preds.tolist())
            all_conf.extend(confs.tolist())
            all_probs.extend(probs.tolist())

    # ---- metrics ----
    y_true = ds.labels
    y_pred = np.array(all_pred)

    overall_acc = float((y_pred == y_true).mean())

    per_class = {}
    for lbl, idx in LABEL_TO_IDX.items():
        mask = y_true == idx
        if mask.sum() == 0:
            continue
        per_class[lbl] = {
            "n": int(mask.sum()),
            "caught": int(((y_pred == idx) & mask).sum()),
            "missed_as": dict(Counter(LABELS[p] for p, t in zip(y_pred, y_true)
                                       if t == idx and p != idx)),
            "recall": float(((y_pred == idx) & mask).sum() / mask.sum()),
        }

    # Per-source accuracy (useful for spotting which collections were too easy / too hard)
    per_source = defaultdict(lambda: {"n": 0, "correct": 0})
    for r, p, t in zip(ds.rows, y_pred, y_true):
        src = r["source"].split("/")[0]
        per_source[src]["n"] += 1
        if p == t:
            per_source[src]["correct"] += 1
    per_source = {k: {"n": v["n"],
                      "correct": v["correct"],
                      "accuracy": v["correct"] / v["n"]}
                  for k, v in per_source.items()}

    # Per-row predictions (so you can investigate misses by hand)
    rows_out = []
    for r, p, t, conf, probs in zip(ds.rows, y_pred, y_true, all_conf, all_probs):
        rows_out.append({
            "method": r["method"],
            "path": r["path"],
            "query_string": r["query_string"],
            "body": r["body"][:200],
            "true_label": LABELS[t],
            "pred_label": LABELS[p],
            "confidence": round(float(conf), 4),
            "correct": bool(p == t),
            "source": r["source"],
            "notes": r["notes"],
            "probs": {LABELS[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)},
        })

    out = {
        "checkpoint": args.checkpoint,
        "csv": args.csv,
        "n_payloads": int(len(ds)),
        "overall_accuracy": overall_acc,
        "per_class": per_class,
        "per_source": per_source,
        "rows": rows_out,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # Summary print
    print("\n" + "=" * 60)
    print("EXTERNAL BYPASS SET RESULTS")
    print("=" * 60)
    print(f"Overall accuracy : {overall_acc:.4f}  ({(y_pred == y_true).sum()}/{len(ds)})")
    print(f"\nPer-class recall:")
    for lbl in ["sqli", "xss", "lfi", "cmdi", "probe"]:
        if lbl in per_class:
            pc = per_class[lbl]
            print(f"  {lbl:8s}: {pc['caught']:>3d}/{pc['n']:<3d}  ({pc['recall']*100:.1f}%)")
            if pc["missed_as"]:
                for as_lbl, cnt in pc["missed_as"].items():
                    print(f"             ↳ {cnt} misclassified as {as_lbl}")
    print(f"\nPer-source accuracy:")
    for src, v in sorted(per_source.items(), key=lambda x: -x[1]["n"]):
        print(f"  {src:25s}: {v['correct']:>3d}/{v['n']:<3d}  ({v['accuracy']*100:.1f}%)")
    print(f"\nFull report → {args.out}")


if __name__ == "__main__":
    main()
