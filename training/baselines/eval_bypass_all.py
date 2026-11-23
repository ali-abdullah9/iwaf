"""
Run external_bypass_set.csv through ALL trained model checkpoints and compare.

Prereqs: each .pt was saved by train_neural.py (it writes <model>.pt
after training). LinearSVC isn't a torch
model so we re-fit it on train_split.csv (fast, ~10s) and predict.

Usage:
  python eval_bypass_all.py
"""

import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_utils import (
    LABELS, LABEL_TO_IDX, META_DIM, NUM_CLASSES,
    CharTokenizer, build_text_input, load_split,
)
from eval_external_bypass import ExternalBypassDataset
from models import (
    BiLSTMAttnFull, BiLSTMAttnNoMeta, BiLSTMMeanPool, CharCNN, PlainLSTM,
)

# Torch checkpoints + their model class
TORCH_MODELS = {
    "bilstm_full":      (BiLSTMAttnFull,    True),
    "bilstm_nometa":    (BiLSTMAttnNoMeta,  False),
    "bilstm_meanpool":  (BiLSTMMeanPool,    True),
    "lstm_plain":       (PlainLSTM,         True),
    "cnn":              (CharCNN,           True),
}

CSV = "external_bypass_set.csv"


def eval_torch(name, ModelCls, uses_meta, device, tokenizer, ds):
    ckpt = f"{name}.pt"
    if not os.path.exists(ckpt):
        return None, f"missing {ckpt}"
    if uses_meta:
        m = ModelCls(vocab_size=tokenizer.vocab_size, num_classes=NUM_CLASSES,
                     meta_dim=META_DIM, pad_idx=tokenizer.pad_idx).to(device)
    else:
        m = ModelCls(vocab_size=tokenizer.vocab_size, num_classes=NUM_CLASSES,
                     pad_idx=tokenizer.pad_idx).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    m.eval()

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for tok, meta, _ in loader:
            tok, meta = tok.to(device), meta.to(device)
            preds.extend(m(tok, meta).argmax(1).cpu().numpy().tolist())
    return np.asarray(preds), None


def eval_linearsvc(ds_rows, ds_labels):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    train_df = load_split("train")
    train_texts = [build_text_input(r["method"], r["path"],
                                    r.get("query_string", "") or "",
                                    r.get("body", "") or "")
                   for _, r in train_df.iterrows()]
    train_y = np.asarray([LABEL_TO_IDX[l] for l in train_df["label"]])
    test_texts = [build_text_input(r["method"], r["path"],
                                   r.get("query_string", "") or "",
                                   r.get("body", "") or "")
                  for r in ds_rows]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          max_features=50000, sublinear_tf=True, min_df=2)
    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(test_texts)
    clf = LinearSVC(class_weight="balanced", C=1.0, max_iter=5000, dual="auto")
    clf.fit(Xtr, train_y)
    return clf.predict(Xte)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer()
    ds = ExternalBypassDataset(CSV, tokenizer)
    y_true = ds.labels
    print(f"Loaded {len(ds)} payloads — comparing models...\n")

    rows = []  # for the summary table

    # Torch models
    for name, (Cls, uses_meta) in TORCH_MODELS.items():
        t0 = time.perf_counter()
        preds, err = eval_torch(name, Cls, uses_meta, device, tokenizer, ds)
        if err:
            print(f"  {name:18s} → SKIP ({err})")
            continue
        acc = float((preds == y_true).mean())
        per_class = {LABELS[i]: float(((preds == i) & (y_true == i)).sum()
                                      / max(1, (y_true == i).sum()))
                     for i in range(NUM_CLASSES) if (y_true == i).sum() > 0}
        rows.append({"model": name, "accuracy": acc, "per_class": per_class,
                     "n_correct": int((preds == y_true).sum())})
        print(f"  {name:18s} → {acc*100:5.2f}%   ({(preds == y_true).sum()}/{len(ds)})")

    # LinearSVC
    print(f"  {'linearsvc':18s} → ...refitting...", end="\r")
    t0 = time.perf_counter()
    preds = eval_linearsvc(ds.rows, y_true)
    acc = float((preds == y_true).mean())
    per_class = {LABELS[i]: float(((preds == i) & (y_true == i)).sum()
                                  / max(1, (y_true == i).sum()))
                 for i in range(NUM_CLASSES) if (y_true == i).sum() > 0}
    rows.append({"model": "linearsvc", "accuracy": acc, "per_class": per_class,
                 "n_correct": int((preds == y_true).sum())})
    print(f"  {'linearsvc':18s} → {acc*100:5.2f}%   ({(preds == y_true).sum()}/{len(ds)})  (refit took {time.perf_counter()-t0:.1f}s)")

    # Save + print table
    os.makedirs("results", exist_ok=True)
    with open("results/external_bypass_all_models.json", "w") as f:
        json.dump(rows, f, indent=2)

    print("\n" + "=" * 70)
    print(f"{'MODEL':<20} {'OVERALL':>8} {'sqli':>7} {'xss':>7} {'lfi':>7} {'cmdi':>7} {'probe':>7}")
    print("=" * 70)
    rows.sort(key=lambda r: -r["accuracy"])
    for r in rows:
        pc = r["per_class"]
        print(f"{r['model']:<20} {r['accuracy']*100:>7.2f}% "
              f"{pc.get('sqli',0)*100:>6.1f}% {pc.get('xss',0)*100:>6.1f}% "
              f"{pc.get('lfi',0)*100:>6.1f}% {pc.get('cmdi',0)*100:>6.1f}% "
              f"{pc.get('probe',0)*100:>6.1f}%")
    print("\nSaved → results/external_bypass_all_models.json")


if __name__ == "__main__":
    main()
