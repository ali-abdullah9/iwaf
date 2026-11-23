"""
Shared evaluation utilities. Every baseline produces results in identical format
so the comparison table is straightforward.

Output JSON schema (results/<model>.json):
{
  "model": "bilstm_full",
  "params": 635847,
  "test_accuracy": 0.9849,
  "macro_f1": 0.978,
  "weighted_f1": 0.985,
  "per_class": {
    "benign": {"precision": 0.99, "recall": 1.00, "f1": 0.99, "support": 1607},
    ...
  },
  "confusion_matrix": [[...], [...], ...],
  "latency_ms": {"p50": 6.08, "p95": 11.29, "p99": 15.27, "mean": 6.68},
  "training_time_sec": 412.3,
  "epochs_trained": 18,
  "best_val_accuracy": 0.984
}
"""

import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from data_utils import IDX_TO_LABEL, LABELS

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_neural(model, dataloader, device):
    """Evaluate a torch model on a DataLoader. Returns y_true, y_pred."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            tokens, meta, labels = batch
            tokens = tokens.to(device)
            meta = meta.to(device)
            logits = model(tokens, meta)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def measure_latency_neural(model, dataloader, device, n_samples=2000):
    """Measure single-request latency. Mirrors ml-service/benchmark_latency.py methodology."""
    model.eval()
    latencies = []
    n_done = 0
    # Warmup
    with torch.no_grad():
        for batch in dataloader:
            tokens, meta, _ = batch
            tokens = tokens[:1].to(device)
            meta = meta[:1].to(device)
            for _ in range(20):
                _ = model(tokens, meta)
            break
    # Measure
    with torch.no_grad():
        for batch in dataloader:
            tokens, meta, _ = batch
            for i in range(tokens.size(0)):
                if n_done >= n_samples:
                    break
                t = tokens[i : i + 1].to(device)
                m = meta[i : i + 1].to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(t, m)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000.0)
                n_done += 1
            if n_done >= n_samples:
                break
    arr = np.array(latencies)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
        "n": len(arr),
    }


def save_results(
    model_name,
    y_true,
    y_pred,
    latency=None,
    n_params=None,
    training_time_sec=None,
    epochs_trained=None,
    best_val_accuracy=None,
    extra=None,
):
    """Save standardized results JSON."""
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    per_class = {}
    for cls in LABELS:
        if cls in report:
            per_class[cls] = {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS)))).tolist()

    out = {
        "model": model_name,
        "test_accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "label_order": LABELS,
    }
    if n_params is not None:
        out["params"] = int(n_params)
    if latency is not None:
        out["latency_ms"] = latency
    if training_time_sec is not None:
        out["training_time_sec"] = float(training_time_sec)
    if epochs_trained is not None:
        out["epochs_trained"] = int(epochs_trained)
    if best_val_accuracy is not None:
        out["best_val_accuracy"] = float(best_val_accuracy)
    if extra:
        out["extra"] = extra

    path = os.path.join(RESULTS_DIR, f"{model_name}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {path}")
    print(f"  test_accuracy : {accuracy:.4f}")
    print(f"  macro_f1      : {macro_f1:.4f}")
    if latency:
        print(f"  latency p99   : {latency['p99']:.2f} ms")
    return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
