"""
train_gcn_real.py — GCN re-training with real IPs/timestamps.

Runs THREE training conditions on the SAME architecture so we can attribute
the learned relation weights honestly:

  Condition A: synthetic IPs/timestamps (replicates the original GCN training setup)
  Condition B: real IPs/timestamps from Zenodo audit logs
  Condition C: real IPs/timestamps + class-blind contrastive signal
               (positives = same-IP rather than same-class)

Why three? In the original training, synthetic IPs were generated to cluster
by class AND the contrastive loss used class labels as positive pairs. That
gives same_source_ip TWO sources of signal pointing the same way, which may
explain α=0.317 winning. By running B (real IPs, class-supervised loss) and
C (real IPs, IP-supervised loss), we isolate which source of signal is
producing the result.

Inputs
------
  artifacts/bilstm.pt              trained BiLSTM checkpoint (the real one,
                                   see ml-service/artifacts/)
  artifacts/tokenizer.pkl
  artifacts/label_encoder.pkl
  artifacts/model_info.json
  zenodo_full_30day.csv            parsed Zenodo data with real IPs/timestamps
  unified_v2/train.csv             original training data (for Condition A)

Output
------
  results_gcn_real/
    condition_A_synthetic.json     learned weights + clustering metrics
    condition_B_real.json
    condition_C_ip_positives.json
    summary.json                   side-by-side comparison

Run
---
  python train_gcn_real.py --zenodo zenodo_full_30day.csv \
                           --train-csv unified_v2/train.csv \
                           --bilstm-dir artifacts \
                           --out results_gcn_real
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from bilstm_model import IWAFBiLSTM
from embedding_extractor import extract_embeddings
from gcn_model import (
    ContrastiveLoss,
    MultiRelationalGCN,
    build_multi_relational_graph,
)


CONFIG = {
    "gcn_input_dim": 256,
    "gcn_hidden_dim": 128,
    "gcn_embedding_dim": 64,
    "num_relations": 4,
    "dropout": 0.3,
    "batch_size": 256,
    "epochs": 30,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "temperature": 0.3,
    "similarity_threshold": 0.7,
    "time_window_minutes": 30,
    "samples_per_class": 800,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

RELATION_NAMES = ["content_similarity", "same_source_ip", "temporal_proximity", "same_path"]


# ---------------------------------------------------------------------------
# Data preparation for each condition
# ---------------------------------------------------------------------------
def prep_synthetic(train_csv, label_encoder):
    """Condition A: original synthetic IPs + timestamps. Replicates the original training setup."""
    df = pd.read_csv(train_csv).fillna("")
    samples = []
    for cls in label_encoder.classes_:
        cls_df = df[df["label"] == cls]
        n = min(len(cls_df), CONFIG["samples_per_class"])
        if n == 0:
            continue
        samples.append(cls_df.sample(n=n, random_state=42))
    df = pd.concat(samples, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # SAME synthesis as original train_gcn.py lines 110-141
    np.random.seed(42)
    df["source_ip"] = "0.0.0.0"
    for cls in label_encoder.classes_:
        idx = df[df["label"] == cls].index
        n_attackers = max(1, len(idx) // 50)
        ips = [f"10.0.{i}.{np.random.randint(1, 255)}" for i in range(n_attackers)]
        for i in idx:
            df.at[i, "source_ip"] = np.random.choice(ips)
    base_unix = 1743465600
    timestamps = []
    for cls in label_encoder.classes_:
        idx = df[df["label"] == cls].index.tolist()
        offsets = np.cumsum(np.random.normal(0, 30, size=len(idx))).astype(int)
        for j, i in enumerate(idx):
            timestamps.append((i, base_unix + int(offsets[j]) * 60))
    timestamp_dict = dict(timestamps)
    df["timestamp_unix"] = df.index.map(lambda i: timestamp_dict.get(i, base_unix))
    df["timestamp"] = pd.to_datetime(df["timestamp_unix"], unit="s", errors="coerce")
    return df


def prep_real(zenodo_csv, label_encoder):
    """Condition B & C: real IPs/timestamps from Zenodo. Same per-class sample size."""
    df = pd.read_csv(zenodo_csv, low_memory=False).fillna("")
    samples = []
    for cls in label_encoder.classes_:
        cls_df = df[df["label"] == cls]
        n = min(len(cls_df), CONFIG["samples_per_class"])
        if n == 0:
            print(f"  WARNING: no rows for class '{cls}' in Zenodo data, skipping.")
            continue
        samples.append(cls_df.sample(n=n, random_state=42))
    out = pd.concat(samples, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(gcn, embeddings, labels, requests_df, optimizer, loss_fn, ip_supervised=False):
    """One epoch. If ip_supervised, positives = same source_ip instead of same class."""
    gcn.train()
    n = len(embeddings)
    indices = np.random.permutation(n)
    epoch_loss = 0.0
    n_batches = 0

    if ip_supervised:
        # Map IPs to integer pseudo-labels
        ip_to_id = {ip: i for i, ip in enumerate(requests_df["source_ip"].unique())}
        ip_ids = np.array([ip_to_id[ip] for ip in requests_df["source_ip"].values])

    for start in range(0, n, CONFIG["batch_size"]):
        batch_idx = indices[start:start + CONFIG["batch_size"]]
        if len(batch_idx) < 4:
            continue

        batch_emb = embeddings[batch_idx]
        batch_df = requests_df.iloc[batch_idx].reset_index(drop=True)
        if ip_supervised:
            batch_supervision = ip_ids[batch_idx]
        else:
            batch_supervision = labels[batch_idx]

        adj_list = build_multi_relational_graph(
            batch_emb, batch_df,
            similarity_threshold=CONFIG["similarity_threshold"],
            time_window_minutes=CONFIG["time_window_minutes"],
        )

        x = torch.tensor(batch_emb, dtype=torch.float32, device=CONFIG["device"])
        adj_tensors = [torch.tensor(a, dtype=torch.float32, device=CONFIG["device"]) for a in adj_list]
        sup_t = torch.tensor(batch_supervision, dtype=torch.long, device=CONFIG["device"])

        optimizer.zero_grad()
        gcn_emb = gcn(x, adj_tensors)
        loss = loss_fn(gcn_emb, sup_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gcn.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    return epoch_loss / max(1, n_batches)


def evaluate_clustering(gcn, embeddings, labels, requests_df):
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        homogeneity_score, normalized_mutual_info_score, silhouette_score,
    )
    gcn.eval()
    adj_list = build_multi_relational_graph(
        embeddings, requests_df,
        similarity_threshold=CONFIG["similarity_threshold"],
        time_window_minutes=CONFIG["time_window_minutes"],
    )
    x = torch.tensor(embeddings, dtype=torch.float32, device=CONFIG["device"])
    adj_tensors = [torch.tensor(a, dtype=torch.float32, device=CONFIG["device"]) for a in adj_list]
    with torch.no_grad():
        gcn_emb = gcn(x, adj_tensors).cpu().numpy()
    n_classes = len(np.unique(labels))
    if n_classes < 2:
        return {"silhouette": 0.0, "homogeneity": 0.0, "nmi": 0.0}
    km = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    cl = km.fit_predict(gcn_emb)
    return {
        "silhouette": float(silhouette_score(gcn_emb, cl)),
        "homogeneity": float(homogeneity_score(labels, cl)),
        "nmi": float(normalized_mutual_info_score(labels, cl)),
    }


def run_condition(name, df, embeddings, labels, ip_supervised=False):
    """Train one condition and return results dict."""
    print(f"\n{'='*60}\nCondition: {name}\n{'='*60}")
    print(f"  Samples: {len(df):,}")
    print(f"  Unique source IPs: {df['source_ip'].nunique()}")
    print(f"  Class distribution: {dict(pd.Series(labels).value_counts().sort_index())}")
    print(f"  Supervision: {'same source_ip' if ip_supervised else 'same class label'}")

    gcn = MultiRelationalGCN(
        input_dim=CONFIG["gcn_input_dim"],
        hidden_dim=CONFIG["gcn_hidden_dim"],
        embedding_dim=CONFIG["gcn_embedding_dim"],
        num_relations=CONFIG["num_relations"],
        dropout=CONFIG["dropout"],
    ).to(CONFIG["device"])

    optimizer = torch.optim.Adam(
        gcn.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]
    )
    loss_fn = ContrastiveLoss(temperature=CONFIG["temperature"])

    initial_metrics = evaluate_clustering(gcn, embeddings, labels, df)
    print(f"  Untrained clustering: silh={initial_metrics['silhouette']:.4f} hom={initial_metrics['homogeneity']:.4f} nmi={initial_metrics['nmi']:.4f}")

    history = []
    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()
        loss = train_one_epoch(gcn, embeddings, labels, df, optimizer, loss_fn, ip_supervised=ip_supervised)
        if epoch % 5 == 0 or epoch == CONFIG["epochs"]:
            m = evaluate_clustering(gcn, embeddings, labels, df)
            print(f"  Epoch {epoch:2d}/{CONFIG['epochs']:2d} | loss={loss:.4f} | silh={m['silhouette']:.4f} hom={m['homogeneity']:.4f} nmi={m['nmi']:.4f} | {time.time()-t0:.1f}s")
            history.append({"epoch": epoch, "loss": loss, **m})

    final_metrics = evaluate_clustering(gcn, embeddings, labels, df)
    weights = gcn.get_relation_weights().tolist()

    print(f"\n  Final learned relation weights:")
    for rname, w in zip(RELATION_NAMES, weights):
        bar = "█" * int(w * 60)
        print(f"    {rname:25s} {w:.4f} {bar}")

    return {
        "condition": name,
        "n_samples": int(len(df)),
        "n_unique_ips": int(df["source_ip"].nunique()),
        "ip_supervised": ip_supervised,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "relation_weights": dict(zip(RELATION_NAMES, weights)),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenodo", default="zenodo_full_30day.csv")
    ap.add_argument("--train-csv", default="unified_v2/train.csv")
    ap.add_argument("--bilstm-dir", default="artifacts")
    ap.add_argument("--out", default="results_gcn_real")
    ap.add_argument("--skip-A", action="store_true", help="Skip Condition A (synthetic)")
    ap.add_argument("--skip-B", action="store_true", help="Skip Condition B (real, class-sup)")
    ap.add_argument("--skip-C", action="store_true", help="Skip Condition C (real, IP-sup)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Device: {CONFIG['device']}")
    print(f"Output dir: {args.out}")

    # --- Load BiLSTM ---
    print("\nLoading BiLSTM artifacts...")
    bilstm_dir = Path(args.bilstm_dir)
    with open(bilstm_dir / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open(bilstm_dir / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open(bilstm_dir / "model_info.json") as f:
        info = json.load(f)
    cfg = info["config"]
    bilstm = IWAFBiLSTM(
        vocab_size=tokenizer.vocab_size,
        num_classes=info["num_classes"],
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        meta_features_dim=cfg["meta_features_dim"],
        dropout=cfg["dropout"],
        pad_idx=tokenizer.pad_idx,
    )
    bilstm.load_state_dict(torch.load(bilstm_dir / "bilstm.pt", map_location=CONFIG["device"], weights_only=True))
    bilstm = bilstm.to(CONFIG["device"]).eval()
    print(f"  ✓ BiLSTM loaded ({sum(p.numel() for p in bilstm.parameters()):,} params)")
    print(f"  ✓ Classes: {list(label_encoder.classes_)}")

    results = {}

    # --- Condition A: synthetic ---
    if not args.skip_A:
        df_a = prep_synthetic(args.train_csv, label_encoder)
        emb_a = extract_embeddings(bilstm, tokenizer, df_a, device=CONFIG["device"], batch_size=64)
        labels_a = label_encoder.transform(df_a["label"].values)
        results["A_synthetic"] = run_condition("A · synthetic IPs/timestamps (replicates original setup)",
                                                df_a, emb_a, labels_a, ip_supervised=False)
        with open(Path(args.out) / "condition_A_synthetic.json", "w") as f:
            json.dump(results["A_synthetic"], f, indent=2)

    # --- Condition B: real, class-supervised ---
    if not args.skip_B:
        df_b = prep_real(args.zenodo, label_encoder)
        emb_b = extract_embeddings(bilstm, tokenizer, df_b, device=CONFIG["device"], batch_size=64)
        labels_b = label_encoder.transform(df_b["label"].values)
        results["B_real_class_sup"] = run_condition("B · real IPs/timestamps, class-supervised loss",
                                                     df_b, emb_b, labels_b, ip_supervised=False)
        with open(Path(args.out) / "condition_B_real.json", "w") as f:
            json.dump(results["B_real_class_sup"], f, indent=2)

    # --- Condition C: real, IP-supervised ---
    if not args.skip_C:
        # Reuse df_b/emb_b if available, else build
        if "B_real_class_sup" not in results:
            df_c = prep_real(args.zenodo, label_encoder)
            emb_c = extract_embeddings(bilstm, tokenizer, df_c, device=CONFIG["device"], batch_size=64)
            labels_c = label_encoder.transform(df_c["label"].values)
        else:
            df_c, emb_c, labels_c = df_b, emb_b, labels_b
        results["C_real_ip_sup"] = run_condition("C · real IPs/timestamps, IP-supervised loss (no class info)",
                                                  df_c, emb_c, labels_c, ip_supervised=True)
        with open(Path(args.out) / "condition_C_ip_positives.json", "w") as f:
            json.dump(results["C_real_ip_sup"], f, indent=2)

    # --- Summary ---
    summary = {"conditions": list(results.keys()), "results": results, "config": CONFIG}
    summary["config"]["device"] = str(CONFIG["device"])
    with open(Path(args.out) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Print side-by-side ---
    print("\n" + "=" * 80)
    print("LEARNED RELATION WEIGHTS — side by side")
    print("=" * 80)
    print(f"{'Relation':<22} " + " ".join(f"{k:>12s}" for k in results))
    for rname in RELATION_NAMES:
        row = f"{rname:<22} "
        for k in results:
            row += f"{results[k]['relation_weights'][rname]:>12.4f} "
        print(row)

    print("\n" + "=" * 80)
    print("CLUSTERING QUALITY — side by side")
    print("=" * 80)
    print(f"{'Metric':<15} " + " ".join(f"{k:>12s}" for k in results))
    for m in ["silhouette", "homogeneity", "nmi"]:
        row = f"{m:<15} "
        for k in results:
            row += f"{results[k]['final_metrics'][m]:>12.4f} "
        print(row)

    print(f"\n✓ Results written to {args.out}/")


if __name__ == "__main__":
    main()
