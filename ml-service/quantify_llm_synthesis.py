"""
quantify_llm_synthesis.py — Convert "we have one DRE example" into real numbers.

What this answers:

  Q1. How many rules does the DRE generate per analysis run, on average?
  Q2. What fraction does the LLM decline / produce invalid output for?
  Q3. What fraction passes the FP-Guard's SAFE/ACCEPTABLE bar?
  Q4. Does the LLM beat a simple heuristic baseline? On what dimensions?
  Q5. Do approved rules actually have low FP rates on a HELD-OUT benign set
      (different from the 500 the FP-Guard already saw)?

How it works
------------
1. Build N synthetic "time windows" by sampling N batches of rows from the
   parsed Zenodo dataset. Each window mimics a `fetch_blocked_requests` call.
2. For each window, run the DRE end-to-end TWICE on the SAME clusters:
     LLM path : llm_analyzer.AnthropicProvider
     Heuristic: llm_analyzer.HeuristicProvider
   This isolates the LLM contribution from the rest of the pipeline.
3. Score every produced rule along four axes:
     - Validity (parses as regex, rule_text is well-formed)
     - FP rate on the dev benign set (the 500 the FP-Guard saw)
     - FP rate on a HELD-OUT benign set (separately sampled, never seen)
     - Confidence after FP-Guard adjustment

Inputs
------
  ml-service/             this folder (the script imports its modules)
  zenodo_full_30day.csv   parsed Zenodo (for blocked-request samples)
  ANTHROPIC_API_KEY       env var, for the LLM path
  unified_v2/train.csv    benign samples for FP-Guard (already used)

Run
---
  cd ml-service
  # zenodo_full_30day.csv is produced by training/parse_modsec_audit.py
  # ANTHROPIC_API_KEY is read from the environment or the project .env file
  python quantify_llm_synthesis.py --runs 10 --window-size 100

Output
------
  results_llm/run_log.json     full per-run details
  results_llm/summary.json     aggregate metrics
"""

import argparse
import json
import os
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()  # picks up ANTHROPIC_API_KEY from the project .env file
from bilstm_model import IWAFBiLSTM
from embedding_extractor import extract_embeddings
from llm_analyzer import AnthropicProvider, HeuristicProvider, build_secrule_from_llm
from fp_guard import FalsePositiveGuard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_bilstm(bilstm_dir, device):
    """Load the trained BiLSTM + tokenizer + label encoder."""
    bilstm_dir = Path(bilstm_dir)
    with open(bilstm_dir / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open(bilstm_dir / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open(bilstm_dir / "model_info.json") as f:
        info = json.load(f)
    cfg = info["config"]
    model = IWAFBiLSTM(
        vocab_size=tokenizer.vocab_size,
        num_classes=info["num_classes"],
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        meta_features_dim=cfg["meta_features_dim"],
        dropout=cfg["dropout"],
        pad_idx=tokenizer.pad_idx,
    )
    model.load_state_dict(torch.load(bilstm_dir / "bilstm.pt", map_location=device, weights_only=True))
    model = model.to(device).eval()
    return model, tokenizer, label_encoder


def cluster_kmeans(embeddings, max_k=10):
    """Simple silhouette-selected k-means (mirrors dre_v2.cluster())."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    n = len(embeddings)
    if n < 4:
        return np.arange(n)
    upper = min(max_k, n // 2)
    best_k, best_score = 2, -1
    for k in range(2, upper + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl = km.fit_predict(embeddings)
            if len(set(lbl)) > 1:
                s = silhouette_score(embeddings, lbl)
                if s > best_score:
                    best_score, best_k = s, k
        except Exception:
            continue
    return KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(embeddings)


def build_cluster_summary(cluster_id, cluster_df):
    """Mirror of dre_v2.build_cluster_summary() but using `label` not `attack_type`."""
    attack_types = Counter(cluster_df["label"])
    primary = attack_types.most_common(1)[0][0] if attack_types else "unknown"
    samples = cluster_df.head(8)[["method", "path", "query_string", "body"]].to_dict("records")
    return {
        "cluster_id": int(cluster_id),
        "cluster_size": int(len(cluster_df)),
        "primary_attack_type": primary,
        "attack_distribution": dict(attack_types),
        "method_distribution": dict(Counter(cluster_df["method"])),
        "unique_ips": int(cluster_df["source_ip"].nunique()) if "source_ip" in cluster_df else 0,
        "top_source_ips": dict(Counter(cluster_df.get("source_ip", [])).most_common(5)),
        "sample_requests": samples,
    }


def score_rule(rule, fp_guard_dev, fp_guard_holdout):
    """Score one rule on validity + dev FP + holdout FP."""
    pattern = rule.get("regex_pattern", "")
    rule_text = rule.get("rule_text", "")
    targets = rule.get("targets", "ARGS|REQUEST_BODY")

    out = {
        "rule_id": rule.get("rule_id"),
        "cluster_size": rule.get("cluster_size"),
        "attack_type": rule.get("attack_type"),
        "technique_name": rule.get("technique_name", ""),
        "regex_pattern": pattern,
        "rule_text_present": bool(rule_text and "SecRule" in rule_text),
        "confidence_after_fpguard": rule.get("confidence", "UNKNOWN"),
    }

    # 1. Validity: does the regex compile?
    try:
        re.compile(pattern)
        out["regex_valid"] = bool(pattern)
    except re.error:
        out["regex_valid"] = False

    # 2. FP rate on the DEV benign set (FP-Guard saw this)
    dev_eval = fp_guard_dev.evaluate_pattern(pattern, targets=targets) if pattern else None
    out["fp_rate_dev"] = dev_eval["false_positive_rate"] if dev_eval else None
    out["fp_verdict_dev"] = dev_eval["verdict"] if dev_eval else "NO_PATTERN"

    # 3. FP rate on the HELD-OUT benign set (FP-Guard never saw this)
    holdout_eval = fp_guard_holdout.evaluate_pattern(pattern, targets=targets) if pattern else None
    out["fp_rate_holdout"] = holdout_eval["false_positive_rate"] if holdout_eval else None
    out["fp_verdict_holdout"] = holdout_eval["verdict"] if holdout_eval else "NO_PATTERN"

    return out


# ---------------------------------------------------------------------------
# One DRE run on one window, with a given LLM provider
# ---------------------------------------------------------------------------
def run_one_window(window_df, embeddings, llm_provider, fp_guard_dev, fp_guard_holdout,
                   run_id, window_id):
    """Cluster, ask provider per cluster, score each produced rule. Returns list of rule scores."""
    if len(window_df) < 4:
        return []

    cluster_labels = cluster_kmeans(embeddings)
    window_df = window_df.copy()
    window_df["cluster_id"] = cluster_labels

    rule_scores = []
    rule_id_base = 9_000_000 + run_id * 1000 + window_id * 100

    for cid in sorted(set(cluster_labels)):
        cdf = window_df[window_df["cluster_id"] == cid]
        if len(cdf) < 2:
            continue                                  # skip singletons (mirrors dre_v2)
        summary = build_cluster_summary(cid, cdf)

        t0 = time.time()
        analysis = llm_provider.analyze_cluster(summary)
        latency = time.time() - t0

        if analysis is None:
            rule_scores.append({
                "rule_id": rule_id_base + int(cid),
                "cluster_size": summary["cluster_size"],
                "attack_type": summary["primary_attack_type"],
                "provider": type(llm_provider).__name__,
                "outcome": "declined",
                "latency_sec": latency,
            })
            continue

        # Build SecRule and run FP-Guard (mirrors dre_v2.analyze)
        rule_id = rule_id_base + int(cid)
        try:
            rule_text = build_secrule_from_llm(summary, analysis, rule_id)
        except Exception as e:
            rule_text = ""

        rule = {
            "rule_id": rule_id,
            "cluster_size": summary["cluster_size"],
            "attack_type": summary["primary_attack_type"],
            "technique_name": analysis.get("technique_name", ""),
            "regex_pattern": analysis.get("regex_pattern", ""),
            "targets": analysis.get("rule_targets", "ARGS|REQUEST_BODY"),
            "rule_text": rule_text,
            "confidence": "MEDIUM",          # placeholder; FP-Guard adjusts below
        }

        if rule["regex_pattern"]:
            fp_eval = fp_guard_dev.evaluate_pattern(rule["regex_pattern"], targets=rule["targets"])
            rule = fp_guard_dev.adjust_rule_confidence(rule, fp_eval)

        scored = score_rule(rule, fp_guard_dev, fp_guard_holdout)
        scored["provider"] = type(llm_provider).__name__
        scored["outcome"] = "produced"
        scored["latency_sec"] = round(latency, 3)
        rule_scores.append(scored)

    return rule_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenodo", default="zenodo_full_30day.csv")
    ap.add_argument("--train-csv", default="unified_v2/train.csv")
    ap.add_argument("--bilstm-dir", default="artifacts")
    ap.add_argument("--out", default="results_llm")
    ap.add_argument("--runs", type=int, default=10,
                    help="Number of DRE invocations (windows). Default 10.")
    ap.add_argument("--window-size", type=int, default=100,
                    help="Blocked requests per window. Default 100.")
    ap.add_argument("--n-benign-dev", type=int, default=500,
                    help="Benign samples FP-Guard sees (matches the DRE default). Default 500.")
    ap.add_argument("--n-benign-holdout", type=int, default=500,
                    help="Held-out benign samples for true FP rate. Default 500.")
    ap.add_argument("--skip-llm", action="store_true",
                    help="Skip the LLM provider (e.g. if no API key, just validate harness).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Runs: {args.runs}, window size: {args.window_size}")

    # --- Load BiLSTM ---
    print("\nLoading BiLSTM...")
    bilstm, tokenizer, _ = load_bilstm(args.bilstm_dir, device)
    print(f"  ✓ {sum(p.numel() for p in bilstm.parameters()):,} params")

    # --- Build FP-Guards (dev + holdout) ---
    print(f"\nLoading benign samples for FP-Guards...")
    train_df = pd.read_csv(args.train_csv).fillna("")
    benign_df = train_df[train_df["label"] == "benign"]
    if len(benign_df) < args.n_benign_dev + args.n_benign_holdout:
        sys.exit(f"Not enough benign samples: have {len(benign_df)}, need {args.n_benign_dev + args.n_benign_holdout}")

    # Deterministic split: dev = first random sample, holdout = different random sample
    dev = benign_df.sample(n=args.n_benign_dev, random_state=42)
    remaining = benign_df.drop(dev.index)
    holdout = remaining.sample(n=args.n_benign_holdout, random_state=99)

    fp_guard_dev = FalsePositiveGuard(
        benign_samples=dev[["method", "path", "query_string", "body"]].to_dict("records"),
        max_fp_rate=0.005,
    )
    fp_guard_holdout = FalsePositiveGuard(
        benign_samples=holdout[["method", "path", "query_string", "body"]].to_dict("records"),
        max_fp_rate=0.005,
    )
    print(f"  ✓ Dev FP-Guard: {len(fp_guard_dev.benign_samples)} samples")
    print(f"  ✓ Holdout FP-Guard: {len(fp_guard_holdout.benign_samples)} samples (disjoint)")

    # --- Load Zenodo, take only blocked-class rows (= rule-firing transactions) ---
    print(f"\nLoading Zenodo data...")
    zdf = pd.read_csv(args.zenodo, low_memory=False).fillna("")
    # We want attack-class rows only (mimics what fetch_blocked_requests would return)
    attack_df = zdf[zdf["label"].isin(["sqli", "xss", "lfi", "cmdi", "probe"])]
    print(f"  ✓ {len(attack_df):,} attack-class rows available")

    # --- Init providers ---
    providers = []
    if not args.skip_llm:
        ant = AnthropicProvider()
        if ant.is_available():
            providers.append(("Anthropic", ant))
            print(f"  ✓ Anthropic provider ready")
        else:
            print("  ⚠ ANTHROPIC_API_KEY not set or anthropic package missing — skipping LLM path")
    providers.append(("Heuristic", HeuristicProvider()))
    print(f"  ✓ Heuristic provider ready")

    # --- Run windows ---
    all_run_results = []
    rng = np.random.default_rng(seed=2026)
    for run_id in range(args.runs):
        # Sample a window without replacement (per run); seed-controlled
        sample_idx = rng.choice(len(attack_df), size=min(args.window_size, len(attack_df)), replace=False)
        window_df = attack_df.iloc[sample_idx].reset_index(drop=True)

        # Embeddings — computed ONCE per window, reused across providers
        emb = extract_embeddings(bilstm, tokenizer, window_df, device=device, batch_size=64)

        for prov_name, prov in providers:
            print(f"\nRun {run_id+1}/{args.runs} · provider={prov_name} ...", flush=True)
            rules = run_one_window(window_df, emb, prov, fp_guard_dev, fp_guard_holdout,
                                   run_id=run_id, window_id=0)
            all_run_results.append({
                "run_id": run_id,
                "provider": prov_name,
                "window_size": int(len(window_df)),
                "rules": rules,
            })
            n_prod = sum(1 for r in rules if r.get("outcome") == "produced")
            n_decl = sum(1 for r in rules if r.get("outcome") == "declined")
            print(f"  → {len(rules)} cluster(s); {n_prod} rules produced, {n_decl} declined")

    # --- Save raw run log ---
    with open(Path(args.out) / "run_log.json", "w") as f:
        json.dump(all_run_results, f, indent=2)

    # --- Aggregate ---
    summary = {"providers": {}, "config": vars(args)}
    for prov_name in [p[0] for p in providers]:
        prov_results = [r for r in all_run_results if r["provider"] == prov_name]
        all_rules = [rule for r in prov_results for rule in r["rules"]]

        produced = [r for r in all_rules if r.get("outcome") == "produced"]
        declined = [r for r in all_rules if r.get("outcome") == "declined"]

        valid = [r for r in produced if r.get("regex_valid")]
        well_formed = [r for r in produced if r.get("rule_text_present")]

        # FP-rate distributions on the held-out set (only rules with a valid pattern)
        fp_holdout = [r["fp_rate_holdout"] for r in produced if r.get("fp_rate_holdout") is not None]
        fp_dev = [r["fp_rate_dev"] for r in produced if r.get("fp_rate_dev") is not None]

        # FP-Guard verdicts (after adjustment)
        verdicts = Counter(r.get("fp_verdict_holdout", "NO_PATTERN") for r in produced)

        # Approval-style filter: SAFE/ACCEPTABLE on dev → would be admin-reviewed
        admin_reviewable = [r for r in produced
                            if r.get("fp_verdict_dev") in ("SAFE", "ACCEPTABLE")]

        # Of those, how many actually had low FP on the held-out set?
        admin_reviewable_holdout_safe = [r for r in admin_reviewable
                                         if r.get("fp_verdict_holdout") in ("SAFE", "ACCEPTABLE")]

        summary["providers"][prov_name] = {
            "n_runs": len(prov_results),
            "n_clusters_attempted": len(all_rules),
            "n_rules_produced": len(produced),
            "n_declined": len(declined),
            "decline_rate": len(declined) / max(1, len(all_rules)),
            "n_valid_regex": len(valid),
            "n_well_formed_secrule": len(well_formed),
            "fp_rate_dev_mean": float(np.mean(fp_dev)) if fp_dev else None,
            "fp_rate_dev_p95": float(np.percentile(fp_dev, 95)) if fp_dev else None,
            "fp_rate_holdout_mean": float(np.mean(fp_holdout)) if fp_holdout else None,
            "fp_rate_holdout_p95": float(np.percentile(fp_holdout, 95)) if fp_holdout else None,
            "verdict_distribution_holdout": dict(verdicts),
            "n_admin_reviewable": len(admin_reviewable),
            "n_admin_reviewable_holdout_safe": len(admin_reviewable_holdout_safe),
            "generalization_to_holdout": (
                len(admin_reviewable_holdout_safe) / max(1, len(admin_reviewable))
            ),
        }

    with open(Path(args.out) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Pretty print ---
    print("\n" + "=" * 80)
    print("LLM RULE SYNTHESIS QUANTIFICATION")
    print("=" * 80)
    metric_keys = [
        ("n_clusters_attempted", "Total clusters analyzed"),
        ("n_rules_produced", "Rules produced"),
        ("decline_rate", "Decline rate"),
        ("n_valid_regex", "Valid regex"),
        ("n_well_formed_secrule", "Well-formed SecRule"),
        ("fp_rate_dev_mean", "Mean FP rate (dev)"),
        ("fp_rate_holdout_mean", "Mean FP rate (holdout)"),
        ("n_admin_reviewable", "Admin-reviewable (SAFE/ACCEPTABLE)"),
        ("generalization_to_holdout", "  → still safe on holdout"),
    ]
    prov_names = list(summary["providers"].keys())
    print(f"{'Metric':<40} " + " ".join(f"{n:>15s}" for n in prov_names))
    for key, label in metric_keys:
        row = f"{label:<40} "
        for n in prov_names:
            v = summary["providers"][n].get(key)
            if v is None:
                row += f"{'—':>15s} "
            elif isinstance(v, float):
                row += f"{v:>15.4f} "
            else:
                row += f"{v:>15} "
        print(row)

    print(f"\n✓ Wrote → {args.out}/run_log.json + summary.json")


if __name__ == "__main__":
    main()
