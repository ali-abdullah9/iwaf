"""
iWAF Unified Dataset Preprocessor

Merges all data sources into a single training CSV with format:
    method, path, query_string, body, label

Data sources:
1. Our synthetic v2 (iwaf_synthetic_v2.csv) — ModSec bypass focus
2. CSIC 2010 (csic_database.csv) — 36k normal + 25k anomalous full HTTP
3. Kaggle SQL dataset (Modified_SQL_Dataset.csv) — payload-only
4. Kaggle XSS dataset (XSS_dataset.csv) — payload-only
5. Zenodo ModSec 2025 (zenodo_parsed.csv) — real production logs

Strategy:
- Fix Zenodo mislabeling (many "LFI" are actually probe - config file access)
- Balance classes via downsampling (probe/lfi over-represented in Zenodo)
- Deduplicate across sources
- Reserve adversarial test set with teammate's bypasses

Output: unified_train.csv, unified_val.csv, unified_test.csv
"""

import os
import re
import csv
import random
import pandas as pd
import numpy as np
from urllib.parse import urlparse, unquote
from collections import Counter
from pathlib import Path

random.seed(42)
np.random.seed(42)


# ============================================================
# Source 1: Our Synthetic v2
# ============================================================
def load_synthetic_v2(filepath: str) -> pd.DataFrame:
    print(f"\n[1] Loading synthetic v2: {filepath}")
    df = pd.read_csv(filepath).fillna("")
    print(f"  → {len(df)} rows, classes: {df['label'].value_counts().to_dict()}")
    return df


# ============================================================
# Source 2: CSIC 2010
# ============================================================
def load_csic(filepath: str) -> pd.DataFrame:
    print(f"\n[2] Loading CSIC 2010: {filepath}")
    df = pd.read_csv(filepath)
    # First column unnamed, contains Normal/Anomalous
    df = df.rename(columns={df.columns[0]: 'Class'})
    
    results = []
    for _, row in df.iterrows():
        url_full = str(row.get('URL', ''))
        if ' HTTP' in url_full:
            url = url_full.split(' HTTP')[0]
        else:
            url = url_full
        
        # Parse URL
        try:
            parsed = urlparse(url)
            path = parsed.path or '/'
            query = parsed.query or ''
        except Exception:
            path = url
            query = ''
        
        method = str(row.get('Method', 'GET'))
        body = str(row.get('content', ''))
        if body == 'nan' or pd.isna(row.get('content')):
            body = ''
        
        cls = row['Class']
        
        if cls == 'Normal':
            label = 'benign'
        else:
            # Classify anomalous based on content
            label = classify_anomalous_content(path, query, body)
        
        results.append({
            'method': method,
            'path': path,
            'query_string': query,
            'body': body,
            'label': label,
        })
    
    result_df = pd.DataFrame(results)
    print(f"  → {len(result_df)} rows, classes: {result_df['label'].value_counts().to_dict()}")
    return result_df


def classify_anomalous_content(path: str, query: str, body: str) -> str:
    """Classify CSIC anomalous request based on URL/body patterns."""
    # URL decode for pattern matching
    combined = (path + ' ' + query + ' ' + body).lower()
    decoded = unquote(combined)
    text = combined + ' ' + decoded
    
    # SQLi - strong patterns
    if re.search(r"(union\s+select|drop\s+table|'\s*or\s*'1'\s*=\s*'1|or\s+1\s*=\s*1|' or\s|--\s*$|'; drop|waitfor\s+delay|sleep\()", text):
        return 'sqli'
    
    # XSS - strong patterns
    if re.search(r"(<script|<img[^>]*onerror|<svg[^>]*onload|javascript:|onerror\s*=|onload\s*=|alert\(|<iframe)", text):
        return 'xss'
    
    # LFI - path traversal or file include
    if re.search(r"(\.\./|%2e%2e|/etc/passwd|/etc/shadow|boot\.ini|win\.ini|/proc/self|php://|file://|expect://|data://)", text):
        return 'lfi'
    
    # CMDi
    if re.search(r"(\|\s*(cat|ls|id|whoami|nc|bash|sh)\s|;\s*(cat|ls|id|whoami)\s|`.*`|\$\(.*\)|%0a(cat|id|whoami|ls))", text):
        return 'cmdi'
    
    # Probe-like
    if re.search(r"(/\.env|/\.git|/wp-admin|/phpmyadmin|/admin\.php|/backup|/\.aws|/server-status|/\.ssh)", text):
        return 'probe'
    
    # Parameter manipulation / anomaly — default to sqli (most common anomaly type in CSIC)
    # Could also be CRLF injection, buffer overflow, etc.
    return 'sqli'


# ============================================================
# Source 3: Kaggle Modified SQL Dataset
# ============================================================
def load_kaggle_sqli(filepath: str) -> pd.DataFrame:
    print(f"\n[3] Loading Kaggle SQLi: {filepath}")
    df = pd.read_csv(filepath)
    
    # Only use attack samples (Label=1)
    attacks = df[df['Label'] == 1]['Query'].dropna().astype(str).tolist()
    
    # Filter out overly simple/random strings that aren't really SQLi
    # (Label=0 had things like "99745017c", "ejerci78" - not useful)
    valid_attacks = [q for q in attacks if len(q) > 10]
    
    results = []
    # Wrap each payload in synthetic HTTP requests
    for payload in valid_attacks:
        # Half as GET query param, half as POST body
        if random.random() < 0.5:
            param = random.choice(['id', 'user', 'q', 'search', 'name', 'category'])
            path = random.choice(['/search', '/products', '/api/data', '/rest/products/search',
                                 '/api/users', '/items'])
            # URL-encode the payload
            from urllib.parse import quote
            results.append({
                'method': 'GET',
                'path': path,
                'query_string': f'{param}={quote(payload, safe="")}',
                'body': '',
                'label': 'sqli',
            })
        else:
            import json
            param = random.choice(['email', 'username', 'query', 'search'])
            path = random.choice(['/login', '/rest/user/login', '/api/search', '/register'])
            body = json.dumps({param: payload, 'password': 'anything123'})
            results.append({
                'method': 'POST',
                'path': path,
                'query_string': '',
                'body': body,
                'label': 'sqli',
            })
    
    # Also include Label=0 as benign — but only the longer, request-like ones
    benign = df[df['Label'] == 0]['Query'].dropna().astype(str).tolist()
    # Filter: skip random garbage, keep things that look like normal queries
    valid_benign = [q for q in benign if len(q) > 15 and re.search(r'[a-zA-Z]{4,}', q)]
    
    for payload in valid_benign[:3000]:  # Cap benign to avoid overflow
        from urllib.parse import quote
        results.append({
            'method': 'GET',
            'path': '/search',
            'query_string': f'q={quote(payload, safe="")}',
            'body': '',
            'label': 'benign',
        })
    
    result_df = pd.DataFrame(results)
    print(f"  → {len(result_df)} rows, classes: {result_df['label'].value_counts().to_dict()}")
    return result_df


# ============================================================
# Source 4: Kaggle XSS Dataset
# ============================================================
def load_kaggle_xss(filepath: str) -> pd.DataFrame:
    print(f"\n[4] Loading Kaggle XSS: {filepath}")
    df = pd.read_csv(filepath)
    
    attacks = df[df['Label'] == 1]['Sentence'].dropna().astype(str).tolist()
    valid_attacks = [s for s in attacks if len(s) > 5 and len(s) < 5000]  # Reasonable size
    
    results = []
    from urllib.parse import quote
    
    for payload in valid_attacks:
        if random.random() < 0.6:
            # GET with payload in query string
            param = random.choice(['q', 'search', 'name', 'comment', 'text', 'msg', 'input'])
            path = random.choice(['/search', '/comment', '/page', '/profile', 
                                 '/rest/products/search', '/api/Feedbacks/'])
            results.append({
                'method': 'GET',
                'path': path,
                'query_string': f'{param}={quote(payload, safe="")}',
                'body': '',
                'label': 'xss',
            })
        else:
            # POST with JSON body
            import json
            param = random.choice(['comment', 'message', 'text', 'bio', 'description'])
            path = random.choice(['/api/Feedbacks/', '/api/Complaints/', '/comment', '/feedback'])
            body = json.dumps({param: payload})
            results.append({
                'method': 'POST',
                'path': path,
                'query_string': '',
                'body': body,
                'label': 'xss',
            })
    
    # Use XSS Label=0 as benign HTML content (they look like real HTML pages)
    benign = df[df['Label'] == 0]['Sentence'].dropna().astype(str).tolist()
    valid_benign = [s for s in benign if 20 < len(s) < 3000]
    
    for payload in valid_benign[:2000]:
        results.append({
            'method': 'POST',
            'path': '/api/content/render',
            'query_string': '',
            'body': payload[:2000],  # Limit size
            'label': 'benign',
        })
    
    result_df = pd.DataFrame(results)
    print(f"  → {len(result_df)} rows, classes: {result_df['label'].value_counts().to_dict()}")
    return result_df


# ============================================================
# Source 5: Zenodo ModSec
# ============================================================
def load_zenodo(filepath: str) -> pd.DataFrame:
    print(f"\n[5] Loading Zenodo ModSec: {filepath}")
    df = pd.read_csv(filepath).fillna("")
    
    # FIX: relabel "lfi" entries that are actually probe (config file access)
    print(f"  Fixing Zenodo LFI mislabeling...")
    probe_patterns = r'(^/\.env|^/\.git|^/\.svn|/wp-config|/config\.php|/\.htaccess|/\.aws|/\.ssh|/database\.yml)'
    is_probe_path = df['path'].astype(str).str.contains(probe_patterns, regex=True, case=False, na=False)
    
    # True LFI has ../ or php:// in path or query
    true_lfi_pattern = r'(\.\./|%2e%2e|php://|file://|/etc/passwd|/etc/shadow|expect://)'
    is_true_lfi = (df['path'].astype(str).str.contains(true_lfi_pattern, regex=True, na=False) | 
                   df['query_string'].astype(str).str.contains(true_lfi_pattern, regex=True, na=False))
    
    # Reclassify mislabeled LFI → probe
    mislabeled_lfi = (df['label'] == 'lfi') & is_probe_path & (~is_true_lfi)
    print(f"    Reclassifying {mislabeled_lfi.sum()} 'lfi' → 'probe'")
    df.loc[mislabeled_lfi, 'label'] = 'probe'
    
    # Remove noise: paths like /d_3b1ec613/d_bd205cb5 (anonymized garbage from Zenodo)
    # These are obfuscated subdirectories - they're real but not useful for training
    # Keep them but just note it
    
    # Downsample probe (it's too dominant at 85k records + now more from reclassification)
    probe_rows = df[df['label'] == 'probe']
    if len(probe_rows) > 15000:
        probe_sample = probe_rows.sample(n=15000, random_state=42)
        non_probe = df[df['label'] != 'probe']
        df = pd.concat([non_probe, probe_sample], ignore_index=True)
        print(f"    Downsampled probe: 15000 kept")
    
    # Downsample LFI similarly
    lfi_rows = df[df['label'] == 'lfi']
    if len(lfi_rows) > 8000:
        lfi_sample = lfi_rows.sample(n=8000, random_state=42)
        non_lfi = df[df['label'] != 'lfi']
        df = pd.concat([non_lfi, lfi_sample], ignore_index=True)
        print(f"    Downsampled lfi: 8000 kept")
    
    print(f"  → {len(df)} rows (after rebalancing)")
    print(f"    Classes: {df['label'].value_counts().to_dict()}")
    return df


# ============================================================
# Deduplication
# ============================================================
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[Dedup] Deduplicating {len(df)} rows...")
    # Create a combined hash field
    df['_dedup_key'] = (df['method'].astype(str) + '|' + 
                        df['path'].astype(str) + '|' + 
                        df['query_string'].astype(str) + '|' + 
                        df['body'].astype(str))
    before = len(df)
    df = df.drop_duplicates(subset=['_dedup_key']).drop(columns=['_dedup_key'])
    after = len(df)
    print(f"  → Removed {before - after} duplicates ({after} remaining)")
    return df


# ============================================================
# Carve out adversarial test set
# ============================================================
def carve_adversarial_test(df: pd.DataFrame) -> tuple:
    """
    Create a special test set with our teammate's known bypasses.
    These are the specific attack types ModSec misses.
    """
    print(f"\n[Test] Carving adversarial test set...")
    
    adversarial_keywords = {
        'utf7': r'\+ADw-|\+AD4-|\+AEA-|\+AD0-',
        'html_entity': r'&#x09;|&#x0a;|&#x0d;|&#09;|&#10;|&#13;',
        'inline_comment_sqli': r'/\*[^*]+\*/\s*(OR|or)\s+\d{6,}\s*=\s*\d{6,}',
        'random_num_sqli': r'\s(OR|or)\s+\d{7,}\s*=\s*\d{7,}',
    }
    
    combined_text = (df['path'].astype(str) + ' ' + 
                     df['query_string'].astype(str) + ' ' + 
                     df['body'].astype(str))
    
    adversarial_mask = pd.Series(False, index=df.index)
    for name, pattern in adversarial_keywords.items():
        matches = combined_text.str.contains(pattern, regex=True, na=False)
        count = matches.sum()
        print(f"  {name}: {count} samples")
        adversarial_mask |= matches
    
    # Take a sample of these as dedicated test set
    adversarial = df[adversarial_mask]
    if len(adversarial) > 300:
        # Sample 100 per type ideally
        adv_test = adversarial.sample(n=min(300, len(adversarial)), random_state=42)
    else:
        adv_test = adversarial
    
    remaining = df.drop(adv_test.index)
    
    print(f"  → Adversarial test set: {len(adv_test)} samples")
    print(f"  → Remaining for train/val/test: {len(remaining)}")
    
    return remaining, adv_test


# ============================================================
# Class balancing
# ============================================================
def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Balance classes through downsampling the majority class.
    Keep attack classes as diverse as possible.
    """
    print(f"\n[Balance] Balancing classes...")
    print(f"  Before: {df['label'].value_counts().to_dict()}")
    
    # Target: each attack class max 15000, benign max 30000 (2x biggest attack)
    limits = {
        'benign': 30000,
        'sqli': 15000,
        'xss': 15000,
        'lfi': 10000,
        'cmdi': 10000,
        'probe': 12000,
    }
    
    balanced_parts = []
    for label, limit in limits.items():
        subset = df[df['label'] == label]
        if len(subset) > limit:
            subset = subset.sample(n=limit, random_state=42)
        balanced_parts.append(subset)
    
    df = pd.concat(balanced_parts, ignore_index=True)
    print(f"  After: {df['label'].value_counts().to_dict()}")
    return df


# ============================================================
# Train/val/test split
# ============================================================
def train_val_test_split(df: pd.DataFrame, val_ratio=0.1, test_ratio=0.1):
    """Stratified split so each class appears proportionally."""
    from sklearn.model_selection import train_test_split
    
    # First split off test
    remaining, test = train_test_split(df, test_size=test_ratio, 
                                       stratify=df['label'], random_state=42)
    # Then split val from remaining
    adjusted_val = val_ratio / (1 - test_ratio)
    train, val = train_test_split(remaining, test_size=adjusted_val,
                                  stratify=remaining['label'], random_state=42)
    
    return train, val, test


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build the unified iWAF dataset from the raw sources.")
    ap.add_argument("--raw-dir", default="data/raw",
                    help="Folder with csic_database.csv, Modified_SQL_Dataset.csv, XSS_dataset.csv, zenodo_parsed.csv")
    ap.add_argument("--synthetic", default=None,
                    help="Synthetic v2 CSV (default: <raw-dir>/iwaf_synthetic_v2.csv)")
    ap.add_argument("--out-dir", default="data/unified", help="Output folder for the splits")
    args = ap.parse_args()
    base = args.raw_dir
    
    # Load each source
    parts = []
    parts.append(load_synthetic_v2(args.synthetic or f"{base}/iwaf_synthetic_v2.csv"))
    parts.append(load_csic(f"{base}/csic_database.csv"))
    parts.append(load_kaggle_sqli(f"{base}/Modified_SQL_Dataset.csv"))
    parts.append(load_kaggle_xss(f"{base}/XSS_dataset.csv"))
    parts.append(load_zenodo(f"{base}/zenodo_parsed.csv"))
    
    # Combine
    print(f"\n{'='*60}")
    print("COMBINING ALL SOURCES")
    print(f"{'='*60}")
    df = pd.concat(parts, ignore_index=True)
    # Ensure consistent columns
    df = df[['method', 'path', 'query_string', 'body', 'label']].fillna("")
    df = df.astype(str)
    print(f"Total combined: {len(df)}")
    print(f"Classes: {df['label'].value_counts().to_dict()}")
    
    # Deduplicate
    df = deduplicate(df)
    
    # Balance
    df = balance_classes(df)
    
    # Carve out adversarial test
    df_main, df_adv_test = carve_adversarial_test(df)
    
    # Train/Val/Test split
    print(f"\n[Split] Creating train/val/test splits...")
    train, val, test = train_val_test_split(df_main)
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    
    # Save all
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    df_adv_test.to_csv(output_dir / "adversarial_test.csv", index=False)
    
    # Full combined for reference
    df.to_csv(output_dir / "full_dataset.csv", index=False)
    
    print(f"\n{'='*60}")
    print("✅ PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}/")
    print(f"  train.csv:              {len(train)}")
    print(f"  val.csv:                {len(val)}")
    print(f"  test.csv:               {len(test)}")
    print(f"  adversarial_test.csv:   {len(df_adv_test)}")
    print(f"  full_dataset.csv:       {len(df)}")
    
    # Final breakdown
    print(f"\nTrain set class distribution:")
    for label, count in train['label'].value_counts().items():
        print(f"  {label:10s}: {count:>6} ({count/len(train)*100:.1f}%)")
    
    print(f"\nAdversarial test set (red-team bypass types):")
    for label, count in df_adv_test['label'].value_counts().items():
        print(f"  {label:10s}: {count:>6}")


if __name__ == "__main__":
    main()
