"""
Create reproducible train/val/test splits from train.csv.

Outputs (stratified by label, seed=42):
  - train_split.csv  (~80%)
  - val_split.csv    (~10%)
  - test_split.csv   (~10%)

Run once: python make_splits.py
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

INPUT = "train.csv"

if not os.path.exists(INPUT):
    sys.stderr.write(
        f"ERROR: {INPUT} not found.\n"
        f"Copy the unified training set (unified_v2/train.csv) into this folder first.\n"
    )
    sys.exit(1)

df = pd.read_csv(INPUT)
print(f"Loaded {len(df):,} rows from {INPUT}")
print(f"Class distribution:\n{df['label'].value_counts()}\n")

# 80 / 10 / 10 stratified split
train_df, temp_df = train_test_split(
    df, test_size=0.20, stratify=df["label"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED
)

train_df.to_csv("train_split.csv", index=False)
val_df.to_csv("val_split.csv", index=False)
test_df.to_csv("test_split.csv", index=False)

print(f"train_split.csv : {len(train_df):,}")
print(f"val_split.csv   : {len(val_df):,}")
print(f"test_split.csv  : {len(test_df):,}")

print("\nTest split class distribution:")
print(test_df["label"].value_counts())
print("\nDone. Splits are deterministic — re-running gives identical files.")
