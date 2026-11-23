"""
Shared utilities for all neural baselines.

Copies the tokenizer + metadata feature extractor + dataset class from
ml-service/bilstm_model.py, so all baselines use IDENTICAL preprocessing.
This is critical for fair comparison.

If anything changes in ml-service/bilstm_model.py, mirror it here.
"""

import re
from urllib.parse import unquote

import numpy as np
import pandas as pd
import torch


# ============================================================
# Constants - match ml-service/bilstm_model.py exactly
# ============================================================
MAX_LEN = 384  # same as the deployed model (ml-service/artifacts/model_info.json)

LABELS = ["benign", "sqli", "xss", "lfi", "cmdi", "probe"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}
IDX_TO_LABEL = {i: l for l, i in LABEL_TO_IDX.items()}
NUM_CLASSES = len(LABELS)


# ============================================================
# CharTokenizer — IDENTICAL to ml-service/bilstm_model.py CharTokenizer
# ============================================================
class CharTokenizer:
    def __init__(self, max_len=MAX_LEN):
        self.max_len = max_len
        chars = list("\t\n\r ")  # whitespace
        chars += [chr(i) for i in range(32, 127)]  # ASCII printable
        chars += ["<UNK>", "<PAD>"]
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        self.pad_idx = self.char_to_idx["<PAD>"]
        self.unk_idx = self.char_to_idx["<UNK>"]

    def tokenize(self, text):
        text = text[: self.max_len]
        tokens = [self.char_to_idx.get(c, self.unk_idx) for c in text]
        if len(tokens) < self.max_len:
            tokens += [self.pad_idx] * (self.max_len - len(tokens))
        return tokens


# ============================================================
# Feature extractors — IDENTICAL to build_text_input + build_meta_features in ml-service/bilstm_model.py
# ============================================================
def build_text_input(method, path, query, body):
    method = str(method).upper()
    path = str(path) if path else "/"
    query = str(query) if query else ""
    body = str(body) if body else ""
    for _ in range(3):
        q_dec = unquote(query)
        b_dec = unquote(body)
        p_dec = unquote(path)
        if q_dec == query and b_dec == body and p_dec == path:
            break
        query, body, path = q_dec, b_dec, p_dec
    return f"{method} {path} {query} {body}"


def build_meta_features(method, path, query, body, text):
    method = str(method) if method else ""
    path = str(path) if path else ""
    query = str(query) if query else ""
    body = str(body) if body else ""
    text = str(text) if text else ""
    features = [
        len(query),
        len(body),
        len(path),
        path.count("/"),
        query.count("&") + (1 if query else 0),
        int(bool(re.search(r"[<>'\";\-\(\){}]", text))),
        int("%" in query or "%" in body),
        text.count(".."),
        text.count("'") + text.count('"'),
        text.count(";"),
        text.count("<") + text.count(">"),
        len(re.findall(
            r"\b(SELECT|UNION|INSERT|UPDATE|DELETE|DROP|FROM|WHERE|AND|OR|SLEEP|BENCHMARK)\b",
            text, re.I)),
    ]
    return np.array(features, dtype=np.float32)


META_DIM = 12


# ============================================================
# Dataset class
# ============================================================
class IWAFDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.tokenizer = tokenizer
        self.texts = []
        self.metas = []
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
            text = build_text_input(method, path, query, body)
            meta = build_meta_features(method, path, query, body, text)
            label = LABEL_TO_IDX[row["label"]]
            self.texts.append(text)
            self.metas.append(meta)
            self.labels.append(label)

        self.metas = np.array(self.metas, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.tokenizer.tokenize(self.texts[idx]), dtype=torch.long)
        meta = torch.tensor(self.metas[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tokens, meta, label


# ============================================================
# Convenience loaders
# ============================================================
def load_split(name):
    """Load 'train' / 'val' / 'test' as DataFrame."""
    path = f"{name}_split.csv"
    df = pd.read_csv(path)
    df = df.fillna("")
    return df


def class_weights_from_train(train_df):
    """Inverse-frequency class weights, normalized to sum to NUM_CLASSES."""
    counts = train_df["label"].value_counts()
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    for label, count in counts.items():
        weights[LABEL_TO_IDX[label]] = 1.0 / count
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)
