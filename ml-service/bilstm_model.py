"""
iWAF BiLSTM Model
Character-level Bidirectional LSTM with attention and metadata feature fusion.

Architecture:
  Input HTTP request
    │
    ├─► Char-level tokenizer → embedding → BiLSTM → attention → 256-dim
    │
    ├─► Metadata extractor → 12-dim numeric features
    │
    └─► Concatenate (256 + 12) → Dense → Softmax → 6 classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from urllib.parse import unquote


# ============================================================
# Character vocabulary
# ============================================================
class CharTokenizer:
    """Character-level tokenizer for HTTP requests."""
    
    def __init__(self, max_len: int = 512):
        self.max_len = max_len
        # Build vocab: ASCII printable + space + special tokens
        chars = list("\t\n\r ")  # whitespace
        chars += [chr(i) for i in range(32, 127)]  # ASCII printable
        chars += ['<UNK>', '<PAD>']
        
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)
        self.pad_idx = self.char_to_idx['<PAD>']
        self.unk_idx = self.char_to_idx['<UNK>']
    
    def tokenize(self, text: str) -> list:
        """Convert text to list of character indices."""
        text = text[:self.max_len]
        tokens = [self.char_to_idx.get(c, self.unk_idx) for c in text]
        # Pad to max_len
        if len(tokens) < self.max_len:
            tokens += [self.pad_idx] * (self.max_len - len(tokens))
        return tokens
    
    def batch_tokenize(self, texts: list) -> torch.Tensor:
        """Tokenize a batch of texts."""
        return torch.tensor([self.tokenize(t) for t in texts], dtype=torch.long)


# ============================================================
# Attention layer
# ============================================================
class AttentionLayer(nn.Module):
    """Self-attention pooling over BiLSTM outputs."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_out: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden)
        # mask: (batch, seq_len) - 1 for real tokens, 0 for padding
        scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = F.softmax(scores, dim=1)  # (batch, seq_len)
        # Weighted sum
        attended = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden)
        return attended


# ============================================================
# Main BiLSTM model
# ============================================================
class IWAFBiLSTM(nn.Module):
    """
    Char-level BiLSTM + metadata features for HTTP request classification.
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        meta_features_dim: int = 12,
        dropout: float = 0.4,
        pad_idx: int = 0,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        lstm_out_dim = hidden_dim * 2  # bidirectional
        self.attention = AttentionLayer(lstm_out_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Combine LSTM output (256) + metadata (12) = 268
        combined_dim = lstm_out_dim + meta_features_dim
        self.fc1 = nn.Linear(combined_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.pad_idx = pad_idx
    
    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) char indices
        # meta: (batch, meta_features_dim)
        
        # Compute mask (1 for real, 0 for pad)
        mask = (x != self.pad_idx).float()
        
        # Embedding + LSTM
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        
        # Attention pooling
        attended = self.attention(lstm_out, mask)
        attended = self.dropout(attended)
        
        # Concatenate with metadata
        combined = torch.cat([attended, meta], dim=1)
        
        # Classifier
        hidden = F.relu(self.fc1(combined))
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)
        
        return logits


# ============================================================
# Feature extraction (matches FastAPI pipeline)
# ============================================================
def build_text_input(method: str, path: str, query: str, body: str) -> str:
    """Combine HTTP fields into single text with URL decoding."""
    method = str(method).upper()
    path = str(path) if path else '/'
    query = str(query) if query else ''
    body = str(body) if body else ''
    
    # Multi-level decode (handle double/triple encoding)
    for _ in range(3):
        q_dec = unquote(query)
        b_dec = unquote(body)
        p_dec = unquote(path)
        if q_dec == query and b_dec == body and p_dec == path:
            break
        query, body, path = q_dec, b_dec, p_dec
    
    return f"{method} {path} {query} {body}"


def build_meta_features(method: str, path: str, query: str, body: str, text: str) -> np.ndarray:
    """Extract 12 numerical metadata features."""
    method = str(method) if method else ''
    path = str(path) if path else ''
    query = str(query) if query else ''
    body = str(body) if body else ''
    text = str(text) if text else ''
    
    features = [
        len(query),                                    # query_len
        len(body),                                     # body_len
        len(path),                                     # path_len
        path.count("/"),                                # path_depth
        query.count("&") + (1 if query else 0),        # num_params
        int(bool(re.search(r"[<>'\";\-\-\(\){}]", text))),  # has_special_chars
        int("%" in query or "%" in body),              # has_encoding
        text.count(".."),                              # dot_dot_count
        text.count("'") + text.count('"'),             # quote_count
        text.count(";"),                               # semicolon_count
        text.count("<") + text.count(">"),             # angle_bracket_count
        len(re.findall(
            r'\b(SELECT|UNION|INSERT|UPDATE|DELETE|DROP|FROM|WHERE|AND|OR|SLEEP|BENCHMARK)\b',
            text, re.I)),                              # sql_keyword_count
    ]
    return np.array(features, dtype=np.float32)


# ============================================================
# Dataset class for training
# ============================================================
class IWAFDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for iWAF training."""
    
    def __init__(self, df, tokenizer: CharTokenizer, label_to_idx: dict):
        self.tokenizer = tokenizer
        self.label_to_idx = label_to_idx
        
        # Pre-process all data
        self.texts = []
        self.metas = []
        self.labels = []
        
        for _, row in df.iterrows():
            method = str(row.get('method', 'GET'))
            path = str(row.get('path', '/'))
            query = str(row.get('query_string', ''))
            body = str(row.get('body', ''))
            
            # Handle NaN/None
            if query == 'nan' or query is None: query = ''
            if body == 'nan' or body is None: body = ''
            
            text = build_text_input(method, path, query, body)
            meta = build_meta_features(method, path, query, body, text)
            label = label_to_idx[row['label']]
            
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
