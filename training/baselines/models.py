"""
Neural baseline architectures.

All models accept (tokens, meta) and return logits of shape (batch, num_classes),
EVEN models that don't use meta (they just ignore it). This keeps the training
loop in train_neural.py uniform across all variants.

Architectures:
  - CharCNN              : Kim-style 1D conv with [3,5,7] kernels + meta branch
  - PlainLSTM            : 1-layer unidirectional LSTM, last hidden state, + meta
  - BiLSTMMeanPool       : 2-layer BiLSTM with mean-pool (no attention) + meta
  - BiLSTMAttnNoMeta     : Full BiLSTM-Attention but no meta branch (ablation)
  - BiLSTMAttnFull       : Identical architecture to ml-service/bilstm_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Attention layer (matches ml-service/bilstm_model.py AttentionLayer)
# ============================================================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out, mask=None):
        scores = self.attention(lstm_out).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)
        attended = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return attended


# ============================================================
# 1. CharCNN
# ============================================================
class CharCNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=64,
                 num_filters=64, kernel_sizes=(3, 5, 7), meta_dim=12,
                 dropout=0.4, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, k, padding=k // 2)
            for k in kernel_sizes
        ])
        conv_out = num_filters * len(kernel_sizes)
        combined_dim = conv_out + meta_dim
        self.fc1 = nn.Linear(combined_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx

    def forward(self, x, meta):
        emb = self.embedding(x).transpose(1, 2)  # (B, E, L)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)  # global max pool
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        h = self.dropout(h)
        h = torch.cat([h, meta], dim=1)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h)


# ============================================================
# 2. PlainLSTM
# ============================================================
class PlainLSTM(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=64,
                 hidden_dim=256, meta_dim=12, dropout=0.4, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,
                            bidirectional=False, batch_first=True)
        combined_dim = hidden_dim + meta_dim
        self.fc1 = nn.Linear(combined_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx

    def forward(self, x, meta):
        emb = self.embedding(x)
        out, (h, _) = self.lstm(emb)
        last = h[-1]  # (B, hidden)
        last = self.dropout(last)
        h = torch.cat([last, meta], dim=1)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h)


# ============================================================
# 3. BiLSTM with mean pooling (no attention) — ablation
# ============================================================
class BiLSTMMeanPool(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=64,
                 hidden_dim=128, num_layers=2, meta_dim=12,
                 dropout=0.4, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        lstm_out = hidden_dim * 2
        combined_dim = lstm_out + meta_dim
        self.fc1 = nn.Linear(combined_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx

    def forward(self, x, meta):
        mask = (x != self.pad_idx).float().unsqueeze(-1)
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        # Masked mean pool
        masked = out * mask
        pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = self.dropout(pooled)
        h = torch.cat([pooled, meta], dim=1)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h)


# ============================================================
# 4. BiLSTM-Attention WITHOUT metadata branch — ablation
# ============================================================
class BiLSTMAttnNoMeta(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=64,
                 hidden_dim=128, num_layers=2,
                 dropout=0.4, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        lstm_out = hidden_dim * 2
        self.attention = AttentionLayer(lstm_out)
        # No meta branch — just FC on attention output
        self.fc1 = nn.Linear(lstm_out, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx

    def forward(self, x, meta):  # meta is ignored
        mask = (x != self.pad_idx).float()
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        attended = self.attention(out, mask)
        attended = self.dropout(attended)
        h = F.relu(self.fc1(attended))
        h = self.dropout(h)
        return self.fc2(h)


# ============================================================
# 5. Full BiLSTM-Attention — IDENTICAL to ml-service/bilstm_model.py
# ============================================================
class BiLSTMAttnFull(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=64,
                 hidden_dim=128, num_layers=2, meta_dim=12,
                 dropout=0.4, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        lstm_out = hidden_dim * 2
        self.attention = AttentionLayer(lstm_out)
        self.dropout = nn.Dropout(dropout)
        combined_dim = lstm_out + meta_dim
        self.fc1 = nn.Linear(combined_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pad_idx = pad_idx

    def forward(self, x, meta):
        mask = (x != self.pad_idx).float()
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        attended = self.attention(out, mask)
        attended = self.dropout(attended)
        h = torch.cat([attended, meta], dim=1)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h)


MODEL_REGISTRY = {
    "cnn": CharCNN,
    "lstm_plain": PlainLSTM,
    "bilstm_meanpool": BiLSTMMeanPool,
    "bilstm_nometa": BiLSTMAttnNoMeta,
    "bilstm_full": BiLSTMAttnFull,
}
