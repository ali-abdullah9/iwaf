"""
Multi-Relational GCN for iWAF Dynamic Rule Engine

Uses BiLSTM embeddings as input features and learns graph-aware embeddings
through multiple edge types:
    - content_similarity: cosine similarity of BiLSTM embeddings
    - same_source_ip:     same attacker IP within time window
    - temporal_proximity: requests close in time (campaign detection)
    - same_path:          requests targeting the same endpoint

Architecture:
    Input: (N, 256) BiLSTM embeddings + multiple adjacency matrices
        ↓
    GCN layer 1 (per relation type, then aggregate) → 128-dim hidden
        ↓
    GCN layer 2 → 64-dim final embedding
        ↓
    Output: (N, 64) graph-aware embeddings ready for K-Means

Trained via self-supervised contrastive learning:
    - Positive pairs: same attack class, similar payloads
    - Negative pairs: different classes
    - Loss: InfoNCE / triplet loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiRelationalGCNLayer(nn.Module):
    """
    GCN layer that handles multiple edge types (relations).
    Each relation has its own learnable weight, then we aggregate.
    """
    
    def __init__(self, in_dim: int, out_dim: int, num_relations: int = 4):
        super().__init__()
        self.num_relations = num_relations
        
        # One linear transformation per relation
        self.relation_weights = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_relations)
        ])
        
        # Self-loop transformation
        self.self_loop = nn.Linear(in_dim, out_dim)
        
        # Learnable relation importance (so the model can weight different relations)
        self.relation_importance = nn.Parameter(torch.ones(num_relations))
    
    def forward(self, x: torch.Tensor, adj_list: list) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) node features
            adj_list: list of (N, N) normalized adjacency matrices, one per relation
            
        Returns:
            (N, out_dim) updated node features
        """
        # Self transformation (always applies)
        out = self.self_loop(x)
        
        # Apply each relation
        importance = F.softmax(self.relation_importance, dim=0)
        for i, (adj, weight_layer) in enumerate(zip(adj_list, self.relation_weights)):
            transformed = weight_layer(x)
            propagated = torch.mm(adj, transformed)
            out = out + importance[i] * propagated
        
        return out


class MultiRelationalGCN(nn.Module):
    """
    Two-layer multi-relational GCN.
    """
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, 
                 embedding_dim: int = 64, num_relations: int = 4,
                 dropout: float = 0.4):
        super().__init__()
        self.gcn1 = MultiRelationalGCNLayer(input_dim, hidden_dim, num_relations)
        self.gcn2 = MultiRelationalGCNLayer(hidden_dim, embedding_dim, num_relations)
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor, adj_list: list) -> torch.Tensor:
        h = F.relu(self.gcn1(x, adj_list))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gcn2(h, adj_list)
        # L2 normalize for cosine similarity in contrastive loss
        h = F.normalize(h, p=2, dim=1)
        return h
    
    def get_relation_weights(self):
        """Return learned relation importance for analysis."""
        return F.softmax(self.gcn1.relation_importance, dim=0).detach().cpu().numpy()


# ============================================================
# Multi-relational graph builder
# ============================================================
def build_multi_relational_graph(
    embeddings: np.ndarray,
    requests_df,
    similarity_threshold: float = 0.7,
    time_window_minutes: int = 30,
):
    """
    Build adjacency matrices for multiple edge types.
    
    Args:
        embeddings: (N, 256) BiLSTM embeddings
        requests_df: DataFrame with source_ip, timestamp, path columns
        similarity_threshold: cosine similarity threshold for content edges
        time_window_minutes: time window for temporal edges
        
    Returns:
        list of 4 adjacency matrices: [content, same_ip, temporal, same_path]
    """
    n = len(embeddings)
    if n == 0:
        return [np.zeros((0, 0), dtype=np.float32)] * 4
    
    # 1. CONTENT SIMILARITY (cosine on embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms
    sim = normalized @ normalized.T
    adj_content = (sim >= similarity_threshold).astype(np.float32)
    np.fill_diagonal(adj_content, 1.0)
    
    # 2. SAME SOURCE IP
    adj_ip = np.zeros((n, n), dtype=np.float32)
    if 'source_ip' in requests_df.columns:
        ips = requests_df['source_ip'].fillna('unknown').values
        for i in range(n):
            for j in range(n):
                if ips[i] == ips[j] and ips[i] != 'unknown':
                    adj_ip[i, j] = 1.0
    np.fill_diagonal(adj_ip, 1.0)
    
    # 3. TEMPORAL PROXIMITY
    adj_time = np.zeros((n, n), dtype=np.float32)
    if 'timestamp' in requests_df.columns:
        try:
            timestamps = pd_to_seconds(requests_df['timestamp'])
            time_window_sec = time_window_minutes * 60
            for i in range(n):
                for j in range(n):
                    if abs(timestamps[i] - timestamps[j]) <= time_window_sec:
                        adj_time[i, j] = 1.0
        except Exception:
            pass
    np.fill_diagonal(adj_time, 1.0)
    
    # 4. SAME PATH
    adj_path = np.zeros((n, n), dtype=np.float32)
    if 'path' in requests_df.columns:
        paths = requests_df['path'].fillna('/').values
        for i in range(n):
            for j in range(n):
                if paths[i] == paths[j]:
                    adj_path[i, j] = 1.0
    np.fill_diagonal(adj_path, 1.0)
    
    # Normalize all
    return [normalize_adj(a) for a in [adj_content, adj_ip, adj_time, adj_path]]


def pd_to_seconds(timestamp_series):
    """Convert pandas timestamp series to Unix seconds."""
    import pandas as pd
    ts = pd.to_datetime(timestamp_series, errors='coerce')
    return (ts.astype('int64') // 10**9).values


def normalize_adj(adj: np.ndarray) -> np.ndarray:
    """Symmetric normalization: D^-1/2 * A * D^-1/2"""
    if len(adj) == 0:
        return adj
    degree = adj.sum(axis=1)
    degree_inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    degree_inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt


# ============================================================
# Self-supervised contrastive training
# ============================================================
class ContrastiveLoss(nn.Module):
    """
    InfoNCE-style contrastive loss.
    For each anchor, has positive (same class) and negatives (different class).
    """
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (N, embed_dim) L2-normalized embeddings
            labels: (N,) class labels
        """
        # Cosine similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.T) / self.temperature
        
        # Mask: 1 for same class (positive), 0 for different (negative)
        labels_expanded = labels.unsqueeze(0)
        positive_mask = (labels_expanded == labels_expanded.T).float()
        # Remove self-similarity from positives
        positive_mask = positive_mask - torch.eye(len(labels), device=labels.device)
        
        # InfoNCE-style: for each anchor, push positives close, negatives far
        # Loss = -log(sum(exp(sim_pos)) / sum(exp(sim_all)))
        
        # Numerator: sum of positive similarities (exp)
        exp_sim = torch.exp(sim_matrix)
        # Mask out self-comparison
        identity = torch.eye(len(labels), device=labels.device)
        exp_sim = exp_sim * (1 - identity)
        
        # Sum across positive samples
        sum_pos = (exp_sim * positive_mask).sum(dim=1)
        sum_all = exp_sim.sum(dim=1)
        
        # Avoid log(0)
        loss = -torch.log((sum_pos + 1e-8) / (sum_all + 1e-8))
        
        # Average only over anchors that have at least one positive
        has_positive = positive_mask.sum(dim=1) > 0
        if has_positive.sum() > 0:
            loss = loss[has_positive].mean()
        else:
            loss = loss.mean()
        
        return loss
