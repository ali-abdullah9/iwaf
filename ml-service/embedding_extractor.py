"""
BiLSTM Embedding Extractor

Reuses the trained BiLSTM model to extract 256-dim embeddings from HTTP requests.
These embeddings replace the hand-engineered features used in the original DRE.

This is the bridge between the inference model (for classification) and the
DRE model (for clustering). One model, two purposes.
"""

import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.insert(0, '.')
from bilstm_model import IWAFBiLSTM, CharTokenizer, build_text_input, build_meta_features


class BiLSTMEmbedder(nn.Module):
    """
    Wraps a trained IWAFBiLSTM and exposes embedding extraction.
    Cuts off the model BEFORE the final classification layer to get
    rich 256-dim representations.
    """
    
    def __init__(self, base_model: IWAFBiLSTM):
        super().__init__()
        self.base = base_model
    
    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """
        Returns 256-dim embeddings (lstm_out_dim from BiLSTM, before classification).
        
        Args:
            x: (batch, seq_len) char indices
            meta: (batch, meta_features_dim)
            
        Returns:
            (batch, 256) embeddings
        """
        # Compute mask
        mask = (x != self.base.pad_idx).float()
        
        # Embedding + LSTM
        embedded = self.base.embedding(x)
        lstm_out, _ = self.base.lstm(embedded)
        
        # Attention pooling - this is the embedding we want
        attended = self.base.attention(lstm_out, mask)
        # attended is (batch, 256) - hidden_dim * 2 for bidirectional
        
        return attended


def extract_embeddings(model, tokenizer, requests_df, device='cpu', batch_size=32):
    """
    Extract BiLSTM embeddings for a batch of requests.
    
    Args:
        model: IWAFBiLSTM (trained)
        tokenizer: CharTokenizer
        requests_df: DataFrame with method, path, query_string, body columns
        device: 'cpu' or 'cuda'
        batch_size: Batch size for inference
        
    Returns:
        numpy array of shape (n_requests, 256)
    """
    embedder = BiLSTMEmbedder(model).to(device)
    embedder.eval()
    
    all_embeddings = []
    
    for start in range(0, len(requests_df), batch_size):
        batch_df = requests_df.iloc[start:start + batch_size]
        
        # Tokenize
        texts = []
        metas = []
        for _, row in batch_df.iterrows():
            method = str(row.get('method', 'GET'))
            path = str(row.get('path', '/'))
            query = str(row.get('query_string', '')) if str(row.get('query_string', '')) != 'nan' else ''
            body = str(row.get('body', '')) if str(row.get('body', '')) != 'nan' else ''
            
            text = build_text_input(method, path, query, body)
            meta = build_meta_features(method, path, query, body, text)
            
            texts.append(tokenizer.tokenize(text))
            metas.append(meta)
        
        x = torch.tensor(np.array(texts), dtype=torch.long).to(device)
        m = torch.tensor(np.array(metas), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            emb = embedder(x, m).cpu().numpy()
        
        all_embeddings.append(emb)
    
    return np.vstack(all_embeddings) if all_embeddings else np.array([])


def extract_single_embedding(model, tokenizer, method: str, path: str, 
                              query: str, body: str, device='cpu') -> np.ndarray:
    """Extract embedding for a single request."""
    embedder = BiLSTMEmbedder(model).to(device)
    embedder.eval()
    
    text = build_text_input(method, path, query, body)
    meta = build_meta_features(method, path, query, body, text)
    
    x = torch.tensor(np.array([tokenizer.tokenize(text)]), dtype=torch.long).to(device)
    m = torch.tensor(np.array([meta]), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        emb = embedder(x, m).cpu().numpy()[0]
    
    return emb
