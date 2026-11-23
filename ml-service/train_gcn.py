"""
Train Multi-Relational GCN for iWAF DRE

Self-supervised contrastive training:
    - Use BiLSTM embeddings as input features
    - Build multi-relational graphs (content + IP + time + path)
    - Train GCN to push same-class attacks together, different-class apart

Inputs:
    artifacts/ folder with trained BiLSTM (from previous training)
    unified_v2/ folder with training datasets

Outputs (saved to artifacts/):
    gcn.pt              - trained GCN weights
    gcn_info.json       - training stats and metrics
"""

import sys
import json
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from bilstm_model import IWAFBiLSTM, CharTokenizer
from embedding_extractor import extract_embeddings
from gcn_model import (
    MultiRelationalGCN, build_multi_relational_graph, ContrastiveLoss
)


CONFIG = {
    'data_dir': './unified_v2',
    'artifacts_dir': './artifacts',
    'gcn_input_dim': 256,        # BiLSTM output dim (hidden_dim * 2)
    'gcn_hidden_dim': 128,
    'gcn_embedding_dim': 64,
    'num_relations': 4,
    'dropout': 0.3,
    'batch_size': 256,           # graph batch size
    'epochs': 30,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'temperature': 0.3,
    'similarity_threshold': 0.7,
    'time_window_minutes': 30,
    'samples_per_class': 800,    # how many requests to use per class for training
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


def load_bilstm():
    """Load the trained BiLSTM and its tokenizer."""
    artifacts = Path(CONFIG['artifacts_dir'])
    
    with open(artifacts / 'tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open(artifacts / 'label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open(artifacts / 'model_info.json') as f:
        info = json.load(f)
    
    cfg = info['config']
    model = IWAFBiLSTM(
        vocab_size=tokenizer.vocab_size,
        num_classes=info['num_classes'],
        embedding_dim=cfg['embedding_dim'],
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        meta_features_dim=cfg['meta_features_dim'],
        dropout=cfg['dropout'],
        pad_idx=tokenizer.pad_idx,
    )
    
    weights = torch.load(artifacts / 'bilstm.pt', 
                         map_location=CONFIG['device'], weights_only=True)
    model.load_state_dict(weights)
    model = model.to(CONFIG['device'])
    model.eval()
    
    return model, tokenizer, label_encoder


def prepare_training_data(label_encoder):
    """
    Sample balanced data for GCN training.
    Inject synthetic source_ip and timestamps if missing.
    """
    train_df = pd.read_csv(Path(CONFIG['data_dir']) / 'train.csv').fillna("")
    
    # Sample balanced
    samples = []
    for cls in label_encoder.classes_:
        cls_df = train_df[train_df['label'] == cls]
        n_take = min(len(cls_df), CONFIG['samples_per_class'])
        if n_take == 0:
            continue
        sampled = cls_df.sample(n=n_take, random_state=42)
        samples.append(sampled)
    
    df = pd.concat(samples, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    
    # Inject synthetic source_ip and timestamps for graph building
    # In production these come from MongoDB; for training we simulate them
    np.random.seed(42)
    n = len(df)
    # Create attacker groups: same attack type often comes from same IP (campaign)
    df['source_ip'] = '0.0.0.0'
    for cls in label_encoder.classes_:
        idx = df[df['label'] == cls].index
        # ~5 attacker IPs per class (simulates real campaigns)
        n_attackers = max(1, len(idx) // 50)
        ips = [f'10.0.{i}.{np.random.randint(1, 255)}' for i in range(n_attackers)]
        for i in idx:
            df.at[i, 'source_ip'] = np.random.choice(ips)
    
    # Synthetic timestamps - cluster same-class attacks in time
    # Use Unix seconds as int instead of pandas Timestamp to avoid precision issues
    base_unix = 1743465600  # 2026-04-01 00:00:00 UTC in seconds
    timestamps = []
    
    for cls in label_encoder.classes_:
        idx = df[df['label'] == cls].index.tolist()
        # Same class attacks happen within hours of each other
        offsets = np.cumsum(np.random.normal(0, 30, size=len(idx))).astype(int)  # ~30 minute std
        for j, i in enumerate(idx):
            timestamps.append((i, base_unix + int(offsets[j]) * 60))  # offsets in minutes -> seconds
    
    # Apply timestamps
    timestamp_dict = dict(timestamps)
    df['timestamp_unix'] = df.index.map(lambda i: timestamp_dict.get(i, base_unix))
    # Use timestamp_unix for graph building (the build_multi_relational_graph expects 'timestamp')
    df['timestamp'] = pd.to_datetime(df['timestamp_unix'], unit='s', errors='coerce')
    
    return df


def train_gcn(gcn, embeddings, labels, requests_df, optimizer, loss_fn, batch_size=256):
    """Train one epoch of GCN."""
    gcn.train()
    n = len(embeddings)
    
    # Create batches by sampling subgraphs
    indices = np.random.permutation(n)
    epoch_loss = 0
    n_batches = 0
    
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        if len(batch_idx) < 4:
            continue
        
        # Subset embeddings, labels, df
        batch_emb = embeddings[batch_idx]
        batch_labels = labels[batch_idx]
        batch_df = requests_df.iloc[batch_idx].reset_index(drop=True)
        
        # Build adjacency matrices for this subgraph
        adj_list = build_multi_relational_graph(
            batch_emb, batch_df,
            similarity_threshold=CONFIG['similarity_threshold'],
            time_window_minutes=CONFIG['time_window_minutes'],
        )
        
        # Convert to tensors
        x = torch.tensor(batch_emb, dtype=torch.float32, device=CONFIG['device'])
        adj_tensors = [
            torch.tensor(a, dtype=torch.float32, device=CONFIG['device'])
            for a in adj_list
        ]
        labels_t = torch.tensor(batch_labels, dtype=torch.long, device=CONFIG['device'])
        
        # Forward
        optimizer.zero_grad()
        gcn_emb = gcn(x, adj_tensors)
        loss = loss_fn(gcn_emb, labels_t)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gcn.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        n_batches += 1
    
    return epoch_loss / max(1, n_batches)


def evaluate_clustering(gcn, embeddings, labels, requests_df):
    """
    Evaluate clustering quality: how well do GCN embeddings cluster by class?
    Uses silhouette score and homogeneity.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, homogeneity_score, normalized_mutual_info_score
    
    gcn.eval()
    
    # Build full graph
    adj_list = build_multi_relational_graph(
        embeddings, requests_df,
        similarity_threshold=CONFIG['similarity_threshold'],
        time_window_minutes=CONFIG['time_window_minutes'],
    )
    
    x = torch.tensor(embeddings, dtype=torch.float32, device=CONFIG['device'])
    adj_tensors = [
        torch.tensor(a, dtype=torch.float32, device=CONFIG['device'])
        for a in adj_list
    ]
    
    with torch.no_grad():
        gcn_emb = gcn(x, adj_tensors).cpu().numpy()
    
    # K-Means with k = number of true classes
    n_classes = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(gcn_emb)
    
    # Metrics
    silhouette = silhouette_score(gcn_emb, cluster_labels)
    homogeneity = homogeneity_score(labels, cluster_labels)
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    
    return {
        'silhouette': float(silhouette),
        'homogeneity': float(homogeneity),
        'nmi': float(nmi),
    }


def main():
    print("=" * 60)
    print("iWAF Multi-Relational GCN Training")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    
    # Load BiLSTM
    print("\n[1/4] Loading trained BiLSTM...")
    bilstm, tokenizer, label_encoder = load_bilstm()
    print(f"      ✓ BiLSTM loaded ({sum(p.numel() for p in bilstm.parameters()):,} params)")
    print(f"      ✓ Classes: {list(label_encoder.classes_)}")
    
    # Prepare training data
    print(f"\n[2/4] Sampling balanced training data...")
    df = prepare_training_data(label_encoder)
    print(f"      ✓ Total samples: {len(df)}")
    print(f"      ✓ Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Extract BiLSTM embeddings (one-time cost, ~1-2 min)
    print(f"\n[3/4] Extracting BiLSTM embeddings for all samples...")
    start = time.time()
    embeddings = extract_embeddings(bilstm, tokenizer, df, 
                                    device=CONFIG['device'], batch_size=64)
    print(f"      ✓ Embeddings shape: {embeddings.shape}")
    print(f"      ✓ Took {time.time()-start:.1f}s")
    
    labels = label_encoder.transform(df['label'].values)
    
    # Initialize GCN
    print(f"\n[4/4] Training Multi-Relational GCN...")
    gcn = MultiRelationalGCN(
        input_dim=CONFIG['gcn_input_dim'],
        hidden_dim=CONFIG['gcn_hidden_dim'],
        embedding_dim=CONFIG['gcn_embedding_dim'],
        num_relations=CONFIG['num_relations'],
        dropout=CONFIG['dropout'],
    ).to(CONFIG['device'])
    
    n_params = sum(p.numel() for p in gcn.parameters())
    print(f"      GCN params: {n_params:,}")
    
    optimizer = torch.optim.Adam(gcn.parameters(), 
                                 lr=CONFIG['learning_rate'],
                                 weight_decay=CONFIG['weight_decay'])
    loss_fn = ContrastiveLoss(temperature=CONFIG['temperature'])
    
    # Initial evaluation (untrained baseline)
    initial_metrics = evaluate_clustering(gcn, embeddings, labels, df)
    print(f"\n      Initial (untrained) clustering quality:")
    print(f"        Silhouette:  {initial_metrics['silhouette']:.4f}")
    print(f"        Homogeneity: {initial_metrics['homogeneity']:.4f}")
    print(f"        NMI:         {initial_metrics['nmi']:.4f}")
    
    # Train
    print(f"\n      Training {CONFIG['epochs']} epochs:")
    best_metrics = initial_metrics
    history = []
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        epoch_start = time.time()
        loss = train_gcn(gcn, embeddings, labels, df, optimizer, loss_fn,
                         batch_size=CONFIG['batch_size'])
        epoch_time = time.time() - epoch_start
        
        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == CONFIG['epochs']:
            metrics = evaluate_clustering(gcn, embeddings, labels, df)
            print(f"      Epoch {epoch:2d}/{CONFIG['epochs']:2d} | "
                  f"loss: {loss:.4f} | "
                  f"silh: {metrics['silhouette']:.4f} | "
                  f"hom: {metrics['homogeneity']:.4f} | "
                  f"nmi: {metrics['nmi']:.4f} | "
                  f"{epoch_time:.1f}s")
            
            if metrics['homogeneity'] > best_metrics['homogeneity']:
                best_metrics = metrics
                # Save best
                torch.save(gcn.state_dict(), 
                          Path(CONFIG['artifacts_dir']) / 'gcn_best.pt')
            
            history.append({'epoch': epoch, 'loss': loss, **metrics})
        else:
            print(f"      Epoch {epoch:2d}/{CONFIG['epochs']:2d} | loss: {loss:.4f} | {epoch_time:.1f}s")
    
    # Final
    final_metrics = evaluate_clustering(gcn, embeddings, labels, df)
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nClustering quality (higher is better):")
    print(f"  {'Metric':<15} {'Untrained':>12} {'Final':>12}  {'Improvement':>12}")
    print(f"  {'-'*55}")
    for key in ['silhouette', 'homogeneity', 'nmi']:
        diff = final_metrics[key] - initial_metrics[key]
        print(f"  {key:<15} {initial_metrics[key]:>12.4f} {final_metrics[key]:>12.4f}  {diff:>+12.4f}")
    
    # Save final model
    torch.save(gcn.state_dict(), Path(CONFIG['artifacts_dir']) / 'gcn.pt')
    
    # Save info
    info = {
        'model_type': 'Multi-Relational-GCN',
        'config': {k: v for k, v in CONFIG.items() if k != 'device'},
        'num_params': n_params,
        'relation_weights_learned': gcn.get_relation_weights().tolist(),
        'metrics': {
            'untrained': initial_metrics,
            'final': final_metrics,
            'best': best_metrics,
        },
        'history': history,
        'classes': list(label_encoder.classes_),
    }
    with open(Path(CONFIG['artifacts_dir']) / 'gcn_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nLearned relation importance (which edge type matters most):")
    relation_names = ['content_similarity', 'same_source_ip', 'temporal_proximity', 'same_path']
    weights = gcn.get_relation_weights()
    for name, w in zip(relation_names, weights):
        bar = '█' * int(w * 50)
        print(f"  {name:25s} {w:.4f} {bar}")
    
    print(f"\n✓ Saved gcn.pt and gcn_info.json to {CONFIG['artifacts_dir']}/")


if __name__ == "__main__":
    main()
