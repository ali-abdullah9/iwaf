"""
Train BiLSTM model on unified iWAF dataset.

Outputs:
- artifacts/bilstm.pt, artifacts/bilstm_best.pt  - trained model weights
- artifacts/tokenizer.pkl                        - char tokenizer
- artifacts/label_encoder.pkl                    - label mapping
- artifacts/model_info.json                      - config + metrics

Run from this folder with the dataset splits in ./unified_v2/:
    python train_bilstm.py
"""

import os
import sys
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Reuse the model definition from the ML service so training and inference match.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'ml-service'))
from bilstm_model import IWAFBiLSTM, CharTokenizer, IWAFDataset


# ============================================================
# Config
# ============================================================
CONFIG = {
    'data_dir': './unified_v2',  # relative path for portability
    'output_dir': './artifacts',
    'max_len': 384,
    'embedding_dim': 64,
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.4,
    'meta_features_dim': 12,
    'batch_size': 128,           # bigger on GPU
    'epochs': 25,                # more epochs since GPU is fast
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'patience': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


def main():
    print("=" * 60)
    print("iWAF BiLSTM Training")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    
    # ===== Load data =====
    data_dir = Path(CONFIG['data_dir'])
    train_df = pd.read_csv(data_dir / 'train.csv').fillna("")
    val_df = pd.read_csv(data_dir / 'val.csv').fillna("")
    test_df = pd.read_csv(data_dir / 'test.csv').fillna("")
    adv_test_df = pd.read_csv(data_dir / 'adversarial_test.csv').fillna("")
    real_world_df = pd.read_csv(data_dir / 'real_world_test.csv').fillna("")
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Adversarial test: {len(adv_test_df)}, Real-world test: {len(real_world_df)}")
    
    # ===== Label encoding =====
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df['label'])
    label_to_idx = {l: int(i) for i, l in enumerate(label_encoder.classes_)}
    idx_to_label = {i: l for l, i in label_to_idx.items()}
    num_classes = len(label_encoder.classes_)
    print(f"Classes: {list(label_encoder.classes_)}")
    
    # ===== Tokenizer =====
    tokenizer = CharTokenizer(max_len=CONFIG['max_len'])
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # ===== Datasets =====
    print("\nBuilding datasets...")
    train_ds = IWAFDataset(train_df, tokenizer, label_to_idx)
    val_ds = IWAFDataset(val_df, tokenizer, label_to_idx)
    test_ds = IWAFDataset(test_df, tokenizer, label_to_idx)
    adv_test_ds = IWAFDataset(adv_test_df, tokenizer, label_to_idx)
    real_world_ds = IWAFDataset(real_world_df, tokenizer, label_to_idx)
    
    # ===== Class weights =====
    y_train = np.array([label_to_idx[l] for l in train_df['label']])
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=CONFIG['device'])
    print(f"Class weights: {dict(zip(label_encoder.classes_, class_weights.round(3)))}")
    
    # ===== DataLoaders =====
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    adv_loader = DataLoader(adv_test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    real_loader = DataLoader(real_world_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    # ===== Model =====
    model = IWAFBiLSTM(
        vocab_size=tokenizer.vocab_size,
        num_classes=num_classes,
        embedding_dim=CONFIG['embedding_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        meta_features_dim=CONFIG['meta_features_dim'],
        dropout=CONFIG['dropout'],
        pad_idx=tokenizer.pad_idx,
    ).to(CONFIG['device'])
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: BiLSTM with {n_params:,} parameters")
    
    # ===== Loss + Optimizer =====
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # ===== Training loop =====
    print(f"\n{'='*60}")
    print(f"TRAINING ({CONFIG['epochs']} epochs max, patience={CONFIG['patience']})")
    print(f"{'='*60}")
    
    best_val_acc = 0
    patience_counter = 0
    history = []
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        epoch_start = time.time()
        
        for batch_idx, (x, meta, y) in enumerate(train_loader):
            x, meta, y = x.to(CONFIG['device']), meta.to(CONFIG['device']), y.to(CONFIG['device'])
            
            optimizer.zero_grad()
            logits = model(x, meta)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
        
        train_acc = train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for x, meta, y in val_loader:
                x, meta, y = x.to(CONFIG['device']), meta.to(CONFIG['device']), y.to(CONFIG['device'])
                logits = model(x, meta)
                loss = criterion(logits, y)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start
        
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch:2d}/{CONFIG['epochs']:2d} | "
              f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | "
              f"time: {epoch_time:.1f}s")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        })
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            output_dir = Path(CONFIG['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_dir / 'bilstm_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {CONFIG['patience']} epochs)")
                break
    
    print(f"\n✅ Training complete. Best val accuracy: {best_val_acc:.4f}")
    
    # ===== Load best model and evaluate =====
    output_dir = Path(CONFIG['output_dir'])
    model.load_state_dict(torch.load(output_dir / 'bilstm_best.pt', weights_only=True))
    model.eval()
    
    def evaluate(loader, name):
        all_preds = []
        all_labels = []
        all_confs = []
        with torch.no_grad():
            for x, meta, y in loader:
                x, meta = x.to(CONFIG['device']), meta.to(CONFIG['device'])
                logits = model(x, meta)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                confs = probs.max(dim=1)[0]
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())
                all_confs.extend(confs.cpu().numpy())
        
        acc = (np.array(all_preds) == np.array(all_labels)).mean()
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(all_labels, all_preds,
              target_names=label_encoder.classes_, zero_division=0))
        return acc, all_preds, all_labels
    
    # Standard test
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    test_acc, test_preds, test_labels = evaluate(test_loader, "Standard Test Set")
    
    # Adversarial test (synthetic bypass patterns)
    adv_acc, adv_preds, adv_labels = evaluate(adv_loader, "Adversarial Test (synthetic bypasses)")
    
    # Real-world test (ModSecurity bypasses captured during red-team testing)
    real_acc, real_preds, real_labels = evaluate(real_loader, "REAL-WORLD TEST (actual ModSec bypasses)")
    
    # Per-bypass-type breakdown for real-world test
    print(f"\n{'='*60}")
    print("REAL-WORLD BYPASS DETECTION RATE")
    print(f"{'='*60}")
    real_pred_labels = [idx_to_label[p] for p in real_preds]
    real_true_labels = [idx_to_label[l] for l in real_labels]
    correct_per_class = {}
    total_per_class = {}
    for true_label, pred_label in zip(real_true_labels, real_pred_labels):
        total_per_class[true_label] = total_per_class.get(true_label, 0) + 1
        if pred_label == true_label:
            correct_per_class[true_label] = correct_per_class.get(true_label, 0) + 1
    for label in total_per_class:
        correct = correct_per_class.get(label, 0)
        total = total_per_class[label]
        print(f"  {label:10s}: {correct}/{total} = {correct/total*100:.1f}% caught")
    
    # ===== Save artifacts =====
    print(f"\n{'='*60}")
    print("SAVING ARTIFACTS")
    print(f"{'='*60}")
    
    # Tokenizer
    with open(output_dir / 'tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Label encoder
    with open(output_dir / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Model info
    info = {
        'model_type': 'BiLSTM-Attention-Meta',
        'vocab_size': tokenizer.vocab_size,
        'num_classes': num_classes,
        'classes': list(label_encoder.classes_),
        'config': {k: v for k, v in CONFIG.items() if k != 'device'},
        'metrics': {
            'best_val_acc': float(best_val_acc),
            'test_acc': float(test_acc),
            'adversarial_acc': float(adv_acc),
            'real_world_acc': float(real_acc),
        },
        'history': history,
        'parameters': n_params,
    }
    
    with open(output_dir / 'model_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'bilstm.pt')
    
    print(f"\n✅ All artifacts saved to {output_dir}/")
    print(f"   bilstm.pt          ({(output_dir / 'bilstm.pt').stat().st_size // 1024} KB)")
    print(f"   tokenizer.pkl      ({(output_dir / 'tokenizer.pkl').stat().st_size // 1024} KB)")
    print(f"   label_encoder.pkl  ({(output_dir / 'label_encoder.pkl').stat().st_size // 1024} KB)")
    print(f"   model_info.json")
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Test accuracy:               {test_acc:.4f}")
    print(f"  Adversarial test accuracy:   {adv_acc:.4f}")
    print(f"  Real-world bypass accuracy:  {real_acc:.4f}  ⭐")


if __name__ == "__main__":
    main()
