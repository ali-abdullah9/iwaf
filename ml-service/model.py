"""
iWAF BiLSTM Inference Module
Drop-in replacement for the old TF-IDF/LinearSVC model.
Same predict() interface, but uses character-level BiLSTM for better detection.
"""

import os
import pickle
import json
import torch
import numpy as np

from bilstm_model import IWAFBiLSTM, build_text_input, build_meta_features

import config


class IWAFClassifier:
    """
    BiLSTM-based HTTP request classifier.
    Loads artifacts at startup, provides fast inference.
    Same interface as the old TF-IDF/LinearSVC model.
    """
    
    def __init__(self, artifacts_dir: str = None):
        self.artifacts_dir = artifacts_dir or config.MODEL_ARTIFACTS_DIR
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.model_info = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load BiLSTM model + tokenizer + label encoder."""
        print(f"[MODEL] Loading BiLSTM artifacts from {self.artifacts_dir}/")
        print(f"[MODEL] Device: {self.device}")
        
        # Load tokenizer
        with open(os.path.join(self.artifacts_dir, "tokenizer.pkl"), "rb") as f:
            self.tokenizer = pickle.load(f)
        print(f"[MODEL] Tokenizer loaded (vocab={self.tokenizer.vocab_size}, max_len={self.tokenizer.max_len})")
        
        # Load label encoder
        with open(os.path.join(self.artifacts_dir, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)
        print(f"[MODEL] Label encoder loaded: {list(self.label_encoder.classes_)}")
        
        # Load model info
        with open(os.path.join(self.artifacts_dir, "model_info.json")) as f:
            self.model_info = json.load(f)
        
        # Build model with same config as training
        cfg = self.model_info['config']
        self.model = IWAFBiLSTM(
            vocab_size=self.tokenizer.vocab_size,
            num_classes=self.model_info['num_classes'],
            embedding_dim=cfg['embedding_dim'],
            hidden_dim=cfg['hidden_dim'],
            num_layers=cfg['num_layers'],
            meta_features_dim=cfg['meta_features_dim'],
            dropout=cfg['dropout'],
            pad_idx=self.tokenizer.pad_idx,
        )
        
        # Load weights
        weights_path = os.path.join(self.artifacts_dir, "bilstm.pt")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(self.artifacts_dir, "bilstm_best.pt")
        
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device, weights_only=True)
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[MODEL] Model loaded: {n_params:,} parameters")
        print(f"[MODEL] Test accuracy: {self.model_info['metrics']['test_acc']:.4f}")
        print(f"[MODEL] Real-world bypass accuracy: {self.model_info['metrics']['real_world_acc']:.4f}")
        print(f"[MODEL] Classes: {self.model_info['classes']}")
    
    def predict(self, method: str, path: str, query: str, body: str) -> dict:
        """
        Run prediction on a request.
        
        Args:
            method: HTTP method
            path: URL path
            query: Query string
            body: Request body
            
        Returns:
            dict with:
                - label: predicted class name (str)
                - confidence: confidence score (float, 0-1)
                - probabilities: dict of class -> probability
                - is_attack: bool
                - should_block: bool (based on confidence threshold)
        """
        # Build features
        text = build_text_input(method, path, query, body)
        meta = build_meta_features(method, path, query, body, text)
        
        # Tokenize
        tokens = torch.tensor(
            np.array([self.tokenizer.tokenize(text)]), 
            dtype=torch.long
        ).to(self.device)
        meta_t = torch.tensor(
            np.array([meta]), 
            dtype=torch.float32
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(tokens, meta_t)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        pred_idx = int(probs.argmax())
        label = str(self.label_encoder.classes_[pred_idx])
        confidence = float(probs[pred_idx])
        
        probabilities = {
            str(self.label_encoder.classes_[i]): round(float(p), 4)
            for i, p in enumerate(probs)
        }
        
        is_attack = label in config.ATTACK_CLASSES
        should_block = is_attack and confidence >= config.CONFIDENCE_THRESHOLD
        
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "is_attack": is_attack,
            "should_block": should_block,
        }
