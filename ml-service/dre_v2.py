"""
iWAF Dynamic Rule Engine v2 - Full Pipeline

Architecture:
    Blocked requests (MongoDB)
        ↓
    Reuse trained BiLSTM → 256-dim embeddings
        ↓
    Multi-relational graph (content, IP, time, path)
        ↓
    Trained GCN → graph-aware embeddings (64-dim)
        ↓
    K-Means clustering
        ↓
    LLM-assisted rule analysis (Anthropic / Ollama / Heuristic)
        ↓
    False-positive guard (validates against benign samples)
        ↓
    SecRule suggestions saved to MongoDB
"""

import os
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch

import config
from bilstm_model import IWAFBiLSTM, CharTokenizer
from embedding_extractor import extract_embeddings
from gcn_model import MultiRelationalGCN, build_multi_relational_graph
from llm_analyzer import get_llm_provider, build_secrule_from_llm
from fp_guard import FalsePositiveGuard

logger = logging.getLogger("iwaf-dre")


class DynamicRuleEngineV2:
    """
    Full DRE pipeline with BiLSTM embeddings, multi-relational GCN, 
    LLM-assisted rule analysis, and false-positive guard.
    """
    
    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        train_csv: Optional[str] = "unified_v2/train.csv",
        mongo_uri: Optional[str] = None,
        db_name: str = "iwaf",
        collection: str = "ml_blocked_requests",
        llm_preference: str = "auto",
        device: str = None,
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.train_csv = train_csv
        self.mongo_uri = mongo_uri or config.MONGO_URI
        self.db_name = db_name
        self.collection_name = collection
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Lazy-load components
        self.bilstm = None
        self.tokenizer = None
        self.gcn = None
        self.label_encoder = None
        self.llm = None
        self.fp_guard = None
        
        self.mongo_client = None
        self.db = None
        self.collection = None
        
        # Initialize
        self._load_models()
        self._init_llm(llm_preference)
        self._init_fp_guard()
        self._init_mongo()
    
    def _load_models(self):
        """Load BiLSTM and GCN models from disk."""
        logger.info(f"Loading models from {self.artifacts_dir}/")
        
        # Tokenizer
        with open(self.artifacts_dir / 'tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # Label encoder
        with open(self.artifacts_dir / 'label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Model info
        with open(self.artifacts_dir / 'model_info.json') as f:
            info = json.load(f)
        cfg = info['config']
        
        # BiLSTM
        self.bilstm = IWAFBiLSTM(
            vocab_size=self.tokenizer.vocab_size,
            num_classes=info['num_classes'],
            embedding_dim=cfg['embedding_dim'],
            hidden_dim=cfg['hidden_dim'],
            num_layers=cfg['num_layers'],
            meta_features_dim=cfg['meta_features_dim'],
            dropout=cfg['dropout'],
            pad_idx=self.tokenizer.pad_idx,
        )
        weights = torch.load(self.artifacts_dir / 'bilstm.pt', 
                             map_location=self.device, weights_only=True)
        self.bilstm.load_state_dict(weights)
        self.bilstm = self.bilstm.to(self.device)
        self.bilstm.eval()
        logger.info(f"BiLSTM loaded ({sum(p.numel() for p in self.bilstm.parameters()):,} params)")
        
        # GCN (optional - falls back if not trained yet)
        gcn_path = self.artifacts_dir / 'gcn.pt'
        if gcn_path.exists():
            try:
                with open(self.artifacts_dir / 'gcn_info.json') as f:
                    gcn_info = json.load(f)
                gcn_cfg = gcn_info['config']
                
                self.gcn = MultiRelationalGCN(
                    input_dim=gcn_cfg['gcn_input_dim'],
                    hidden_dim=gcn_cfg['gcn_hidden_dim'],
                    embedding_dim=gcn_cfg['gcn_embedding_dim'],
                    num_relations=gcn_cfg['num_relations'],
                    dropout=gcn_cfg['dropout'],
                )
                gcn_weights = torch.load(gcn_path, map_location=self.device, weights_only=True)
                self.gcn.load_state_dict(gcn_weights)
                self.gcn = self.gcn.to(self.device)
                self.gcn.eval()
                logger.info(f"GCN loaded ({sum(p.numel() for p in self.gcn.parameters()):,} params)")
            except Exception as e:
                logger.warning(f"Failed to load GCN, will use raw BiLSTM embeddings: {e}")
                self.gcn = None
        else:
            logger.warning(f"No trained GCN at {gcn_path} - using BiLSTM embeddings directly")
    
    def _init_llm(self, preference: str):
        """Initialize LLM provider."""
        self.llm = get_llm_provider(preference=preference)
        logger.info(f"LLM provider: {type(self.llm).__name__}")
    
    def _init_fp_guard(self):
        """Initialize false-positive guard."""
        if self.train_csv and Path(self.train_csv).exists():
            self.fp_guard = FalsePositiveGuard.from_csv(self.train_csv, n_samples=500)
            logger.info(f"FP guard loaded with {len(self.fp_guard.benign_samples)} benign samples")
        else:
            logger.warning(f"No train CSV at {self.train_csv} - FP guard disabled")
            self.fp_guard = None
    
    def _init_mongo(self):
        """Initialize MongoDB connection."""
        try:
            from pymongo import MongoClient
            self.mongo_client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            self.mongo_client.admin.command('ping')
            self.db = self.mongo_client[self.db_name]
            self.collection = self.db[self.collection_name]
            logger.info("MongoDB connected")
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e}")
            self.mongo_client = None
    
    # ========================================================
    # Pipeline steps
    # ========================================================
    def fetch_blocked_requests(self, hours_back: int = 24, limit: int = 500) -> pd.DataFrame:
        """Fetch recent blocked requests from MongoDB."""
        if self.collection is None:
            logger.error("MongoDB not available")
            return pd.DataFrame()
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        cursor = self.collection.find(
            {'timestamp': {'$gte': cutoff}}
        ).sort('timestamp', -1).limit(limit)
        
        rows = []
        for doc in cursor:
            req = doc.get('request', {})
            pred = doc.get('prediction', {})
            rows.append({
                '_id': str(doc.get('_id', '')),
                'method': req.get('method', 'GET'),
                'path': req.get('path', '/'),
                'query_string': req.get('query_string', '') or '',
                'body': req.get('body', '') or '',
                'source_ip': doc.get('source_ip', 'unknown'),
                'user_agent': req.get('user_agent', '') or '',
                'timestamp': doc.get('timestamp'),
                'attack_type': pred.get('label', 'unknown'),
                'confidence': pred.get('confidence', 0.0),
            })
        
        df = pd.DataFrame(rows)
        logger.info(f"Fetched {len(df)} blocked requests")
        return df
    
    def get_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """Extract embeddings via BiLSTM (and optionally GCN)."""
        if len(df) == 0:
            return np.array([])
        
        # BiLSTM embeddings
        bilstm_emb = extract_embeddings(self.bilstm, self.tokenizer, df,
                                        device=self.device, batch_size=32)
        
        # If GCN is loaded, refine via GCN
        if self.gcn is not None:
            adj_list = build_multi_relational_graph(bilstm_emb, df,
                                                    similarity_threshold=0.7,
                                                    time_window_minutes=30)
            x = torch.tensor(bilstm_emb, dtype=torch.float32, device=self.device)
            adj_tensors = [torch.tensor(a, dtype=torch.float32, device=self.device) 
                           for a in adj_list]
            
            with torch.no_grad():
                gcn_emb = self.gcn(x, adj_tensors).cpu().numpy()
            return gcn_emb
        
        return bilstm_emb
    
    def cluster(self, embeddings: np.ndarray, min_cluster_size: int = 2) -> np.ndarray:
        """Cluster embeddings into groups via K-Means + silhouette."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        n = len(embeddings)
        if n < 4:
            return np.arange(n)  # too few for clustering
        
        max_k = min(10, n // 2)
        best_k = 2
        best_score = -1
        
        for k in range(2, max_k + 1):
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(embeddings)
                if len(set(labels)) > 1:
                    score = silhouette_score(embeddings, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            except Exception:
                continue
        
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        return km.fit_predict(embeddings)
    
    def build_cluster_summary(self, cluster_id: int, cluster_df: pd.DataFrame) -> dict:
        """Build summary dict for LLM analysis."""
        attack_types = Counter(cluster_df['attack_type'])
        primary_attack = attack_types.most_common(1)[0][0] if attack_types else 'unknown'
        
        # Sample requests for LLM (up to 8)
        sample_requests = cluster_df.head(8)[['method', 'path', 'query_string', 'body']].to_dict('records')
        
        return {
            'cluster_id': cluster_id,
            'cluster_size': len(cluster_df),
            'primary_attack_type': primary_attack,
            'attack_distribution': dict(attack_types),
            'method_distribution': dict(Counter(cluster_df['method'])),
            'unique_ips': cluster_df['source_ip'].nunique(),
            'top_source_ips': dict(Counter(cluster_df['source_ip']).most_common(5)),
            'sample_requests': sample_requests,
        }
    
    def analyze(self, hours_back: int = 24) -> dict:
        """
        Run the full DRE pipeline.
        
        Returns:
            dict with status, suggestions list, metadata
        """
        logger.info(f"Starting DRE analysis (last {hours_back}h)...")
        
        # Step 1: Fetch
        df = self.fetch_blocked_requests(hours_back=hours_back)
        if len(df) < 4:
            return {
                'status': 'insufficient_data',
                'message': f'Only {len(df)} blocked requests in last {hours_back}h. Need at least 4.',
                'suggestions': [],
                'analyzed_requests': len(df),
            }
        
        # Step 2: Embeddings
        logger.info(f"Extracting embeddings for {len(df)} requests...")
        embeddings = self.get_embeddings(df)
        
        # Step 3: Cluster
        logger.info("Clustering...")
        cluster_labels = self.cluster(embeddings)
        
        # Group by cluster
        df['cluster_id'] = cluster_labels
        clusters_info = df.groupby('cluster_id').size().to_dict()
        logger.info(f"Found {len(clusters_info)} clusters: {clusters_info}")
        
        # Step 4: For each cluster, run LLM analysis + FP guard
        suggestions = []
        rule_id_base = 9000000 + (datetime.now().day * 1000)
        
        for cluster_id in sorted(clusters_info.keys()):
            cluster_df = df[df['cluster_id'] == cluster_id]
            
            if len(cluster_df) < 2:
                continue  # skip singleton clusters
            
            # Build summary
            summary = self.build_cluster_summary(cluster_id, cluster_df)
            
            # LLM analysis
            logger.info(f"Cluster {cluster_id} ({summary['cluster_size']} requests, "
                        f"{summary['primary_attack_type']}): asking LLM...")
            llm_analysis = self.llm.analyze_cluster(summary)
            
            if llm_analysis is None:
                logger.warning(f"LLM returned no analysis for cluster {cluster_id}, using heuristic")
                from llm_analyzer import HeuristicProvider
                llm_analysis = HeuristicProvider().analyze_cluster(summary)
            
            # Build SecRule
            rule_id = rule_id_base + cluster_id
            rule_text = build_secrule_from_llm(summary, llm_analysis, rule_id)
            
            # Initial confidence based on cluster size + LLM signals
            cluster_size = summary['cluster_size']
            llm_fp_risk = llm_analysis.get('false_positive_risk', 'MEDIUM')
            
            if cluster_size >= 10 and llm_fp_risk == 'LOW':
                confidence = 'HIGH'
            elif cluster_size >= 5 and llm_fp_risk in ('LOW', 'MEDIUM'):
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
            
            rule_obj = {
                'rule_id': rule_id,
                'cluster_id': int(cluster_id),
                'cluster_size': cluster_size,
                'attack_type': summary['primary_attack_type'],
                'technique_name': llm_analysis.get('technique_name', ''),
                'description': llm_analysis.get('technique_description', ''),
                'regex_pattern': llm_analysis.get('regex_pattern', ''),
                'severity': llm_analysis.get('suggested_severity', 'WARNING'),
                'targets': llm_analysis.get('rule_targets', 'ARGS'),
                'rule_text': rule_text,
                'confidence': confidence,
                'llm_fp_risk': llm_fp_risk,
                'cluster_summary': summary,
                'llm_analysis': llm_analysis,
            }
            
            # Step 5: FP guard
            if self.fp_guard and llm_analysis.get('regex_pattern'):
                fp_eval = self.fp_guard.evaluate_pattern(
                    llm_analysis['regex_pattern'],
                    targets=llm_analysis.get('rule_targets', 'ARGS')
                )
                rule_obj = self.fp_guard.adjust_rule_confidence(rule_obj, fp_eval)
                logger.info(f"Cluster {cluster_id}: FP rate={fp_eval['false_positive_rate']*100:.2f}%, "
                            f"verdict={fp_eval['verdict']}, confidence={rule_obj['confidence']}")
            
            suggestions.append(rule_obj)
        
        # Sort by cluster size descending
        suggestions.sort(key=lambda r: r['cluster_size'], reverse=True)
        
        result = {
            'status': 'success',
            'analyzed_requests': len(df),
            'time_window_hours': hours_back,
            'num_clusters': len(clusters_info),
            'clusters_with_rules': len(suggestions),
            'llm_provider': type(self.llm).__name__,
            'gcn_used': self.gcn is not None,
            'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
            'suggestions': suggestions,
        }
        
        return result
    
    def save_suggestions_to_mongo(self, results: dict) -> dict:
        """
        Save suggestions to MongoDB with deduplication.
        
        Dedup logic:
        - Check if a suggestion with same (attack_type, technique_name) exists
          and has status 'pending_review' or 'approved'
        - If yes: increment seen_count, update last_seen_at, skip insert
        - If a previously REJECTED suggestion exists for same technique,
          create a new suggestion but include the rejection context
        - If no match: insert as new
        
        Returns summary dict:
            {
                'new': int,           # new suggestions inserted
                'merged': int,        # duplicates merged into existing
                're_suggested': int,  # previously rejected, suggested again
                'total_processed': int,
            }
        """
        summary = {
            'new': 0,
            'merged': 0,
            're_suggested': 0,
            'total_processed': 0,
        }
        
        if self.db is None:
            return summary
        
        col = self.db['ml_rule_suggestions']
        now = datetime.now(timezone.utc)
        
        for rule in results.get('suggestions', []):
            summary['total_processed'] += 1
            
            attack_type = rule.get('attack_type')
            technique_name = rule.get('technique_name', '')
            
            if not attack_type or not technique_name:
                # Can't dedup without these — just insert
                doc = {**rule, 
                       'analyzed_at': now,
                       'first_seen_at': now,
                       'last_seen_at': now,
                       'seen_count': 1,
                       'status': 'pending_review'}
                col.insert_one(doc)
                summary['new'] += 1
                continue
            
            # Dedup key: same attack_type + same technique_name
            dedup_query = {
                'attack_type': attack_type,
                'technique_name': technique_name,
            }
            
            # 1. Check for active (pending or approved) duplicate
            active = col.find_one({
                **dedup_query,
                'status': {'$in': ['pending_review', 'approved']}
            })
            
            if active:
                # MERGE: increment seen_count, update last_seen
                col.update_one(
                    {'_id': active['_id']},
                    {
                        '$inc': {'seen_count': 1},
                        '$set': {
                            'last_seen_at': now,
                            'last_cluster_size': rule.get('cluster_size', 1),
                        },
                    }
                )
                summary['merged'] += 1
                logger.info(
                    f"Merged duplicate: {technique_name} "
                    f"(now seen {active.get('seen_count', 1) + 1} times)"
                )
                continue
            
            # 2. Check for previously rejected — re-suggest with context
            rejected = col.find_one({
                **dedup_query,
                'status': 'rejected'
            })
            
            doc = {
                **rule,
                'analyzed_at': now,
                'first_seen_at': now,
                'last_seen_at': now,
                'seen_count': 1,
                'status': 'pending_review',
            }
            
            if rejected:
                # Add context that this technique was previously rejected
                doc['previously_rejected'] = True
                doc['previous_rejection'] = {
                    'rejected_at': rejected.get('rejected_at'),
                    'reason': rejected.get('rejection_reason', ''),
                }
                summary['re_suggested'] += 1
                logger.info(
                    f"Re-suggesting previously rejected technique: {technique_name}"
                )
            else:
                summary['new'] += 1
            
            col.insert_one(doc)
        
        # Cap at 200 most recent (was 100, increased to handle re-suggestions)
        count = col.count_documents({})
        if count > 200:
            oldest = list(col.find().sort('analyzed_at', 1).limit(count - 200))
            for o in oldest:
                col.delete_one({'_id': o['_id']})
        
        return summary
