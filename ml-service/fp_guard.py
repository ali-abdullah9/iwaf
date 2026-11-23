"""
False Positive Guard

Before suggesting a rule to the admin, validate it against benign traffic
samples to estimate its false-positive rate. Rules that match too much
legitimate traffic get downgraded to LOW confidence with warnings.
"""

import re
from typing import List
import pandas as pd


class FalsePositiveGuard:
    """
    Tests proposed rules against samples of known-benign traffic.
    """
    
    def __init__(self, benign_samples: List[dict] = None, max_fp_rate: float = 0.005):
        """
        Args:
            benign_samples: List of dicts with method/path/query_string/body
                            (from train.csv with label=='benign')
            max_fp_rate: Max acceptable false positive rate (default 0.5%)
        """
        self.benign_samples = benign_samples or []
        self.max_fp_rate = max_fp_rate
    
    @classmethod
    def from_csv(cls, train_csv_path: str, n_samples: int = 1000, max_fp_rate: float = 0.005):
        """Build guard from training CSV."""
        df = pd.read_csv(train_csv_path).fillna("")
        benign_df = df[df['label'] == 'benign']
        
        # Sample to keep it fast
        if len(benign_df) > n_samples:
            benign_df = benign_df.sample(n=n_samples, random_state=42)
        
        samples = benign_df[['method', 'path', 'query_string', 'body']].to_dict('records')
        return cls(benign_samples=samples, max_fp_rate=max_fp_rate)
    
    def evaluate_pattern(self, pattern: str, targets: str = "ARGS|REQUEST_BODY") -> dict:
        """
        Test a regex pattern against benign samples.
        
        Returns:
            dict with:
                - false_positive_count: int
                - false_positive_rate: float
                - sample_size: int
                - verdict: 'SAFE' | 'RISKY' | 'DANGEROUS'
                - matching_samples: list of matched benign requests (up to 5)
        """
        if not pattern or pattern == ".*":
            return {
                'false_positive_count': 0,
                'false_positive_rate': 0.0,
                'sample_size': len(self.benign_samples),
                'verdict': 'NO_PATTERN',
                'matching_samples': [],
            }
        
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return {
                'false_positive_count': 0,
                'false_positive_rate': 0.0,
                'sample_size': len(self.benign_samples),
                'verdict': 'INVALID_REGEX',
                'error': str(e),
                'matching_samples': [],
            }
        
        # Determine which fields to test based on `targets`
        target_fields = self._parse_targets(targets)
        
        matches = []
        for sample in self.benign_samples:
            content = self._extract_content(sample, target_fields)
            if regex.search(content):
                matches.append(sample)
        
        n = len(self.benign_samples)
        fp_rate = len(matches) / n if n else 0.0
        
        # Verdict
        if fp_rate == 0:
            verdict = 'SAFE'
        elif fp_rate <= self.max_fp_rate:
            verdict = 'ACCEPTABLE'
        elif fp_rate <= self.max_fp_rate * 5:
            verdict = 'RISKY'
        else:
            verdict = 'DANGEROUS'
        
        # Truncate matching samples for readability
        matching_truncated = []
        for m in matches[:5]:
            matching_truncated.append({
                'method': m.get('method'),
                'path': m.get('path', '')[:60],
                'query_string': str(m.get('query_string', ''))[:80],
                'body': str(m.get('body', ''))[:80],
            })
        
        return {
            'false_positive_count': len(matches),
            'false_positive_rate': round(fp_rate, 4),
            'sample_size': n,
            'verdict': verdict,
            'max_acceptable_rate': self.max_fp_rate,
            'matching_samples': matching_truncated,
        }
    
    def _parse_targets(self, targets: str) -> list:
        """Parse ModSec target string into list of fields."""
        targets = targets.upper()
        fields = []
        if 'ARGS' in targets or 'QUERY' in targets:
            fields.append('query_string')
        if 'BODY' in targets:
            fields.append('body')
        if 'URI' in targets or 'URL' in targets or 'REQUEST_LINE' in targets:
            fields.append('path')
        if not fields:
            fields = ['query_string', 'body', 'path']
        return fields
    
    def _extract_content(self, sample: dict, fields: list) -> str:
        """Extract relevant fields as combined string."""
        parts = []
        for field in fields:
            v = str(sample.get(field, '') or '')
            if v and v != 'nan':
                parts.append(v)
        return ' '.join(parts)
    
    def adjust_rule_confidence(self, rule: dict, fp_evaluation: dict) -> dict:
        """
        Update a rule's confidence based on FP evaluation.
        Modifies in place and returns.
        """
        verdict = fp_evaluation['verdict']
        rule['fp_evaluation'] = fp_evaluation
        
        # Original confidence from cluster size
        original_confidence = rule.get('confidence', 'MEDIUM')
        
        # Downgrade based on FP risk
        if verdict == 'DANGEROUS':
            rule['confidence'] = 'LOW'
            rule['warning'] = (
                f"⚠️ FP RISK: Pattern matches {fp_evaluation['false_positive_rate']*100:.2f}% "
                f"of benign traffic. DO NOT deploy without revision."
            )
            # Also change action to log-only
            if 'rule_text' in rule:
                rule['rule_text'] = rule['rule_text'].replace('deny,log', 'log,pass')
        elif verdict == 'RISKY':
            # Downgrade one notch
            confidence_map = {'HIGH': 'MEDIUM', 'MEDIUM': 'LOW', 'LOW': 'LOW'}
            rule['confidence'] = confidence_map.get(original_confidence, 'LOW')
            rule['warning'] = (
                f"⚠️ Some FP risk: matches {fp_evaluation['false_positive_rate']*100:.2f}% "
                f"of benign traffic. Review carefully before deploying."
            )
        elif verdict == 'INVALID_REGEX':
            rule['confidence'] = 'LOW'
            rule['warning'] = f"❌ Invalid regex: {fp_evaluation.get('error', 'unknown')}"
        elif verdict == 'NO_PATTERN':
            rule['confidence'] = 'LOW'
            rule['warning'] = "⚠️ No specific regex pattern - human analysis required"
        # SAFE/ACCEPTABLE: keep original confidence
        
        return rule
