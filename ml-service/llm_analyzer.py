"""
LLM-Assisted Rule Analyzer

Sends attack clusters to an LLM for pattern analysis and rule generation.
Designed with a provider abstraction so we can swap Anthropic for local
models (Llama, Phi-3) later if needed.

The LLM is asked to:
    1. Identify the obfuscation/evasion technique shared across the cluster
    2. Generate a precise regex that would match the pattern
    3. Estimate false-positive risk
    4. Suggest rule severity
"""

import os
import json
import re
import logging
from typing import List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger("iwaf-llm")


# ============================================================
# Provider abstraction
# ============================================================
class LLMProvider(ABC):
    """Abstract LLM provider for rule analysis."""
    
    @abstractmethod
    def analyze_cluster(self, cluster_summary: dict) -> Optional[dict]:
        """Analyze a cluster of attacks and suggest a rule."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is configured and reachable."""
        pass


# ============================================================
# Anthropic Provider (Claude)
# ============================================================
class AnthropicProvider(LLMProvider):
    """Uses Claude API for rule analysis."""
    
    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info(f"Anthropic provider initialized: {model}")
            except ImportError:
                logger.warning("anthropic package not installed. Run: pip install anthropic")
            except Exception as e:
                logger.warning(f"Anthropic init failed: {e}")
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def analyze_cluster(self, cluster_summary: dict) -> Optional[dict]:
        """Send cluster to Claude for analysis."""
        if not self.is_available():
            return None
        
        prompt = self._build_prompt(cluster_summary)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            
            response_text = response.content[0].text
            return self._parse_response(response_text)
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return None
    
    def _build_prompt(self, cluster_summary: dict) -> str:
        """Construct prompt for rule analysis."""
        attack_type = cluster_summary['primary_attack_type']
        cluster_size = cluster_summary['cluster_size']
        sample_requests = cluster_summary.get('sample_requests', [])[:8]  # Cap at 8
        
        # Format sample requests cleanly
        samples_text = ""
        for i, req in enumerate(sample_requests, 1):
            method = req.get('method', 'GET')
            path = req.get('path', '/')[:80]
            qs = (req.get('query_string') or '')[:200]
            body = (req.get('body') or '')[:200]
            samples_text += f"  {i}. {method} {path}"
            if qs:
                samples_text += f"?{qs}"
            if body:
                samples_text += f"  [body: {body}]"
            samples_text += "\n"
        
        prompt = f"""You are a security analyst reviewing a cluster of HTTP requests that bypassed a Web Application Firewall (ModSecurity CRS) but were caught by an ML model. Your job is to identify the precise attack pattern so a human admin can write a ModSecurity rule to catch it.

## Cluster Information
- Predicted attack type: **{attack_type}**
- Number of similar requests: {cluster_size}
- Source IPs in cluster: {cluster_summary.get('unique_ips', 'unknown')}
- Common HTTP methods: {list(cluster_summary.get('method_distribution', {}).keys())}

## Sample requests from this cluster:
{samples_text}

## Your Task
Analyze these samples and respond in **strict JSON format** with these fields:

```json
{{
  "technique_name": "Brief name of the obfuscation/evasion technique (e.g., 'HTML Entity Tab Injection in JS Context')",
  "technique_description": "1-2 sentence explanation of how this attack works and why it bypasses standard WAF regex rules",
  "regex_pattern": "A precise regex that would match these specific attack patterns. Be SPECIFIC - avoid generic patterns like 'admin' or 'password' that would cause false positives. Use ModSecurity-compatible regex syntax",
  "regex_explanation": "What the regex matches and why it's specific to this attack",
  "false_positive_risk": "LOW|MEDIUM|HIGH",
  "false_positive_reason": "Why this risk level - what kinds of legitimate traffic might also match",
  "suggested_severity": "CRITICAL|ERROR|WARNING|NOTICE",
  "rule_targets": "ARGS|REQUEST_BODY|REQUEST_URI|REQUEST_HEADERS|REQUEST_LINE - which parts of the request to inspect"
}}
```

Rules:
1. The regex MUST be specific - it should match the cluster's pattern, not random text
2. If the requests share a clear obfuscation marker (UTF-7 +AD-, HTML entities &#x09, MySQL inline comments /*...*/), focus on THAT
3. If the pattern is too diverse to capture in one regex, set false_positive_risk to HIGH and explain
4. Use ModSec-compatible regex (PCRE) syntax
5. Respond with ONLY the JSON, no markdown wrapper, no other text"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Optional[dict]:
        """Parse JSON from LLM response, handling common formatting issues."""
        # Strip markdown code fences if present
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        try:
            data = json.loads(text)
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            return None


# ============================================================
# Local LLM Provider (Ollama-based, optional fallback)
# ============================================================
class OllamaProvider(LLMProvider):
    """Uses local Ollama instance for rule analysis."""
    
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._available = self._check_available()
    
    def _check_available(self) -> bool:
        try:
            import requests
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False
    
    def is_available(self) -> bool:
        return self._available
    
    def analyze_cluster(self, cluster_summary: dict) -> Optional[dict]:
        if not self.is_available():
            return None
        
        try:
            import requests
            
            # Reuse Anthropic prompt builder for consistency
            prompt_builder = AnthropicProvider().__class__._build_prompt
            prompt = prompt_builder(self, cluster_summary)
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.2},
                },
                timeout=60,
            )
            
            if response.status_code != 200:
                return None
            
            result = response.json()
            response_text = result.get("response", "")
            
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                return None
        
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return None


# ============================================================
# Fallback: Heuristic Provider (no LLM)
# ============================================================
class HeuristicProvider(LLMProvider):
    """
    Fallback when no LLM is available. Uses regex-based pattern detection
    to suggest rules. Lower quality but always works.
    """
    
    def is_available(self) -> bool:
        return True
    
    def analyze_cluster(self, cluster_summary: dict) -> Optional[dict]:
        attack_type = cluster_summary['primary_attack_type']
        sample_requests = cluster_summary.get('sample_requests', [])
        
        # Combine all content
        all_content = ""
        for req in sample_requests:
            all_content += " " + (req.get('query_string') or '') + " " + (req.get('body') or '')
        
        # Detect obvious patterns
        if re.search(r'\+AD[w-]', all_content):
            return {
                "technique_name": "UTF-7 Encoding",
                "technique_description": "Uses UTF-7 character encoding to hide attack payloads from WAF regex",
                "regex_pattern": r"\+(ADw|AD4|AEA|AD0|ACI)",
                "regex_explanation": "Matches UTF-7 encoded special characters used in HTML/attack contexts",
                "false_positive_risk": "LOW",
                "false_positive_reason": "UTF-7 encoded special chars rarely appear in legitimate web traffic",
                "suggested_severity": "CRITICAL",
                "rule_targets": "ARGS|REQUEST_BODY",
            }
        
        if re.search(r'&#x?(09|0a|0d)', all_content.lower()):
            return {
                "technique_name": "HTML Entity Tab/Newline Injection",
                "technique_description": "Injects HTML tab/newline entities to break up attack keywords (jav&#x09;ascript)",
                "regex_pattern": r"\w+&#x?(09|0a|0d|0[ad])",
                "regex_explanation": "Matches words split by HTML control character entities",
                "false_positive_risk": "LOW",
                "false_positive_reason": "These entities rarely appear inside legitimate identifiers",
                "suggested_severity": "CRITICAL",
                "rule_targets": "ARGS|REQUEST_BODY|REQUEST_URI",
            }
        
        if re.search(r'/\*[^*]+\*/.*?\b(or|and)\s+\d{6,}\s*=\s*\d{6,}', all_content, re.I):
            return {
                "technique_name": "Inline Comment SQLi with Random Numeric Equality",
                "technique_description": "Uses inline comments and random number equalities to bypass simple SQLi regex",
                "regex_pattern": r"/\*[^*]*\*/\s*(?:or|and)\s+\d{6,}\s*=\s*\d{6,}",
                "regex_explanation": "Matches inline-commented logical operators with large numeric equalities",
                "false_positive_risk": "LOW",
                "false_positive_reason": "This pattern is highly specific to SQLi and unlikely in legitimate input",
                "suggested_severity": "CRITICAL",
                "rule_targets": "ARGS|REQUEST_BODY",
            }
        
        # Fallback: generic pattern based on attack type
        return {
            "technique_name": f"Generic {attack_type.upper()} Pattern (heuristic)",
            "technique_description": f"Cluster of {len(sample_requests)} {attack_type} attacks - manual review needed",
            "regex_pattern": "",
            "regex_explanation": "No specific pattern detected - human analysis required",
            "false_positive_risk": "HIGH",
            "false_positive_reason": "Heuristic could not identify a specific pattern",
            "suggested_severity": "WARNING",
            "rule_targets": "ARGS",
        }


# ============================================================
# Provider Factory
# ============================================================
def get_llm_provider(preference: str = "auto") -> LLMProvider:
    """
    Get the best available LLM provider.
    
    Args:
        preference: "anthropic" | "ollama" | "heuristic" | "auto"
    """
    if preference == "anthropic":
        provider = AnthropicProvider()
        if provider.is_available():
            return provider
        logger.warning("Anthropic not available, falling back")
    
    if preference == "ollama":
        provider = OllamaProvider()
        if provider.is_available():
            return provider
        logger.warning("Ollama not available, falling back")
    
    # Auto: try Anthropic first, then Ollama, then heuristic
    if preference == "auto":
        for cls in [AnthropicProvider, OllamaProvider]:
            provider = cls()
            if provider.is_available():
                logger.info(f"Using {cls.__name__}")
                return provider
        logger.info("No LLM provider available, using heuristic fallback")
    
    return HeuristicProvider()


# ============================================================
# Rule Builder (combines LLM analysis + ModSec template)
# ============================================================
def build_secrule_from_llm(
    cluster_summary: dict, 
    llm_analysis: dict, 
    rule_id: int,
) -> str:
    """
    Build a complete ModSecurity rule from LLM analysis.
    """
    technique = llm_analysis.get('technique_name', 'Unknown Pattern')
    description = llm_analysis.get('technique_description', '')
    pattern = llm_analysis.get('regex_pattern', '')
    fp_risk = llm_analysis.get('false_positive_risk', 'MEDIUM')
    severity = llm_analysis.get('suggested_severity', 'WARNING')
    targets = llm_analysis.get('rule_targets', 'ARGS|REQUEST_BODY')
    cluster_size = cluster_summary.get('cluster_size', 0)
    cluster_id = cluster_summary.get('cluster_id', 0)
    
    # If LLM gave no pattern, fall back to log-only mode
    if not pattern:
        action = "log,pass"
        pattern = ".*"  # placeholder
    elif fp_risk == "HIGH":
        action = "log,pass"   # don't auto-block high-FP rules
    else:
        action = "deny,log"
    
    msg = f"iWAF DRE: {technique}"
    
    rule = f"""# Cluster #{cluster_id} - {technique}
# Description: {description}
# Cluster size: {cluster_size} similar requests
# False-positive risk: {fp_risk}
# Generated by: LLM-assisted rule analysis
SecRule {targets} "@rx {pattern}" \\
    "id:{rule_id},\\
    phase:2,\\
    {action},\\
    severity:'{severity}',\\
    msg:'{msg}',\\
    tag:'iwaf-suggested',\\
    tag:'attack-{cluster_summary.get("primary_attack_type", "unknown")}',\\
    tag:'fp-risk-{fp_risk.lower()}'"
"""
    return rule.strip()
