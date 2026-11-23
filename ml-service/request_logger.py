"""
iWAF Training Data Capture Logger
Captures every incoming HTTP request to a JSONL file for training data collection.
Each line = one JSON object representing one request.

Usage:
    from request_logger import init_logger, log_request
    
    init_logger("captured_attacks.jsonl")
    
    @app.route(...)
    def handler():
        log_request(request, attack_tag="sqlmap")  # tag optional
        ...
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# Logger state
# ============================================================
_log_file_path = None
_lock = threading.Lock()  # Thread-safe writes
_enabled = True


def init_logger(filepath: str = "captured_requests.jsonl", enabled: bool = True):
    """
    Initialize the logger. Call once at startup.
    
    Args:
        filepath: Where to save captured requests (JSONL format)
        enabled: Whether logging is active (set False to disable quickly)
    """
    global _log_file_path, _enabled
    _log_file_path = Path(filepath).absolute()
    _enabled = enabled
    
    # Create parent directory if needed
    _log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[LOGGER] Request logger initialized: {_log_file_path}")
    print(f"[LOGGER] Status: {'ENABLED' if enabled else 'DISABLED'}")


def set_enabled(enabled: bool):
    """Enable or disable logging at runtime."""
    global _enabled
    _enabled = enabled
    print(f"[LOGGER] Logging {'ENABLED' if enabled else 'DISABLED'}")


def log_request(request, attack_tag: str = None, ml_prediction: dict = None):
    """
    Log a single HTTP request to the JSONL file.
    
    Args:
        request: Flask request object
        attack_tag: Optional tag to label the attack source 
                    (e.g., 'sqlmap', 'nikto', 'manual_utf7', 'benign_browsing')
        ml_prediction: Optional dict with ML classification result
    """
    if not _enabled or _log_file_path is None:
        return
    
    try:
        # Extract raw body
        try:
            body = request.get_data(as_text=True) or ""
        except Exception:
            body = ""
        
        # Build the log entry
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "url": str(request.url),
            "path": request.path,
            "query_string": request.query_string.decode("utf-8", errors="ignore") if hasattr(request.query_string, 'decode') else str(request.query_string),
            "body": body[:5000],  # Cap at 5KB to avoid huge logs
            "content_type": request.headers.get("Content-Type", ""),
            "user_agent": request.headers.get("User-Agent", ""),
            "source_ip": request.headers.get("X-Real-Ip", 
                         request.headers.get("X-Forwarded-For", 
                         request.remote_addr or "unknown")),
            "headers": {k: v for k, v in request.headers.items() 
                       if k.lower() not in ['cookie', 'authorization']},  # Exclude sensitive
            "attack_tag": attack_tag,         # You can label batches
            "ml_prediction": ml_prediction,   # If ML ran, include result
            "label": None,                    # To be filled during labeling step
        }
        
        # Thread-safe append
        with _lock:
            with open(_log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
    
    except Exception as e:
        # Never let logging break the main service
        print(f"[LOGGER] Error logging request: {e}")


def get_log_count() -> int:
    """Return current number of logged requests."""
    if not _log_file_path or not _log_file_path.exists():
        return 0
    with open(_log_file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def rotate_log(new_tag: str = None):
    """
    Rotate the log file. Useful between attack runs to separate sources.
    Example: before running sqlmap, rotate with tag='sqlmap'
    """
    global _log_file_path
    if _log_file_path and _log_file_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag_part = f"_{new_tag}" if new_tag else ""
        backup_path = _log_file_path.parent / f"{_log_file_path.stem}_{timestamp}{tag_part}.jsonl"
        _log_file_path.rename(backup_path)
        print(f"[LOGGER] Rotated log to: {backup_path}")


# ============================================================
# Attack tag management
# ============================================================
_current_attack_tag = None

def set_attack_tag(tag: str):
    """
    Set the current attack tag for incoming requests.
    Example: before running sqlmap, call set_attack_tag('sqlmap')
    """
    global _current_attack_tag
    _current_attack_tag = tag
    print(f"[LOGGER] Attack tag set to: {tag}")


def get_attack_tag() -> str:
    """Get current attack tag (used by log_request if none specified)."""
    return _current_attack_tag
