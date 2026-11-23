"""
iWAF ML Inference Service — v3 (bug fixes + capture logger)

FIXES from previous version:
1. Removed dangling `mongo_logs.insert_one()` call that caused 500 errors
2. Integrated request_logger for training data capture
3. Added /_logger/tag/<tag> and /_logger/stats endpoints
4. Made tag endpoints bypass classification (they're internal control routes)

Flow:
  Client → NGINX → ModSecurity (CRS) → [if passes] → This Service
    ├── Extract features from request
    ├── Run ML classifier
    ├── Log to MongoDB Atlas (blocked/allowed collections)
    ├── Log to captured_requests.jsonl (training data capture)
    ├── If malicious (confidence >= threshold) → return 403
    └── If benign → return 418 (forward to application)

Response codes:
  403 = Blocked by ML layer
  418 = Benign, allow through
"""

from flask import Flask, request, jsonify
from datetime import datetime, timezone
import logging
import traceback
from fnmatch import fnmatch

import config
from model import IWAFClassifier
from feature_extractor import extract_request_fields

# ============================================================
# Import capture logger
# ============================================================
try:
    from request_logger import (
        init_logger, log_request, set_attack_tag, get_attack_tag, get_log_count
    )
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False

# ============================================================
# Import Dynamic Rule Engine (DRE)
# ============================================================
try:
    from dre_blueprint import dre_bp
    DRE_AVAILABLE = True
except ImportError as e:
    print(f"[DRE] Not available: {e}")
    DRE_AVAILABLE = False


def is_whitelisted(path: str) -> bool:
    """Check if a request path matches any whitelisted pattern."""
    path = path.rstrip("/") or "/"
    for pattern in config.WHITELISTED_PATHS:
        pattern_clean = pattern.rstrip("/") or "/"
        if fnmatch(path, pattern_clean):
            return True
    return False


def is_internal_route(path: str) -> bool:
    """Internal control routes that should NOT be classified by ML."""
    return (path.startswith("/_logger/") 
            or path.startswith("/dre/")
            or path in ("/health", "/stats"))


# ============================================================
# Initialize Flask App
# ============================================================
app = Flask(__name__)

# Register DRE blueprint (rule suggestions)
if DRE_AVAILABLE:
    app.register_blueprint(dre_bp, url_prefix='/dre')

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("iwaf-ml")

# ============================================================
# Initialize ML Model (loaded once at startup)
# ============================================================
logger.info("Initializing ML classifier...")
classifier = IWAFClassifier()
logger.info("ML classifier ready")

# ============================================================
# Initialize Capture Logger
# ============================================================
if LOGGER_AVAILABLE:
    init_logger("captured_requests.jsonl", enabled=True)
    logger.info("Capture logger initialized")
else:
    logger.warning("request_logger module not found - captures DISABLED")

# ============================================================
# Initialize MongoDB Connection
# ============================================================
mongo_client = None
mongo_db = None
mongo_blocked = None
mongo_allowed = None

def init_mongodb():
    """Initialize MongoDB Atlas connection with separate collections."""
    global mongo_client, mongo_db, mongo_blocked, mongo_allowed
    
    try:
        from pymongo import MongoClient
        
        logger.info("Connecting to MongoDB Atlas...")
        mongo_client = MongoClient(
            config.MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
        mongo_client.admin.command("ping")
        
        mongo_db = mongo_client[config.MONGO_DB]
        mongo_blocked = mongo_db[config.MONGO_COLLECTION_BLOCKED]
        mongo_allowed = mongo_db[config.MONGO_COLLECTION_ALLOWED]
        
        mongo_blocked.create_index("timestamp")
        mongo_blocked.create_index("source_ip")
        mongo_blocked.create_index("prediction.label")
        mongo_allowed.create_index("timestamp")
        mongo_allowed.create_index("source_ip")
        
        logger.info(f"MongoDB connected: {config.MONGO_DB} [{config.MONGO_COLLECTION_BLOCKED}, {config.MONGO_COLLECTION_ALLOWED}]")
        return True
        
    except Exception as e:
        logger.warning(f"MongoDB connection failed: {e}")
        logger.warning("Service will continue WITHOUT logging to MongoDB")
        return False

mongo_available = init_mongodb()


# ============================================================
# MongoDB Logging (FIXED: removed dangling mongo_logs reference)
# ============================================================
def log_to_mongodb(fields: dict, prediction: dict, whitelisted: bool = False):
    """
    Log request to the appropriate MongoDB collection.
    Blocked → ml_blocked_requests, Allowed → ml_allowed_requests.
    """
    if not mongo_available or mongo_blocked is None:
        return
    
    try:
        doc = {
            "timestamp": datetime.now(timezone.utc),
            "source_ip": fields.get("source_ip", "unknown"),
            "request": {
                "method": fields["method"],
                "url": fields["url"],
                "path": fields["path"],
                "query_string": fields["query_string"],
                "body": fields["body"][:2000],
                "user_agent": fields["user_agent"],
                "content_type": fields["content_type"],
            },
            "prediction": {
                "label": prediction["label"],
                "confidence": prediction["confidence"],
                "probabilities": prediction["probabilities"],
                "is_attack": prediction["is_attack"],
                "should_block": prediction["should_block"],
            },
            "whitelisted": whitelisted,
        }
        
        if prediction["should_block"]:
            doc["blocked_by"] = "ml_layer"
            doc["attack_type"] = prediction["label"]
            mongo_blocked.insert_one(doc)
        else:
            mongo_allowed.insert_one(doc)
    except Exception as e:
        logger.error(f"MongoDB logging failed: {e}")


# ============================================================
# Logger Control Endpoints (registered FIRST to take priority)
# ============================================================
@app.route("/_logger/tag/<tag>", methods=["GET", "POST"])
def set_tag_endpoint(tag):
    """Set the current attack tag for capture logging."""
    if LOGGER_AVAILABLE:
        set_attack_tag(tag)
        return jsonify({"tag": tag, "status": "set"}), 200
    return jsonify({"error": "logger not available"}), 503


@app.route("/_logger/stats", methods=["GET"])
def logger_stats():
    """Check how many requests have been captured."""
    if LOGGER_AVAILABLE:
        return jsonify({
            "captured_requests": get_log_count(),
            "current_tag": get_attack_tag(),
            "file": "captured_requests.jsonl",
        }), 200
    return jsonify({"error": "logger not available"}), 503


@app.route("/_logger/disable", methods=["POST"])
def logger_disable():
    if LOGGER_AVAILABLE:
        from request_logger import set_enabled
        set_enabled(False)
        return jsonify({"status": "disabled"}), 200
    return jsonify({"error": "logger not available"}), 503


@app.route("/_logger/enable", methods=["POST"])
def logger_enable():
    if LOGGER_AVAILABLE:
        from request_logger import set_enabled
        set_enabled(True)
        return jsonify({"status": "enabled"}), 200
    return jsonify({"error": "logger not available"}), 503


# ============================================================
# Health & Stats
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": classifier.model is not None,
        "mongodb_connected": mongo_available,
        "logger_available": LOGGER_AVAILABLE,
        "captures_so_far": get_log_count() if LOGGER_AVAILABLE else 0,
        "model_info": classifier.model_info,
        "config": {
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "response_block": config.RESPONSE_BLOCK,
            "response_allow": config.RESPONSE_ALLOW,
        }
    }), 200


@app.route("/stats", methods=["GET"])
def stats():
    """Get ML classification statistics from MongoDB."""
    if not mongo_available:
        return jsonify({"error": "MongoDB not connected"}), 503
    
    try:
        blocked_pipeline = [
            {"$group": {
                "_id": "$prediction.label",
                "count": {"$sum": 1},
                "avg_confidence": {"$avg": "$prediction.confidence"},
            }},
            {"$sort": {"count": -1}}
        ]
        blocked_results = list(mongo_blocked.aggregate(blocked_pipeline))
        total_blocked = mongo_blocked.count_documents({})
        total_allowed = mongo_allowed.count_documents({})
        total_whitelisted = mongo_allowed.count_documents({"whitelisted": True})
        total_ml_allowed = total_allowed - total_whitelisted
        
        return jsonify({
            "total_requests": total_blocked + total_allowed,
            "total_blocked": total_blocked,
            "total_allowed": total_allowed,
            "total_whitelisted": total_whitelisted,
            "total_ml_classified_allowed": total_ml_allowed,
            "blocked_breakdown": [
                {
                    "attack_type": r["_id"],
                    "count": r["count"],
                    "avg_confidence": round(r["avg_confidence"], 4),
                }
                for r in blocked_results
            ]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# MAIN ROUTE: Classify all other requests
# ============================================================
@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
def classify_request(path):
    """
    Main endpoint: receives ALL traffic forwarded by NGINX after ModSecurity.
    Classifies the request and returns 403 (block) or 418 (allow).
    """
    try:
        req_path = request.path or "/"
        
        # Skip internal routes (shouldn't reach here, but just in case)
        if is_internal_route(req_path):
            return jsonify({"error": "internal route collision"}), 404
        
        # Step 0a: Bypass ML for multipart file uploads
        # ML model is purpose-built for text-based injection attacks (SQLi, XSS,
        # LFI, CMDi). Binary file content (PNGs, JPEGs, etc.) contains random
        # byte sequences that look like injection patterns to a character-level
        # model, causing false positives. File-content security is delegated to
        # ModSecurity rules and application-layer scanners — the architecturally
        # correct place for it.
        content_type = request.headers.get('Content-Type', '').lower()
        if content_type.startswith('multipart/form-data'):
            fields = extract_request_fields(request)
            multipart_prediction = {
                "label": "benign",
                "confidence": 1.0,
                "probabilities": {"benign": 1.0},
                "is_attack": False,
                "should_block": False,
                "bypass_reason": "multipart_upload",
            }
            log_to_mongodb(fields, multipart_prediction, whitelisted=True)
            
            if LOGGER_AVAILABLE:
                log_request(request, attack_tag=get_attack_tag(),
                           ml_prediction=multipart_prediction)
            
            return jsonify({
                "status": "allowed",
                "classification": "multipart_bypass",
                "confidence": 1.0,
            }), config.RESPONSE_ALLOW
        
        # Step 0b: Check whitelist — bypass ML for known-good paths
        if is_whitelisted(req_path):
            fields = extract_request_fields(request)
            whitelisted_prediction = {
                "label": "benign",
                "confidence": 1.0,
                "probabilities": {"benign": 1.0},
                "is_attack": False,
                "should_block": False,
            }
            log_to_mongodb(fields, whitelisted_prediction, whitelisted=True)
            
            # Capture for training data too (tagged as benign/whitelisted)
            if LOGGER_AVAILABLE:
                log_request(request, attack_tag=get_attack_tag(), 
                           ml_prediction=whitelisted_prediction)
            
            return jsonify({
                "status": "allowed",
                "classification": "whitelisted",
                "confidence": 1.0,
            }), config.RESPONSE_ALLOW
        
        # Step 1: Extract features
        fields = extract_request_fields(request)
        
        # Step 2: Run ML prediction (BiLSTM takes raw fields, does its own tokenization)
        prediction = classifier.predict(
            method=fields["method"],
            path=fields["path"],
            query=fields["query_string"],
            body=fields["body"],
        )
        
        # Step 4: Log to MongoDB
        log_to_mongodb(fields, prediction)
        
        # Step 5: Capture for training data
        if LOGGER_AVAILABLE:
            log_request(request, attack_tag=get_attack_tag(), ml_prediction=prediction)
        
        # Step 6: Console log
        log_level = "WARNING" if prediction["should_block"] else "INFO"
        logger.log(
            getattr(logging, log_level),
            f"{fields['source_ip']} - {fields['method']} {fields['path']} "
            f"→ {prediction['label'].upper()} ({prediction['confidence']:.3f}) "
            f"{'[BLOCKED]' if prediction['should_block'] else '[ALLOWED]'}"
        )
        
        # Step 7: Return response
        if prediction["should_block"]:
            return jsonify({
                "status": "blocked",
                "blocked_by": "ml_layer",
                "attack_type": prediction["label"],
                "confidence": prediction["confidence"],
            }), config.RESPONSE_BLOCK
        else:
            return jsonify({
                "status": "allowed",
                "classification": prediction["label"],
                "confidence": prediction["confidence"],
            }), config.RESPONSE_ALLOW
    
    except Exception as e:
        # Fail-open on unexpected errors
        logger.error(f"Classification error: {e}")
        logger.error(traceback.format_exc())
        
        # Still try to capture the request that crashed (so we can learn from it)
        if LOGGER_AVAILABLE:
            try:
                log_request(request, attack_tag=(get_attack_tag() or "") + "_crashed")
            except:
                pass
        
        return jsonify({
            "status": "allowed",
            "classification": "error",
            "confidence": 0.0,
            "error": "ML service internal error - fail-open applied",
        }), config.RESPONSE_ALLOW


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    logger.info(f"Starting iWAF ML Service on {config.SERVICE_HOST}:{config.SERVICE_PORT}")
    logger.info(f"Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    logger.info(f"Block response: {config.RESPONSE_BLOCK}, Allow response: {config.RESPONSE_ALLOW}")
    logger.info(f"Capture logger: {'ENABLED' if LOGGER_AVAILABLE else 'DISABLED'}")
    
    app.run(
        host=config.SERVICE_HOST,
        port=config.SERVICE_PORT,
        debug=False,
        threaded=True,   # Allow concurrent requests
    )
