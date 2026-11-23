"""
iWAF ML Service Configuration
"""
import os

# Load secrets from the project-level .env file (python-dotenv).
# load_dotenv() searches upwards from this file for the nearest .env.
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# MongoDB Atlas Configuration
# ============================================================
MONGO_URI = os.getenv("MONGODB_URI")
MONGO_DB = os.environ.get("MONGO_DB", "waf_db")
MONGO_COLLECTION_BLOCKED = "ml_blocked_requests"
MONGO_COLLECTION_ALLOWED = "ml_allowed_requests"
MONGO_COLLECTION_RULE_SUGGESTIONS = "ml_rule_suggestions"

# ============================================================
# ML Model Configuration
# ============================================================
MODEL_ARTIFACTS_DIR = os.environ.get("MODEL_ARTIFACTS_DIR", "artifacts")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.70"))

ATTACK_CLASSES = ["sqli", "xss", "lfi", "cmdi", "probe"]
BENIGN_CLASS = "benign"

# ============================================================
# HTTP Response Codes
# ============================================================
RESPONSE_BLOCK = 403
RESPONSE_ALLOW = 418

# ============================================================
# Service Configuration
# ============================================================
SERVICE_HOST = os.environ.get("SERVICE_HOST", "0.0.0.0")
SERVICE_PORT = int(os.environ.get("SERVICE_PORT", "5000"))

# ============================================================
# Logging
# ============================================================
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# ============================================================
# DRE (Dynamic Rule Engine) Configuration
# ============================================================
# Anthropic API for LLM-assisted rule analysis
# Get your key at: https://console.anthropic.com
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")
LLM_PREFERENCE = os.environ.get("LLM_PREFERENCE", "auto")

# DRE training data (for false-positive guard)
DRE_TRAIN_CSV = os.environ.get("DRE_TRAIN_CSV", "unified_v2/train.csv")

# ============================================================
# Application Whitelist (paths bypass ML classification)
# ============================================================
WHITELISTED_PATHS = [
    # Root and common navigation (was getting wrongly classified as probe)
    "/",
    "/index.html",
    "/index.php",
    "/home",
    "/about",
    "/contact",
    "/help",
    "/support",
    "/login",
    "/register",
    "/signup",
    "/logout",
    "/account",
    "/profile",
    "/settings",
    "/dashboard",
    "/cart",
    "/checkout",
    "/orders",
    "/wishlist",
    "/sitemap.xml",
    "/robots.txt",
    "/favicon.ico",
    
    # Juice Shop endpoints
    "/rest/admin/application-version",
    "/rest/admin/application-configuration",
    "/rest/languages",
    "/rest/products/*",
    "/rest/basket/*",
    "/rest/user/*",
    "/rest/saveLoginIp",
    "/rest/deluxe-membership",
    "/rest/memories",
    "/rest/chatbot/*",
    "/rest/repeat-notification",
    "/assets/*",
    "/socket.io/*",
    "/api/*",
    
    # Static asset patterns
    "/runtime.js", "/polyfills.js", "/vendor.js", "/main.js", "/styles.css",
    "/*.js", "/*.css", "/*.ico", "/*.png", "/*.jpg", "/*.svg",
    "/*.woff", "/*.woff2", "/*.ttf", "/*.map",
]
