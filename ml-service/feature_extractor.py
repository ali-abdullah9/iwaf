"""
iWAF Feature Extractor
Extracts HTTP request fields for the BiLSTM classifier.
Note: BiLSTM does its own char-level tokenization in model.predict().
"""

def extract_request_fields(request) -> dict:
    """
    Extract structured fields from a Flask request object.
    
    Args:
        request: The incoming HTTP request object
        
    Returns:
        dict with method, path, query_string, body, headers, etc.
    """
    method = request.method.upper()
    url = str(request.url)
    path = request.path or "/"
    
    # Decode query string
    if hasattr(request.query_string, 'decode'):
        query_string = request.query_string.decode("utf-8", errors="ignore")
    else:
        query_string = str(request.query_string)
    
    # Get body
    try:
        body = request.get_data(as_text=True) or ""
    except Exception:
        body = ""
    
    # Get headers
    headers = dict(request.headers) if hasattr(request, 'headers') else {}
    user_agent = headers.get("User-Agent", headers.get("user-agent", ""))
    content_type = headers.get("Content-Type", headers.get("content-type", ""))
    source_ip = headers.get("X-Real-Ip", 
                headers.get("X-Forwarded-For", 
                request.remote_addr or "unknown"))
    
    return {
        "method": method,
        "url": url,
        "path": path,
        "query_string": query_string,
        "body": body,
        "headers": headers,
        "user_agent": user_agent,
        "content_type": content_type,
        "source_ip": source_ip,
    }
