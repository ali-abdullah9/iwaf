"""
iWAF DRE v2 Flask Blueprint

Endpoints (all under /dre/):
    POST /dre/analyze                    - Trigger DRE analysis
    GET  /dre/suggestions                - List rule suggestions
    GET  /dre/suggestions/<id>           - Get one suggestion
    POST /dre/suggestions/<id>/approve   - Approve suggestion
    POST /dre/suggestions/<id>/reject    - Reject suggestion
    GET  /dre/suggestions/<id>/export    - Export as .conf file
    GET  /dre/stats                      - DRE statistics
    GET  /dre/info                       - DRE configuration info
"""

from flask import Blueprint, request, jsonify, Response
from datetime import datetime, timezone
import logging
import json

import config
from dre_v2 import DynamicRuleEngineV2

logger = logging.getLogger("iwaf-dre")
dre_bp = Blueprint('dre', __name__)


# Lazy-load DRE so service starts fast (model loading takes a few seconds)
_dre_instance = None

def get_dre():
    global _dre_instance
    if _dre_instance is None:
        try:
            _dre_instance = DynamicRuleEngineV2(
                artifacts_dir=config.MODEL_ARTIFACTS_DIR,
                train_csv=config.DRE_TRAIN_CSV,
                mongo_uri=config.MONGO_URI,
                db_name=config.MONGO_DB,
                collection=config.MONGO_COLLECTION_BLOCKED,
                llm_preference=config.LLM_PREFERENCE,
            )
            logger.info("DRE v2 initialized")
        except Exception as e:
            logger.error(f"Failed to initialize DRE v2: {e}")
            import traceback
            traceback.print_exc()
            return None
    return _dre_instance


@dre_bp.route('/info', methods=['GET'])
def dre_info():
    """Return DRE configuration and component status."""
    dre = get_dre()
    if dre is None:
        return jsonify({"error": "DRE not available"}), 503
    
    return jsonify({
        "components": {
            "bilstm_loaded": dre.bilstm is not None,
            "gcn_loaded": dre.gcn is not None,
            "llm_provider": type(dre.llm).__name__,
            "fp_guard_enabled": dre.fp_guard is not None,
            "mongo_connected": dre.collection is not None,
        },
        "config": {
            "device": dre.device,
            "fp_guard_samples": len(dre.fp_guard.benign_samples) if dre.fp_guard else 0,
            "fp_guard_threshold": dre.fp_guard.max_fp_rate if dre.fp_guard else 0,
            "llm_preference": config.LLM_PREFERENCE,
        },
    }), 200


@dre_bp.route('/analyze', methods=['POST'])
def trigger_analysis():
    """
    Trigger a DRE analysis run.
    
    POST body (JSON, all optional):
        {
            "hours": 24,        // time window (1, 24, 168, 720)
            "save": true        // save suggestions to MongoDB (with dedup)
        }
    
    Response includes deduplication summary:
        {
            ...analysis results...,
            "dedup_summary": {
                "new": 2,             // truly new suggestions
                "merged": 3,          // duplicates merged into existing
                "re_suggested": 0,    // previously rejected, suggested again
                "total_processed": 5
            }
        }
    """
    dre = get_dre()
    if dre is None:
        return jsonify({"error": "DRE not available - check logs"}), 503
    
    body = request.get_json(silent=True) or {}
    hours = int(body.get('hours', 24))
    save = body.get('save', True)
    
    try:
        results = dre.analyze(hours_back=hours)
        
        # Save with deduplication
        if save and results.get('status') == 'success':
            dedup_summary = dre.save_suggestions_to_mongo(results)
            results['dedup_summary'] = dedup_summary
            logger.info(
                f"DRE analyze: {dedup_summary['total_processed']} clusters → "
                f"{dedup_summary['new']} new, "
                f"{dedup_summary['merged']} merged, "
                f"{dedup_summary['re_suggested']} re-suggested"
            )
        
        # Make response JSON-serializable
        return jsonify(json.loads(json.dumps(results, default=str))), 200
    
    except Exception as e:
        logger.error(f"DRE analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@dre_bp.route('/suggestions', methods=['GET'])
def list_suggestions():
    """
    List rule suggestions.
    
    Query params:
        ?status=pending_review|approved|rejected
        ?attack_type=sqli|xss|lfi|cmdi|probe
        ?confidence=HIGH|MEDIUM|LOW
        ?limit=20
    """
    dre = get_dre()
    if dre is None:
        return jsonify({"error": "DRE not available"}), 503
    
    if dre.db is None:
        return jsonify({"error": "MongoDB not connected"}), 503
    
    suggestions_col = dre.db[config.MONGO_COLLECTION_RULE_SUGGESTIONS]
    
    query = {}
    status = request.args.get('status', 'pending_review')
    if status:
        query['status'] = status
    
    attack_type = request.args.get('attack_type')
    if attack_type:
        query['attack_type'] = attack_type
    
    confidence = request.args.get('confidence')
    if confidence:
        query['confidence'] = confidence
    
    limit = int(request.args.get('limit', 20))
    
    try:
        suggestions = list(
            suggestions_col.find(query)
            .sort('analyzed_at', -1)
            .limit(limit)
        )
        
        for s in suggestions:
            s['_id'] = str(s['_id'])
            for k, v in s.items():
                if isinstance(v, datetime):
                    s[k] = v.isoformat()
        
        return jsonify({
            "count": len(suggestions),
            "suggestions": suggestions,
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@dre_bp.route('/suggestions/<suggestion_id>', methods=['GET'])
def get_suggestion(suggestion_id):
    """Get details for a single rule suggestion."""
    from bson import ObjectId
    
    dre = get_dre()
    if dre is None or dre.db is None:
        return jsonify({"error": "DRE not available"}), 503
    
    try:
        col = dre.db[config.MONGO_COLLECTION_RULE_SUGGESTIONS]
        suggestion = col.find_one({'_id': ObjectId(suggestion_id)})
        if not suggestion:
            return jsonify({"error": "not found"}), 404
        
        suggestion['_id'] = str(suggestion['_id'])
        for k, v in suggestion.items():
            if isinstance(v, datetime):
                suggestion[k] = v.isoformat()
        
        return jsonify(suggestion), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@dre_bp.route('/suggestions/<suggestion_id>/approve', methods=['POST'])
def approve_suggestion(suggestion_id):
    """Mark a suggestion as approved by admin."""
    from bson import ObjectId
    
    dre = get_dre()
    if dre is None or dre.db is None:
        return jsonify({"error": "DRE not available"}), 503
    
    body = request.get_json(silent=True) or {}
    notes = body.get('notes', '')
    
    try:
        col = dre.db[config.MONGO_COLLECTION_RULE_SUGGESTIONS]
        result = col.update_one(
            {'_id': ObjectId(suggestion_id)},
            {'$set': {
                'status': 'approved',
                'approved_at': datetime.now(timezone.utc),
                'admin_notes': notes,
            }}
        )
        if result.matched_count == 0:
            return jsonify({"error": "not found"}), 404
        return jsonify({"status": "approved"}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@dre_bp.route('/suggestions/<suggestion_id>/reject', methods=['POST'])
def reject_suggestion(suggestion_id):
    """Mark a suggestion as rejected (false positive or unsuitable)."""
    from bson import ObjectId
    
    dre = get_dre()
    if dre is None or dre.db is None:
        return jsonify({"error": "DRE not available"}), 503
    
    body = request.get_json(silent=True) or {}
    reason = body.get('reason', '')
    
    try:
        col = dre.db[config.MONGO_COLLECTION_RULE_SUGGESTIONS]
        result = col.update_one(
            {'_id': ObjectId(suggestion_id)},
            {'$set': {
                'status': 'rejected',
                'rejected_at': datetime.now(timezone.utc),
                'rejection_reason': reason,
            }}
        )
        if result.matched_count == 0:
            return jsonify({"error": "not found"}), 404
        return jsonify({"status": "rejected"}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@dre_bp.route('/suggestions/<suggestion_id>/export', methods=['GET'])
def export_secrule(suggestion_id):
    """Export rule as a downloadable .conf file ready for ModSecurity."""
    from bson import ObjectId
    
    dre = get_dre()
    if dre is None or dre.db is None:
        return jsonify({"error": "DRE not available"}), 503
    
    try:
        col = dre.db[config.MONGO_COLLECTION_RULE_SUGGESTIONS]
        s = col.find_one({'_id': ObjectId(suggestion_id)})
        if not s:
            return jsonify({"error": "not found"}), 404
        
        rule_text = s.get('rule_text', '')
        rule_id = s.get('rule_id', 'X')
        
        return Response(
            rule_text,
            mimetype='text/plain',
            headers={
                'Content-Disposition': f'attachment; filename=iwaf-rule-{rule_id}.conf'
            }
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@dre_bp.route('/stats', methods=['GET'])
def dre_stats():
    """Aggregate DRE statistics."""
    dre = get_dre()
    if dre is None or dre.db is None:
        return jsonify({"error": "DRE not available"}), 503
    
    try:
        col = dre.db[config.MONGO_COLLECTION_RULE_SUGGESTIONS]
        total = col.count_documents({})
        
        if total == 0:
            return jsonify({
                "total_suggestions": 0,
                "by_status": {},
                "by_attack_type": {},
                "by_confidence": {},
            }), 200
        
        by_status = {}
        by_attack = {}
        by_confidence = {}
        
        for r in col.aggregate([{"$group": {"_id": "$status", "count": {"$sum": 1}}}]):
            by_status[r['_id']] = r['count']
        
        for r in col.aggregate([{"$group": {"_id": "$attack_type", "count": {"$sum": 1}}}]):
            by_attack[r['_id']] = r['count']
        
        for r in col.aggregate([{"$group": {"_id": "$confidence", "count": {"$sum": 1}}}]):
            by_confidence[r['_id']] = r['count']
        
        return jsonify({
            "total_suggestions": total,
            "by_status": by_status,
            "by_attack_type": by_attack,
            "by_confidence": by_confidence,
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
