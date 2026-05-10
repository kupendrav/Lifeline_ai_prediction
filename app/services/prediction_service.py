"""
LIFELINE AI — Prediction Service
Orchestrates: validation -> engine -> persistence -> analytics logging.
"""
from __future__ import annotations
import json, logging, uuid
from typing import Dict, Any

from app.services.validator import validate_and_clean
from app.ml.longevity_engine import predict

logger = logging.getLogger(__name__)


def run_prediction(raw_inputs: Dict[str, Any], session_id=None) -> Dict[str, Any]:
    sid = session_id or str(uuid.uuid4())
    cleaned, validation_errors = validate_and_clean(raw_inputs)
    if validation_errors:
        logger.warning("Validation warnings session %s: %s", sid, validation_errors)

    result = predict(cleaned)

    prediction_id = None
    try:
        prediction_id = _persist(cleaned, result, sid)
    except Exception as exc:
        logger.error("DB persist failed session %s: %s", sid, exc)

    return {
        "session_id":      sid,
        "prediction_id":   prediction_id,
        "predicted_age":   result.predicted_age,
        "remaining_years": result.remaining_years,
        "biological_age":  result.biological_age,
        "longevity_score": result.longevity_score,
        "percentile":      result.percentile,
        "confidence_low":  result.confidence_low,
        "confidence_high": result.confidence_high,
        "shap_factors":    result.shap_factors,
        "domain_scores":   result.domain_scores,
        "insights":        result.insights,
        "reasoning":       result.reasoning,
    }


def _persist(cleaned, result, sid):
    from app import db
    from app.models import Prediction
    rec = Prediction(
        session_id=sid,
        inputs_json=json.dumps(cleaned),
        predicted_age=result.predicted_age,
        remaining_yrs=result.remaining_years,
        biological_age=result.biological_age,
        longevity_score=result.longevity_score,
        percentile=result.percentile,
        shap_json=json.dumps(result.shap_factors),
        insights_json=json.dumps(result.insights),
        reasoning=result.reasoning,
    )
    from flask_sqlalchemy import SQLAlchemy
    db.session.add(rec)
    db.session.commit()
    logger.info("Prediction %s persisted (session %s)", rec.id, sid)
    return rec.id
