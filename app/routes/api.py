"""LIFELINE AI — REST API blueprint."""
import logging
from flask import Blueprint, request, jsonify, session
from app.services.prediction_service import run_prediction
from app import limiter

api_bp  = Blueprint("api", __name__)
logger  = logging.getLogger(__name__)


@api_bp.route("/predict", methods=["POST"])
@limiter.limit("10 per minute")
def predict():
    data = request.get_json(silent=True) or request.form.to_dict()
    if not data:
        return jsonify({"error": "No input data provided"}), 400
    try:
        result = run_prediction(data, session_id=session.get("sid"))
        return jsonify({"status": "ok", "data": result}), 200
    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        return jsonify({"error": "Prediction failed", "detail": str(exc)}), 500


@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "lifeline-ai"}), 200


@api_bp.route("/features", methods=["GET"])
def features():
    """Returns feature metadata for API consumers."""
    return jsonify({
        "numeric": {
            "age": {"min": 18, "max": 100, "unit": "years"},
            "exercise": {"min": 0, "max": 7, "unit": "days/week"},
            "sleep": {"min": 3, "max": 12, "unit": "hours"},
            "stress": {"min": 1, "max": 10, "unit": "scale"},
            "social": {"min": 1, "max": 10, "unit": "scale"},
            "fv": {"min": 0, "max": 14, "unit": "servings/day"},
            "bmi": {"min": 12, "max": 60, "unit": "kg/m2"},
            "hr": {"min": 30, "max": 130, "unit": "bpm"},
            "bp": {"min": 70, "max": 200, "unit": "mmHg systolic"},
            "aqi": {"min": 0, "max": 500, "unit": "AQI"},
        },
        "categorical": {
            "sex": ["female", "male", "other"],
            "smoking": ["never", "former", "light", "moderate", "heavy"],
            "alcohol": ["never", "light", "moderate", "heavy"],
            "conditions": ["none", "hypertension", "diabetes", "heart_disease", "multiple"],
            "family": ["below_70", "70_80", "80_90", "above_90"],
            "mental": ["excellent", "good", "moderate", "poor"],
        }
    })
