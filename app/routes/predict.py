"""
LIFELINE AI — Prediction Routes
Handles form submission, validation, ML inference, and result rendering.
"""

import hashlib
import uuid
import logging
from flask import (
    Blueprint, render_template, request, session,
    redirect, url_for, jsonify, current_app, abort
)
from app import db, limiter
from app.models import Prediction, AnalyticsEvent
from app.services.prediction_service import predict, validate_inputs

predict_bp = Blueprint("predict", __name__)
logger = logging.getLogger(__name__)


def _get_or_create_session_id() -> str:
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        session.permanent = True
    return session["session_id"]


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:32]


def _parse_inputs(form) -> dict:
    """Parse and type-coerce form data."""
    return {
        "age":             int(form.get("age", 35)),
        "sex":             form.get("sex", "female"),
        "country":         form.get("country", "other"),
        "education":       form.get("education", "some_college"),
        "ses":             form.get("ses", "middle"),
        "smoking":         form.get("smoking", "never"),
        "alcohol":         form.get("alcohol", "light"),
        "exercise":        int(form.get("exercise", 3)),
        "sleep":           float(form.get("sleep", 7.0)),
        "stress_level":    int(form.get("stress_level", 5)),
        "social_score":    int(form.get("social_score", 7)),
        "fv_servings":     int(form.get("fv_servings", 4)),
        "processed_food":  form.get("processed_food", "sometimes"),
        "bmi":             float(form.get("bmi", 23.0)),
        "resting_hr":      int(form.get("resting_hr", 68)),
        "blood_pressure":  int(form.get("blood_pressure", 120)),
        "conditions":      form.get("conditions", "none"),
        "family_longevity":form.get("family_longevity", "80_90"),
        "mental_health":   form.get("mental_health", "good"),
        "environment":     form.get("environment", "suburban"),
        "aqi":             int(form.get("aqi", 45)),
    }


@predict_bp.route("/", methods=["POST"])
@limiter.limit("30 per hour")
def run_prediction():
    """Main prediction endpoint — accepts form POST, returns results page."""
    session_id = _get_or_create_session_id()

    try:
        inputs = _parse_inputs(request.form)
    except (ValueError, TypeError) as e:
        logger.warning(f"Input parse error: {e}")
        return render_template("index.html", error="Invalid input values. Please check your entries."), 400

    valid, errors = validate_inputs(inputs)
    if not valid:
        logger.warning(f"Validation errors: {errors}")
        return render_template("index.html", error="; ".join(errors), prev_inputs=inputs), 400

    # Run prediction
    try:
        result = predict(inputs)
    except Exception as e:
        logger.error(f"Prediction engine error: {e}", exc_info=True)
        abort(500)

    # Persist to DB
    try:
        prediction_row = Prediction(
            session_id=session_id,
            age=inputs["age"],
            sex=inputs["sex"],
            bmi=inputs["bmi"],
            smoking=inputs["smoking"],
            exercise=inputs["exercise"],
            sleep=inputs["sleep"],
            predicted_lifespan=result.predicted_lifespan,
            years_remaining=result.years_remaining,
            biological_age=result.biological_age,
            longevity_score=result.longevity_score,
            peer_percentile=result.peer_percentile,
            confidence_interval_low=result.confidence_interval_low,
            confidence_interval_high=result.confidence_interval_high,
            ip_hash=_hash(request.remote_addr or ""),
            user_agent_hash=_hash(request.user_agent.string or ""),
        )
        prediction_row.set_inputs(inputs)
        prediction_row.set_shap(result.feature_contributions)
        db.session.add(prediction_row)

        # Analytics event
        event = AnalyticsEvent(
            session_id=session_id,
            event_type="prediction_completed",
            page="/predict/",
        )
        event.set_data({
            "longevity_score": result.longevity_score,
            "age_group": f"{(inputs['age']//10)*10}s",
            "sex": inputs["sex"],
        })
        db.session.add(event)
        db.session.commit()

        prediction_id = prediction_row.id
    except Exception as e:
        logger.error(f"DB write error: {e}", exc_info=True)
        db.session.rollback()
        prediction_id = None

    return render_template(
        "results.html",
        result=result,
        inputs=inputs,
        prediction_id=prediction_id,
    )


@predict_bp.route("/history")
def history():
    """Shows prediction history for current session."""
    session_id = _get_or_create_session_id()
    predictions = (
        Prediction.query
        .filter_by(session_id=session_id)
        .order_by(Prediction.created_at.desc())
        .limit(20)
        .all()
    )
    return render_template("history.html", predictions=predictions)
