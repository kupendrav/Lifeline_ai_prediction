"""LIFELINE AI — Main blueprint (page routes)."""
from flask import Blueprint, render_template, session, redirect, url_for
import uuid

main_bp = Blueprint("main", __name__)

@main_bp.before_request
def ensure_session():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())

@main_bp.route("/")
def index():
    return render_template("index.html")

@main_bp.route("/about")
def about():
    return render_template("about.html")

@main_bp.route("/results/<string:prediction_id>")
def results(prediction_id):
    from app.models import Prediction
    import json
    pred = Prediction.query.get(prediction_id)
    if pred is None:
        pred = (
            Prediction.query
            .filter_by(session_id=prediction_id)
            .order_by(Prediction.created_at.desc())
            .first_or_404()
        )
    data = pred.to_dict()
    data["shap_factors"] = json.loads(pred.shap_json or "{}")
    data["insights"]     = json.loads(pred.insights_json or "[]")
    data["reasoning"]    = pred.reasoning
    return render_template("results.html", result=data)

@main_bp.route("/history")
def history():
    sid = session.get("sid")
    from app.models import Prediction
    preds = Prediction.query.filter_by(session_id=sid).order_by(
        Prediction.created_at.desc()).limit(10).all()
    return render_template("history.html", predictions=[p.to_dict() for p in preds])

@main_bp.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, msg="Page not found"), 404

@main_bp.errorhandler(500)
def server_error(e):
    return render_template("error.html", code=500, msg="Internal server error"), 500
