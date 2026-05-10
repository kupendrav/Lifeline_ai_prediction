"""LIFELINE AI — Analytics dashboard blueprint."""
from flask import Blueprint, render_template, jsonify
from app.models import Prediction, AnalyticsEvent
from app import db
import sqlalchemy as sa

analytics_bp = Blueprint("analytics", __name__)


@analytics_bp.route("/")
def dashboard():
    total      = db.session.execute(sa.select(sa.func.count(Prediction.id))).scalar()
    avg_pred   = db.session.execute(sa.select(sa.func.avg(Prediction.predicted_age))).scalar()
    avg_score  = db.session.execute(sa.select(sa.func.avg(Prediction.longevity_score))).scalar()
    return render_template("analytics.html",
        total=total or 0,
        avg_predicted=round(avg_pred or 0, 1),
        avg_score=round(avg_score or 0, 1))


@analytics_bp.route("/api/summary")
def api_summary():
    total    = db.session.execute(sa.select(sa.func.count(Prediction.id))).scalar()
    avg_pred = db.session.execute(sa.select(sa.func.avg(Prediction.predicted_age))).scalar()
    return jsonify({"total_predictions": total, "avg_predicted_age": round(avg_pred or 0, 1)})
