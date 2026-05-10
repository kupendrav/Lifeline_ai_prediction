"""
SQLAlchemy ORM models — predictions, sessions, analytics events.
"""
import uuid
from datetime import datetime
from app import db


def new_uuid():
    return str(uuid.uuid4())


class Prediction(db.Model):
    __tablename__ = "predictions"

    id            = db.Column(db.String(36), primary_key=True, default=new_uuid)
    session_id    = db.Column(db.String(36), nullable=False, index=True)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Inputs snapshot (JSON string)
    inputs_json   = db.Column(db.Text, nullable=False)

    # Outputs
    predicted_age = db.Column(db.Float, nullable=False)
    remaining_yrs = db.Column(db.Float, nullable=False)
    biological_age= db.Column(db.Float, nullable=False)
    longevity_score= db.Column(db.Integer, nullable=False)
    percentile    = db.Column(db.Integer, nullable=False)
    shap_json     = db.Column(db.Text)          # SHAP values JSON
    insights_json = db.Column(db.Text)          # Insights array JSON
    reasoning     = db.Column(db.Text)

    def to_dict(self):
        import json
        from app.ml.longevity_engine import SIGMA
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "inputs": json.loads(self.inputs_json),
            "predicted_age": self.predicted_age,
            "remaining_years": self.remaining_yrs,
            "biological_age": self.biological_age,
            "longevity_score": self.longevity_score,
            "percentile": self.percentile,
            "confidence_low": round(self.predicted_age - SIGMA, 0),
            "confidence_high": round(self.predicted_age + SIGMA, 0),
        }


class AnalyticsEvent(db.Model):
    __tablename__ = "analytics_events"

    id         = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id = db.Column(db.String(36), nullable=False, index=True)
    event_name = db.Column(db.String(64), nullable=False)
    payload    = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class FeatureUsageLog(db.Model):
    """Aggregated feature value distributions for research analytics."""
    __tablename__ = "feature_usage"

    id         = db.Column(db.Integer, primary_key=True, autoincrement=True)
    feature    = db.Column(db.String(64), nullable=False)
    value_bucket = db.Column(db.String(64), nullable=False)
    count      = db.Column(db.Integer, default=1, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("feature", "value_bucket", name="uq_feature_bucket"),
    )
