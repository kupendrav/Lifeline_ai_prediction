"""
LIFELINE AI — Flask Application Factory
Production-grade Human Longevity Intelligence Platform
"""
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf import CSRFProtect

db = SQLAlchemy()
limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])
csrf = CSRFProtect()


def create_app(config_name: str = "development") -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    from config.settings import config_map
    app.config.from_object(config_map[config_name])

    db.init_app(app)
    limiter.init_app(app)
    csrf.init_app(app)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from app.routes.main import main_bp
    from app.routes.api import api_bp
    from app.routes.analytics import analytics_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api/v1")
    app.register_blueprint(analytics_bp, url_prefix="/analytics")

    with app.app_context():
        db.create_all()

    return app
