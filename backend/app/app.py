import flask
from flask.app import Flask
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import find_modules, import_string

from app.config import BaseConfig


def register_blueprints(*, app: Flask) -> None:
    """Register all blueprints located in /routes modules to the flask app

    Args:
        app (Flask): flask server object to hook blueprints to.
    """
    for name in find_modules("app.routes"):        
        name = name.split(".")
        name = ".".join(name + ['blueprint'])
        try:
            module = import_string(name)
        except:
            continue

        if isinstance(module, flask.blueprints.Blueprint):
            app.logger.info(f"Registering {name}...")
            app.register_blueprint(module)


def create_app(*, config: BaseConfig) -> Flask:
    """Creates flask server from config file and register all associated
    blueprints.

    Args:
        config (BaseConfig): config object that specifies app config

    Returns:
        Flask: Flask app object
    """
    app = flask.Flask(__name__, static_folder="assets",)
    app.config.from_object(config())
    
    # Setting up SQLAlchemy dB
    app.logger.info(f'Initializing db...')
    db = SQLAlchemy(app)
    from app.model import db
    db.init_app(app)
    app.logger.info(f'Initializiing db complete!')

    app.logger.info(f'Initializing blueprints...')
    register_blueprints(app=app)
    app.logger.info(f'Initializing blueprints complete!')

    return app
