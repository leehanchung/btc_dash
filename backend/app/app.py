import flask
from flask.app import Flask
from sqlalchemy.orm import scoped_session
from werkzeug.utils import find_modules, import_string

from app.config import Config
from app.db.core import init_database


def register_blueprints(*, app: Flask) -> None:
    """Register all blueprints located in /routes modules to the flask app

    Args:
        app (Flask): flask server object to hook blueprints to.
    """
    for name in find_modules("app.routes"):
        name = name.split(".")
        name = ".".join(name + ["blueprint"])
        try:
            module = import_string(name)
        except:  # noqa: E722
            continue

        if isinstance(module, flask.blueprints.Blueprint):
            app.logger.info(f"Registering {name}...")
            app.register_blueprint(module)


def create_app(*, config: Config, db_session: scoped_session = None) -> Flask:
    """Creates flask server from config file and register all associated
    blueprints.

    Args:
        config (BaseConfig): config object that specifies app config

    Returns:
        Flask: Flask app object
    """
    app = flask.Flask(
        __name__,
        static_folder="assets",
    )
    app.config.from_object(config)

    # Setting up SQLAlchemy dB
    app.logger.info("Initializing db...")
    init_database(app, config=config, db_session=db_session)
    # db = SQLAlchemy(app)
    # from app.model import db
    # db.init_app(app)
    app.logger.info("Initializing db complete!")

    app.logger.info("Initializing blueprints...")
    register_blueprints(app=app)
    app.logger.info("Initializing blueprints complete!")

    return app
