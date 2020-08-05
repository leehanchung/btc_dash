import flask
from flask.app import Flask
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import find_modules, import_string
from app.config import BaseConfig


def register_blueprints(*, app: Flask) -> None:
    """Register all blueprints located in /routes modules to the flask app

    Args:
        server: flask server object to hook blueprints to.

    Returns:
        None

    """
    for name in find_modules("app.routes"):
        # it doesnt like to import robots from app.routes.robots
        # so hack around the import string.
        name = name.split(".")
        name = ".".join(name + [name[-1]])
        module = import_string(name)
        app.register_blueprint(module)
        # if hasattr(module, 'blueprint'):
        #     server.register_blueprint(module)


def create_app(*, config: BaseConfig) -> Flask:
    """Creates flask server from config file and register all associated
    blueprints.

    Args:
        server: flask server object for initializing plotly dash

    Returns:
        Dash app object

    """
    app = flask.Flask(__name__, static_folder="assets",)
    app.config["TESTING"] = config.TESTING
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
    
    db = SQLAlchemy(app)
    register_blueprints(app=app)

    return app