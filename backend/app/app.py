import flask
from flask.app import Flask
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
        module = import_string(name)
        app.register_blueprint(module)

        if hasattr(module, 'blueprint'):
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
    app.config.from_object(config())#["TESTING"] = config.TESTING
    
    # Setting up SQLAlchemy dB
    # db = SQLAlchemy(app)
    app.logger.info(f'Setting up db...')
    from app.model import db
    db.init_app(app)

    app.logger.info(f'Setting up blueprints...')
    register_blueprints(app=app)
    app.logger.info(f'Finished setting up app...')

    return app
