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
        # it doesnt like to import robots from app.routes.robots
        # so hack around the import string.
        # print(f'[DEBUG] register_blueprints name: {name}')
        name = name.split(".")
        # print(f'[DEBUG] register_blueprints name: {name}')
        # name = ".".join(name + [name[-1]]) #['blueprint'])#
        name = ".".join(name + ['blueprint'])
        # print(f'[DEBUG] register_blueprints name: {name}')
        module = import_string(name)
        # print(f'[DEBUG] register_blueprints name: {name}, module: {module}')
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
    
    print(f'[DEBUG] setting up db...')
    # db = SQLAlchemy(app)
    from app.model import db
    db.init_app(app)
    print(f'[DEBUG] setting up blueprints...')
    register_blueprints(app=app)
    print(f'[DEBUG] finished setting up app...')

    return app