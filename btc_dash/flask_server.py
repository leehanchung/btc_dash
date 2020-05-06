import flask
from flask.app import Flask
from werkzeug.utils import find_modules, import_string

from btc_dash.config import BaseConfig


def register_blueprints(*, server: Flask) -> None:
    """Register all blueprints located in /routes modules to the flask app

    Args:
        server: flask server object to hook blueprints to.

    Returns:
        None

    """
    for name in find_modules("btc_dash.routes"):
        # it doesnt like to import robots from btc_dash.routes.robots
        # so hack around the import string.
        name = name.split(".")
        name = ".".join(name + [name[-1]])
        module = import_string(name)
        server.register_blueprint(module)
        # if hasattr(module, 'blueprint'):
        #     server.register_blueprint(module)


def create_flask_server(*, config: BaseConfig) -> Flask:
    """Creates flask server from config file and register all associated
    blueprints.

    Args:
        server: flask server object for initializing plotly dash

    Returns:
        Dash app object

    """
    server = flask.Flask(__name__, static_folder="assets",)
    server.config["TESTING"] = config.TESTING

    register_blueprints(server=server)

    return server
