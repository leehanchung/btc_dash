import flask
from flask.app import Flask

from btc_dash.config import BaseConfig
from btc_dash.routes.sitemap_route import sitemap
from btc_dash.routes.robots_route import robots
# from btc_dash.routes import sitemap
# from btc_dash.routes import robots
# from btc_dash.routes import about


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

    server.register_blueprint(sitemap)
    server.register_blueprint(robots)
    # server.register_blueprint(about)

    return server
