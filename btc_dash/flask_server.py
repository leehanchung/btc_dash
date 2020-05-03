import flask

from btc_dash import config
from btc_dash.routes.sitemap_route import sitemap
from btc_dash.routes.robots_route import robots

# from btc_dash.routes import about


###############################################################################
#
#    Initialize Flask server
#
###############################################################################
server = flask.Flask(__name__, static_folder="assets",)
server.config["TESTING"] = config.TESTING

server.register_blueprint(sitemap)
server.register_blueprint(robots)
# server.register_blueprint(about)
