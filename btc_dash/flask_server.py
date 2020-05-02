import flask

from btc_dash import config
# from btc_dash.routes import sitemap
# from btc_dash.routes import robots

print("flask_server")
###############################################################################
#
#    Initialize Flask server
#
###############################################################################
server = flask.Flask(
    __name__,
    static_folder='btc_dash/assets',
)
server.config['TESTING'] = config.TESTING

# server.register_blueprint(sitemap)
print("flask_server end")
