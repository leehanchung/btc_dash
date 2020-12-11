###############################################################################
# Run the app by executing run_dash.py. Alternatively, set FLASK_APP
# environment variable in Windows by $Env:FLASK_APP="btc_dash:app" and then
# execute 'flask run' from command line.
###############################################################################
from btc_dash.config import PACKAGE_ROOT, config
from btc_dash.flask_server import create_flask_server
from btc_dash.dash_app import create_dash_app


with open(PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()

server = create_flask_server(config=config)
app = create_dash_app(flask_server=server)
app.logger.info(f'Using config: {config}')
app.logger.info(f'Package root directory is: {PACKAGE_ROOT}')
