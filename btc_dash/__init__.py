###############################################################################
# Run the app by executing run_dash.py. Alternatively, set FLASK_APP
# environment variable in Windows by $Env:FLASK_APP="btc_dash:server" and then
# execute 'flask run' from command line.
###############################################################################
from btc_dash.config import PACKAGE_ROOT, config
from btc_dash.flask_server import server
from btc_dash.dash_app import app


with open(PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()
