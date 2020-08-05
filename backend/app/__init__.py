###############################################################################
# Run the app by executing run_app.py. Alternatively, set FLASK_APP
# environment variable in Windows by $Env:FLASK_APP="app:app" and then
# execute 'flask run' from command line.
###############################################################################
from app.config import PACKAGE_ROOT, config
from app.app import create_app


with open(PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()

app = create_app(config=config)
