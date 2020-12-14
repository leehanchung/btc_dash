###############################################################################
# Run the app by executing run_app.py. Alternatively, set FLASK_APP
# environment variable in Windows by $Env:FLASK_APP="backend/app" and then
# execute 'flask run' from command line.
###############################################################################
from app.config import PACKAGE_ROOT, get_config
from app.app import create_app

with open(PACKAGE_ROOT / "VERSION", "rb") as version_file:
    __version__ = version_file.read().strip()

config = get_config()
app = create_app(config=config)
app.logger.info(f"Using config: {config}")
app.logger.info(f"Package root directory is: {PACKAGE_ROOT}")
