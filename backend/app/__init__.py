###############################################################################
# Run the app by executing entrypoint.py. Alternatively, set environment
# variable $Env:FASTAPI_ENV="development" and then execute
# 'uvicorn app:app --host=0.0.0.0 --port=5000 --reload' from command line.
###############################################################################
from loguru import logger
from app.config import PACKAGE_ROOT, get_config
from app.app import create_app


with open(PACKAGE_ROOT / "VERSION", "rb") as version_file:
    __version__ = version_file.read().strip()

config = get_config()
app = create_app(config=config)
logger.info(f"Using config: {config}")
logger.info(f"Package root directory is: {PACKAGE_ROOT}")
