from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import Config
from app.logging import setup_logging
from app.routes import api_router


def create_app(*, config: Config) -> FastAPI:
    """Creates FastAPI server from config file and register all associated
    routers.

    Args:
        config (BaseConfig): config object that specifies app config

    Returns:
        Flask: Flask app object
    """
    app = FastAPI(
        title=config.PROJECT_NAME,
        openapi_url=f"{config.API_PREFIX}/openapi.json",
        debug=config.DEBUG,
    )

    # Setup unified logging with loguru
    setup_logging(config=config)

    # Setup CORS
    if config.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                str(origin) for origin in config.BACKEND_CORS_ORIGINS
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Registering routes
    logger.info("Registering routes...")
    app.include_router(api_router, prefix=config.API_PREFIX)
    logger.info("Routes registration complete!")

    return app
