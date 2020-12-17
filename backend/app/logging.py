import logging
import sys

from loguru import logger

from app.config import Config


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(*, config: Config) -> None:
    """Setting up logging to unify and simplify uvicorn and FastAPI logging
    with the same formatting.

    Args:
        config (Config): app config object

    Returns:
        Any: [description]
    """

    # intercept everything at the root logger
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(config.LOG_LEVEL)

    # remove every other logger's handlers
    # and propagate to root logger
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True

    # configure loguru
    logger.configure(
        handlers=[{"sink": sys.stdout, "serialize": config.JSON_LOGS}]
    )
