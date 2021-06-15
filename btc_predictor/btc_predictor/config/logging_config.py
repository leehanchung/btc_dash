import logging
import sys


FORMATTER = logging.Formatter(
    "%(asctime)s [%(levelname)-5.5s] [%(funcName)-20.20s:%(lineno)d] - %(message)s"  # noqa: E501
)


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler
