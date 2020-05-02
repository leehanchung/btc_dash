import os
import pathlib
from dotenv import load_dotenv


load_dotenv()
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
print(PACKAGE_ROOT)


class DataReadingError(Exception):
    """DataReadingError exception used for sanity checking.
    """

    def __init__(self, *args):
        super(DataReadingError, self).__init__(*args)
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"DataReadingError {self.message}"

        return "DataReadingError"


class BaseConfig:
    """Base config"""

    DEBUG = True
    TESTING = True


config = BaseConfig
