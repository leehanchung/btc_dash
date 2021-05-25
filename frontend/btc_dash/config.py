import os
import pathlib
import pandas as pd
from dotenv import load_dotenv


load_dotenv()
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent


class DataReadingError(Exception):
    """DataReadingError exception used for sanity checking."""

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

    GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 60000)
    app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}
    df_pred = pd.DataFrame(columns=["pred_log_ret", "pred_Close"])


config = BaseConfig
