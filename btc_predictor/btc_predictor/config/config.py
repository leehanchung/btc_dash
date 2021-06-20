import pathlib
import random

import numpy as np
import pandas as pd

import btc_predictor

random.seed(78)
np.random.seed(78)
pd.options.display.max_rows = None
pd.options.display.max_columns = 500


class Config:
    PACKAGE_ROOT = pathlib.Path(btc_predictor.__file__).resolve().parent
    DATASET_DIR = PACKAGE_ROOT / "datasets"

    AWS_ACCESS_KEY_ID = ""
    AWS_SECRET_ACCESS_KEY = ""
    BUCKET = ""

    RANDOM_STATE = 78
    TRACKING_URI = ""


config = Config()
