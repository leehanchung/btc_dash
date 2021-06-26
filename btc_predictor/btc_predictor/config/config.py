import pathlib
import random

from dotenv import load_dotenv
import numpy as np
import pandas as pd

import btc_predictor

random.seed(78)
np.random.seed(78)
pd.options.display.max_rows = None
pd.options.display.max_columns = 500
load_dotenv()


class Config:
    PACKAGE_ROOT = pathlib.Path(btc_predictor.__file__).resolve().parent
    DATASET_DIR = PACKAGE_ROOT / "datasets"

    AWS_ACCESS_KEY_ID = ""
    AWS_SECRET_ACCESS_KEY = ""
    BUCKET = ""

    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")

    RANDOM_STATE = 78
    TRACKING_URI = ""


config = Config()
