import os
import pathlib
import random

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import btc_predictor

RANDOM_SEED = 78
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
pd.options.display.max_rows = None
pd.options.display.max_columns = 500
load_dotenv()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Config:
    PACKAGE_ROOT = pathlib.Path(btc_predictor.__file__).resolve().parent
    DATASET_DIR = PACKAGE_ROOT / "datasets"

    AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
    BUCKET = os.environ.get("S3_BUCKET")

    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")

    RANDOM_STATE = RANDOM_SEED
    TRACKING_URI = os.environ.get("TRACKING_URI")


config = Config()
