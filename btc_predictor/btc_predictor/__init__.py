import random

import numpy as np
import pandas as pd
import tensorflow as tf

from btc_predictor.config import config

VERSION_PATH = config.PACKAGE_ROOT / "VERSION"

with open(VERSION_PATH, "r") as version_file:
    __version__ = version_file.read().strip()

# Set random seed
random.seed(config.RANDOM_STATE)
np.random.seed(config.RANDOM_STATE)
tf.random.set_seed(config.RANDOM_STATE)

# Set Pandas display options
pd.options.display.max_rows = None
pd.options.display.max_columns = 500
