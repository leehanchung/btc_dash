# import json
import logging
import sys
from typing import Dict

# import numpy as np
# import pandas as pd

from btc_predictor.models.utils import load_wo_hydra


# Set Logging
FORMATTER = logging.Formatter(
    "%(asctime)s [%(levelname)-5.5s] [%(name)s] "
    "[%(funcName)-15.15s:%(lineno)d] - %(message)s"
)
_logger = logging.getLogger()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(FORMATTER)
_logger.setLevel(logging.INFO)
_logger.addHandler(console_handler)

##############################################################################
#
#
# Credit Offering page CTR model configs and model loading
#
#
##############################################################################
BTC_PREDICTOR_CONFIG = "experiments/btc_predictor_sample.yaml"
BTC_PREDICTOR = load_wo_hydra(config_file=BTC_PREDICTOR_CONFIG)


def lambda_handler(event: Dict, context) -> Dict:
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
