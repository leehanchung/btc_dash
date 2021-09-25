import json
import logging
import sys
from typing import Dict

import btc_predictor
from btc_predictor.models.lstm import load_wo_hydra

# import numpy as np
# import pandas as pd


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
# Loading BTC Predictor based on experiment config
#
##############################################################################
BTC_PREDICTOR_CONFIG = "experiments/btc_predictor_sample.yaml"
BTC_PREDICTOR = load_wo_hydra(config_file=BTC_PREDICTOR_CONFIG)
__version__ = f"BTC_PREDICTOR.{btc_predictor.__version__}"


def generate_failure_response(msg: str) -> Dict:
    return {
        "status": "failed",
        "data": [],
        "error": f"{msg}",
        "version": f"{__version__}",
        "headers": {"Content-Type": "application/json"},
    }


def generate_output_dict(*, body: float) -> Dict:
    return {
        "statusCode": 200,
        "body": json.dumps(body),
    }


def lambda_handler(event: Dict, context) -> Dict:

    _logger.info(f"Loaded model name: {BTC_PREDICTOR.name}")

    if not event:
        response = generate_failure_response("Empty event")
        return generate_output_dict(body=response)

    # Loads event body
    try:
        if "body" in event and event.get("body"):
            data = json.loads(event["body"])
    except Exception:
        response = generate_failure_response("Failed loading event")
        return generate_output_dict(body=response)

    # Model inference and error handling
    preds = BTC_PREDICTOR.predict(X=data)

    response = {
        "status": "success",
        "data": str(preds),
        "error": "",
        "version": f"{__version__}",
    }

    return generate_output_dict(body=response)
