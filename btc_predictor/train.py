import logging
import sys

import numpy as np
import tensorflow as tf

from btc_predictor.config import config, logging_config
from btc_predictor.datasets import BitfinexCandlesAPIData
from btc_predictor.models import LSTMBTCPredictor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging_config.get_console_handler())
logger.propagate = True


model_params = {
    "input_shape": (29, 1),
    "dropout": 0.4,
    "num_forward": 1,
}

train_params = {
    "TRAIN_SIZE": 5000,
    "VAL_SIZE": 1000,
    "WINDOW_SIZE": 20,
    "BATCH_SIZE": 256,
    "EPOCHS": 20,
    "EVALUATION_INTERVAL": 64,
    "VALIDATION_STEPS": 64,
    "WALK_FORWARD": 30,
}


def train():
    # data_file = "btc_predictor/datasets/Bitstamp_BTCUSD_d.csv"
    # data = DataReader(data_file=data_file)

    candles = BitfinexCandlesAPIData()
    candles.load(start_time=1610000000000)

    btc_predictor = LSTMBTCPredictor(
        model_args=model_params,
        train_args=train_params
    )

    btc_predictor.train(data=candles)
    rmse, dir_acc, mean_dir_acc = btc_predictor.eval(data=candles)
    logger.info(f"RMSE {rmse}")
    logger.info(f"Directional accuracy: {dir_acc}")
    logger.info(f"Mean directional accuracy {mean_dir_acc}")
    
    logger.info(f"Saving model {btc_predictor.name}...")
    btc_predictor.save()

    logger.info("Loading model...")
    model = LSTMBTCPredictor(
        model_args=model_params,
        train_args=train_params
    )

    model.load(model_filename="saved_model/lstm_20210106_20210106_1m")
    logger.info(f"Loaded model name: {model.name}")
    rmse, dir_acc, mean_dir_acc = model.eval(data=candles)
    logger.info(f"RMSE {rmse}")
    logger.info(f"Directional accuracy: {dir_acc}")
    logger.info(f"Mean directional accuracy {mean_dir_acc}")


if __name__ == "__main__":
    train()
