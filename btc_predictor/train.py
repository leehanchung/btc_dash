import logging

import hydra
import numpy as np
from omegaconf import DictConfig

# from btc_predictor.config import logging_config
from btc_predictor.datasets import BitfinexCandlesAPIData
from btc_predictor.models import LSTMBTCPredictor


_logger = logging.getLogger(__name__)


model_params = {
    "input_shape": (29, 1),
    "dropout": 0.4,
    "num_forward": 1,
}

train_params = {
    "TRAIN_SIZE": 5000,
    "VAL_SIZE": 1000,
    "WINDOW_SIZE": 16,
    "BATCH_SIZE": 256,
    "EPOCHS": 20,
    "EVALUATION_INTERVAL": 64,
    "VALIDATION_STEPS": 64,
    "WALK_FORWARD": 30,
}


@hydra.main(
    config_path="experiments",
    config_name="btc_predictor_sample.yaml"
)
def train(params: DictConfig) -> None:
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
    _logger.info(f"RMSE {rmse}")
    _logger.info(f"Directional accuracy: {dir_acc}")
    _logger.info(f"Mean directional accuracy {mean_dir_acc}")

    _logger.info(f"Saving model {btc_predictor.name}...")
    btc_predictor.save()

    _logger.info("Loading model...")
    model = LSTMBTCPredictor(
        model_args=model_params,
        train_args=train_params
    )

    model.load(model_name="lstm_20210106_20210106_1m", origin_pwd=False)
    _logger.info(f"Loaded model name: {model.name}")
    rmse, dir_acc, mean_dir_acc = model.eval(data=candles)
    _logger.info(f"RMSE {rmse}")
    _logger.info(f"Directional accuracy: {dir_acc}")
    _logger.info(f"Mean directional accuracy {mean_dir_acc}")

    X_test = np.array([35581, 35787, 35702, 35786, 35576, 35427,
        35356.49956215, 35499, 35417, 35461, 35500, 35535, 35620, 35466, 35422,
        35385.847202, 35389])

    pred = model.predict(X=X_test)
    _logger.info(f"pred: {pred}")


if __name__ == "__main__":
    train()
