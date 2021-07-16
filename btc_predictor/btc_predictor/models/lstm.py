import json
import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd

from btc_predictor.datasets import BaseDataset, util
from btc_predictor.models import (
    BasePredictor,
    ModelLoadingError,
    ModelSavingError,
    ModelTrainingError,
)
from btc_predictor.utils import calculate_metrics

_logger = logging.getLogger(__name__)


class LSTMModel(tf.keras.Model):
    """LSTM Model for univariate time series prediction"""

    def __init__(
        self, *, input_shape: Tuple[int, int], dropout: float, num_forward: int
    ):
        super(LSTMModel, self).__init__()
        _logger.info(
            f"\ninput_shape {input_shape}\ndropout: {dropout}\n"
            f"num_forward: {num_forward}"
        )
        self.lstm_input = tf.keras.layers.LSTM(
            128, input_shape=input_shape, return_sequences=True
        )
        self.lstm_hidden = tf.keras.layers.LSTM(64)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense = tf.keras.layers.Dense(num_forward)

    def call(self, x):
        # x = self.inputs(inputs)
        x = self.lstm_input(x)
        x = self.dropout(x)
        x = self.lstm_hidden(x)
        x = self.dropout(x)
        return self.dense(x)


class LSTMBTCPredictor(BasePredictor):
    """Predictor Wrapper that predicts, trains, save and load LSTM models"""

    def __init__(self, *, model_args: Dict = None, train_args: Dict = None):
        self.model = None
        self.model_args = model_args

        for arg, value in train_args.items():
            setattr(self, arg, value)

    def train(self, *, data: BaseDataset) -> None:
        """Function that accept input training data and train the model

        Args:
            data: Features required by the model to generate a
            prediction. Numpy array of shape (1, n) where n is the dimension
            of the feature vector.

        Returns:
            None
        """

        if self.model:
            raise ModelTrainingError("Model already trained.")

        self.model = LSTMModel(**self.model_args)

        # load data
        df = data.pd

        self.start_time = data.start_time
        self.end_time = data.end_time
        self.resolution = data.resolution
        self.name = f"lstm_{self.start_time}_{self.end_time}_{self.resolution}"

        time_series_data = self._preproc(df=data.pd)

        train = time_series_data[: self.TRAIN_SIZE]
        val = time_series_data[
            self.TRAIN_SIZE : self.VAL_SIZE + self.TRAIN_SIZE
        ]
        _logger.info(f"train first 20: {train[:20]}")
        _logger.info(f"val first 20: {val[:20]}")
        train_tfds = util.create_tfds_from_np(
            data=train,
            window_size=self.WINDOW_SIZE,
            batch_size=self.BATCH_SIZE,
        )
        val_tfds = util.create_tfds_from_np(
            data=val,
            window_size=self.WINDOW_SIZE,
            batch_size=self.BATCH_SIZE,
        )
        _logger.info(f"Total {self.resolution} data: {df.shape[0]}")

        self.model.compile(
            optimizer="adam",
            loss="mse",
        )

        train_history = self.model.fit(
            train_tfds,
            epochs=self.EPOCHS,
            steps_per_epoch=self.EVALUATION_INTERVAL,
            validation_data=val_tfds,
            validation_steps=self.VALIDATION_STEPS,
        )

        self.history = train_history

        return None

    def eval(self, *, data: BaseDataset) -> Tuple[float, float, float]:
        """Function that accept input training data and train the model

        Args:
            data (DataReader): Features required by the model to generate a
            prediction. Numpy array of shape (1, n) where n is the dimension
            of the feature vector.

        Returns:
            Tuple[float, float, float]: eval_scores of RMSE, directional
            accuracy, and mean directional accuracy
        """
        # load data
        df = data.pd

        time_series_data = self._preproc(df=df)

        test = time_series_data[self.VAL_SIZE + self.TRAIN_SIZE :]
        test_tfds = util.create_tfds_from_np(
            data=test,
            window_size=self.WINDOW_SIZE,
            batch_size=1,
        )

        # evaluate
        test_y_true = np.array([])
        test_y_pred = np.array([])
        for x, y in test_tfds.take(self.WALK_FORWARD):
            test_y_true = np.append(test_y_true, y[0].numpy())
            test_y_pred = np.append(test_y_pred, self.model.predict(x)[0])

        skip_num_index = self.VAL_SIZE + self.TRAIN_SIZE
        eval_df = df.iloc[skip_num_index:, :][:46]
        true_close = eval_df["close"].to_numpy()
        true_log_ret = eval_df["log_ret"].to_numpy()
        true_log_ret_diff = eval_df["log_ret_diff"].to_numpy()
        # Logging code to print out how the fuck things line up.
        _logger.info(f"eval_df:\n{eval_df}")
        _logger.info(f"true_close:\n{true_close}")
        _logger.info(f"log_ret:\n{true_log_ret}")
        _logger.info(f"log_ret_diff:\n{true_log_ret_diff}")
        _logger.info(f"ts_true:\n{test[:45]}")
        _logger.info(f"ts_true len:\n{len(test)}")
        _logger.info(f"y_true:\n{test_y_true}")
        _logger.info(f"y_pred:\n{test_y_pred}")

        rmse, dir_acc, mda = calculate_metrics(
            y_true=test_y_true, y_pred=test_y_pred
        )

        return rmse, dir_acc, mda

    def predict(self, *, input_features: Any) -> np.ndarray:
        """Function that accept input data for the model to generate a prediction

        Args:
            input_features: Features required by the model to generate a
            prediction. Numpy array of shape (1, n) where n is the dimension
            of the feature vector.

        Returns:
            prediction: Prediction of the model. Numpy array of shape (1,).
        """
        raise NotImplementedError

    def save(self) -> None:
        """Function that saves a serialized model.

        Args:
            None

        Returns:
            None
        """
        if not self.name:
            raise ModelSavingError("Model not trained; aborting save.")

        try:
            self.model.save(f"saved_model/{self.name}", save_format="tf")
        except (ValueError, AttributeError):
            raise ModelSavingError("Problem saving model weights.")

        try:
            with open(f"saved_model/{self.name}/model_attr.json", "w") as f:
                attrs = self.__dict__
                attrs.pop("model", None)
                attrs.pop("history", None)
                json.dump(self.__dict__, f)
        except (OSError, TypeError, Exception):
            raise ModelSavingError("Problem saving model wrapper attributes.")

    def load(self, *, model_name: str) -> None:
        """Function that saves a serialized model.

        Args:
            model_name[str]

        Returns:
            None
        """
        if not model_name or hasattr(self, "name"):
            raise ModelLoadingError("Model name not specified")

        try:
            if HydraConfig.initialized():
                model_name = get_original_cwd() + f"/{model_name}"

            with open(f"{model_name}/model_attr.json", "r") as f:
                attrs = json.load(f)
                for attr, value in attrs.items():
                    setattr(self, attr, value)
        except (OSError, TypeError, Exception):
            raise ModelLoadingError(
                "Problem loading model wrapper attributes."
            )

        try:
            self.model = tf.keras.models.load_model(model_name)
        except (ValueError, AttributeError):
            raise ModelLoadingError("Error loading Tensorflow Keras model.")

    def _preproc(self, *, df: pd.DataFrame) -> np.ndarray:
        """[summary]

        Args:
            data (pd.DataFrame): input dataframe containing 'close' field

        Returns:
            np.ndarray: log return numpy array.
        """
        df["log_ret"] = np.log(df["close"]) - np.log(df["close"].shift(1))
        df["log_ret_diff"] = df["log_ret"].diff()
        df.dropna(inplace=True)
        return df["log_ret_diff"].to_numpy().astype("float16")

    def _postproc(self, *, data: pd.DataFrame) -> float:
        raise NotImplementedError
