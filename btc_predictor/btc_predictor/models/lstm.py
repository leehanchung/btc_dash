import json
import logging
from typing import Any, Dict, Tuple, Union

import numpy as np
import tensorflow as tf

from btc_predictor.datasets import BitfinexCandlesAPI, DataReader, util
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
        out = self.dense(x)
        return out


class LSTMBTCPredictor(BasePredictor):
    """Predictor Wrapper that predicts, trains, save and load LSTM models"""

    def __init__(self, *, model_args: Dict = None, train_args: Dict = None):
        self.model = None
        self.model_args = model_args

        for arg, value in train_args.items():
            setattr(self, arg, value)

    def train(self, *, data: Union[DataReader, BitfinexCandlesAPI]) -> None:
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
        self.resolution = data.period
        self.name = f"lstm_{self.start_time}_{self.end_time}_{self.resolution}"

        df["log_ret"] = np.log(df["close"]) - np.log(df["close"].shift(1))
        df.dropna(inplace=True)

        # preprocess
        time_series_data = np.diff(df["log_ret"].to_numpy()).astype("float32")
        train = time_series_data[: self.TRAIN_SIZE]
        val = time_series_data[
            self.TRAIN_SIZE : self.VAL_SIZE + self.TRAIN_SIZE
        ]

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

    def eval(
        self, *, data: Union[DataReader, BitfinexCandlesAPI]
    ) -> Tuple[float, float, float]:
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

        # preprocess
        df["log_ret"] = np.log(df["close"]) - np.log(df["close"].shift(1))
        df.dropna(inplace=True)

        time_series_data = np.diff(df["log_ret"].to_numpy()).astype("float32")
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
        except (OSError, FileNotFoundError, TypeError, Exception):
            raise ModelSavingError("Problem saving model wrapper attributes.")

    def load(self, *, model_filename: str) -> None:
        """Function that saves a serialized model.

        Args:
            model_filename[str]

        Returns:
            None
        """
        if not model_filename or hasattr(self, "name"):
            raise ModelLoadingError("Model name not specified")

        try:
            with open(f"{model_filename}/model_attr.json", "r") as f:
                attrs = json.load(f)
                for attr, value in attrs.items():
                    setattr(self, attr, value)
        except (OSError, FileNotFoundError, TypeError, Exception):
            raise ModelLoadingError(
                "Problem loading model wrapper attributes."
            )

        try:
            self.model = tf.keras.models.load_model(model_filename)
        except (ValueError, AttributeError):
            raise ModelLoadingError("Error loading Tensorflow Keras model.")

    # def _preproc(self, *) -> None:
    #     raise NotImplementedError
