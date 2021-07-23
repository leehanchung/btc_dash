import json
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
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

        skip_num_index = self.VAL_SIZE + self.TRAIN_SIZE
        eval_df = df.iloc[skip_num_index:, :][:46]
        test = time_series_data[skip_num_index:]
        _logger.info(test)
        test_tfds = util.create_tfds_from_np(
            data=test,
            window_size=self.WINDOW_SIZE,
            batch_size=1,
        )

        # evaluate
        test_y_true = np.array([])
        test_y_pred = np.array([])
        test_y_close_true = np.array([])
        test_y_close_pred = np.array([])
        for i, (X, y) in enumerate(test_tfds.take(self.WALK_FORWARD)):
            pred = self.model.predict(X)[0]
            test_y_true = np.append(test_y_true, y[0].numpy())
            test_y_pred = np.append(test_y_pred, pred)

            log_ret = eval_df["log_ret"].iloc[self.WINDOW_SIZE + i - 2]
            close = eval_df["close"].iloc[self.WINDOW_SIZE + i - 2]
            next_close = eval_df["close"].iloc[self.WINDOW_SIZE + i - 1]

            pred_clos = self._postproc(pred=pred, log_ret=log_ret, close=close)
            test_y_close_true = np.append(test_y_close_true, next_close)
            test_y_close_pred = np.append(test_y_close_pred, pred_clos[0])

        _logger.debug(f"y_true:\n{test_y_true}")
        _logger.debug(f"y_pred:\n{test_y_pred}")

        _logger.debug(f"y_close_true:\n{test_y_close_true}")
        _logger.debug(f"y_close_pred:\n{test_y_close_pred}")

        rmse, dir_acc, mda = calculate_metrics(
            y_true=test_y_true, y_pred=test_y_pred
        )

        return rmse, dir_acc, mda

    def predict(self, *, X: np.ndarray) -> float:
        """Function that accept input data for the model to generate a prediction

        Args:
            X (np.ndarray) (15,): Features required by the model to generate a
            prediction. Numpy vector of shape (15,), where 15 is the shape of
            the specified WINDOW_SIZE - 1.

        Returns:
            float: A single float number .
        """
        X_log_ret = np.diff(np.log(X))
        X_lr_diff = np.diff(X_log_ret)

        # Model requires shape of [None, 15, 1], thus gotta change shape.
        X_lr_diff = np.expand_dims(X_lr_diff, axis=(0, 2))
        X_lr_diff_tensor = tf.convert_to_tensor(X_lr_diff)

        _logger.info(f"X_log_ret_diff_tensor type: {X_lr_diff_tensor.shape}")

        pred = self.model.predict(X_lr_diff_tensor)[0]
        return (X[-1] * np.exp(pred + X_log_ret[-1]))[0]

    def save(self, *, origin_pwd: bool = False) -> None:
        """Function that saves a serialized model.

        Args:
            origin_pwd (bool, optional): Load file from original package root
                if True. Else load from Hydra config current working dir.
                Defaults to False.

        Returns:
            None
        """
        if not self.name:
            raise ModelSavingError("Model not trained; aborting save.")

        model_dir = f"{self.name}"
        if origin_pwd:
            model_dir = (
                get_original_cwd() + f"/btc_predictor/saved_models/{model_dir}"
            )

        self.model.save(model_dir, save_format="tf")

        with open(f"{model_dir}/model_attr.json", "w") as f:
            attrs = self.__dict__
            attrs.pop("model", None)
            attrs.pop("history", None)
            json.dump(self.__dict__, f)

    def load(self, *, model_name: str, origin_pwd: bool = False) -> None:
        """Function that saves a serialized model.

        Args:
            model_name (str): specify model name to load. or model directory
                if saved using tf format.
            origin_pwd (bool, optional): Load file from original package root
                if True. Else load from Hydra config current working dir.
                Defaults to False.

        Returns:
            None
        """
        if self.model:
            raise ModelLoadingError("Model already exists!")

        if not model_name or not hasattr(self, "name"):
            raise ModelLoadingError("Model name not specified")

        model_dir = f"{model_name}"
        if origin_pwd:
            model_dir = (
                get_original_cwd() + f"/btc_predictor/saved_models/{model_dir}"
            )
        _logger.info(f"load model dir {model_dir}")
        with open(f"{model_dir}/model_attr.json", "r") as f:
            attrs = json.load(f)
            for attr, value in attrs.items():
                setattr(self, attr, value)

        self.model = tf.keras.models.load_model(model_dir)

    def _preproc(self, *, df: pd.DataFrame) -> np.ndarray:
        """We are taking first differences of log return. Thus we calculate
        log return, and take first differences.

        Args:
            data (pd.DataFrame): input dataframe containing 'close' field

        Returns:
            np.ndarray: log return numpy array.
        """
        df["log_ret"] = np.log(df["close"]) - np.log(df["close"].shift(1))
        df["log_ret_diff"] = df["log_ret"].diff()
        df.dropna(inplace=True)
        return df["log_ret_diff"].to_numpy().astype("float16")

    def _postproc(self, *, pred: float, log_ret: float, close: float) -> float:
        """Take the predicted log return diff, log return, close, and return
        the predicted close.

        predicted log return = log return + predicted log return diff
        predicted close = close * e^(predicted_log_return)

        Args:
            pred (float): [description]
            log_ret (float): [description]
            close (float): [description]

        Returns:
            float: [description]
        """
        next_log_ret = log_ret + pred
        return close * np.exp(next_log_ret)
