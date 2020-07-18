from typing import Any, Dict, Tuple
import numpy as np
import tensorflow as tf
from btc_predictor.datasets import DataReader


class ModelLoadingError(Exception):
    pass


class ModelSavingError(Exception):
    pass


class ModelDataError(Exception):
    def __init__(self, message="Dataset has not been defined"):
        self.message = message
        super.__init__(self.message)

    def __str__(self):
        return f'{self.message}'


class BaseModel:
    """BaseModel provides an unified fit, eval, predict, load, and save API
    to accomodate different data and modeling frameworks.
    """
    # def __init__(self, *, model: Callable, model_args: Dict = None):
    def __init__(self, *, model_args: Dict = None, train_args: Dict = None):
        # dataset: Callable, dataset_args: Dict = None):
        # self.name = (f'{self.__class__.__name__}',
        #              f'_{dataset.__name__}_{model.__name__}')

        # if dataset_args is None:
        #     dataset_args = {}
        # self.data = DataReader(**dataset_args)

        if model_args is None:
            model_args = {}

        if train_args is None:
            train_args = {}

    def fit(self, *, data: DataReader) -> None:
        """Function that accept input training data and train the model

        Args:
            data: Features required by the model to generate a
            prediction. Numpy array of shape (1, n) where n is the dimension
            of the feature vector.

        Returns:
            None
        """
        raise NotImplementedError

    def eval(self, *, data: DataReader) -> Tuple[float]:
        """Function that accept input training data and train the model

        Args:
            data (DataReader): Features required by the model to generate a
            prediction. Numpy array of shape (1, n) where n is the dimension
            of the feature vector.

        Raises:
            NotImplementedError: [description]

        Returns:
            Tuple[float]: a tuple of RMSE, directional accuracy, and mean
            directional accuracy scores.
        """
        raise NotImplementedError

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

    def load(self, *, model_file: str) -> bool:
        """Function that loads pretrained weights for making a prediction.
        Currently only TF is supported.
        # TODO: extend for AWS S3 support

        Args:
            model_file (str): serialized model weights file

        Returns:
            bool: success of fail
        """
        try:
            self.model = tf.keras.models.load_model(model_file)
        except ModelLoadingError:
            return False
        return True

    def save(self) -> bool:
        """Function that saves a serialized model. Currently only TF is supported.
        # TODO: extend for AWS S3 support

        Args:
            None

        Returns:
            bool: success of fail
        """
        raise NotImplementedError
