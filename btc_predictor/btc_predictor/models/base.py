from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from btc_predictor.datasets import BaseDataset


class ModelLoadingError(Exception):
    """Custom error that's raised when problem arises at model loading"""

    def __init__(self, message: str = "Model loading error") -> None:
        self.message = message
        super().__init__(message)

    def __str__(self):
        return f"{self.message}"


class ModelSavingError(Exception):
    """Custom error that's raised when problem arises at model saving"""

    def __init__(self, message: str = "Model saving error") -> None:
        self.message = message
        super().__init__(message)

    def __str__(self):
        return f"{self.message}"


class ModelTrainingError(Exception):
    """Custom error that's raised when problem arises at model training"""

    def __init__(self, message="Dataset has not been defined"):
        self.message = message
        super.__init__(self.message)

    def __str__(self):
        return f"{self.message}"


class ModelDataError(Exception):
    def __init__(self, message="Dataset has not been defined"):
        self.message = message
        super.__init__(self.message)

    def __str__(self):
        return f"{self.message}"


class BasePredictor(ABC):
    """BasePredictor provides an unified fit, eval, predict, load, and save API
    to accomodate different data and modeling frameworks.
    """

    @abstractmethod
    def train(self, *, data: BaseDataset) -> None:
        """Function that accept input training data and train the model

        Args:
            data: Features required by the model to generate a
            prediction. Numpy array of shape (1, n) where n is the dimension
            of the feature vector.

        Returns:
            None
        """
        pass

    @abstractmethod
    def eval(self, *, data: BaseDataset) -> Tuple[float]:
        """Function that accept input training data and train the model

        Args:
            data (DataReader): Features required by the model to generate a
            prediction. Numpy array of shape (1, n) where n is the dimension
            of the feature vector.

        Returns:
            Tuple[float]: a tuple of RMSE, directional accuracy, and mean
            directional accuracy scores.
        """
        pass

    @abstractmethod
    def predict(self, *, X: np.ndarray) -> np.ndarray:
        """Function that accept input data for the model to generate a prediction

        Args:
            input_features: Features required by the model to generate a
            prediction. Numpy array of shape (1, n) where n is the dimension
            of the feature vector.

        Returns:
            prediction: Prediction of the model. Numpy array of shape (1,).
        """
        pass

    @abstractmethod
    def save(self, *, origin_pwd: bool = False) -> None:
        """Function that saves a serialized model.

        Args:
            None

        Returns:
            None
        """
        pass

    @abstractmethod
    def load(self, *, model_name: str, origin_pwd: bool = False) -> None:
        """Function that loads pretrained weights for making a prediction.

        Args:
            model_file (str): serialized model weights file

        Returns:
            bool: success of fail
        """
        pass
