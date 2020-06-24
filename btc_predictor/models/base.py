from typing import Any
from btc_predictor.datasets import DataReader


class BaseModel:
    """BaseModel class to provide an unify APIs of multiple different modeling
    framework choices.
    """
    def __init__(self, model):
        self.model = model

    def predict(self, input_features: Any):
        """Function that accept input data for the model to generate a prediction

        Args:
            model: a machine learning model.
            input_features: Features required by the model to generate a 
            prediction. Numpy array of shape (1, n) where n is the dimension
            of the feature vector.

        Returns:
            prediction: Prediction of the model. Numpy array of shape (1,).
        """
        raise NotImplementedError

    def fit(self, data: DataReader):
        raise NotImplementedError

    def eval(self, data: DataReader):
        raise NotImplementedError

    def load(self, data):
        raise NotImplementedError

    def save(self, data):
        raise NotImplementedError
