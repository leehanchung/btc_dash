from typing import Any, Dict, Callable, Tuple
import numpy as np
import tensorflow as tf
from btc_predictor.datasets import DataReader


class ModelLoadingError(Exception):
    pass


class ModelSavingError(Exception):
    pass


class BaseModel:
    """Basemodel class API, to be subclassed by different data and
    modeling framework choices.
    """
    def __init__(self, *, dataset: Callable, model: Callable,
                 dataset_args: Dict = None, model_args: Dict = None):
        self.name = (f'{self.__class__.__name__}',
                     f'_{dataset.__name__}_{model.__name__}')

        if dataset_args is None:
            dataset_args = {}
        self.data = DataReader(**dataset_args)

        if model_args is None:
            model_args = {}
        self.model = model(**model_args)

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
            data: Features required by the model to generate a
            prediction. Numpy array of shape (1, n) where n is the dimension
            of the feature vector.

        Returns:
            eval_scores: a tuple of MSE and MAE scores.
        """
        raise NotImplementedError

    def load(self, *, model_file: str) -> bool:
        """Function that loads pretrained weights for making a prediction.
        Currently only TF is supported.
        # TODO: extend for AWS S3 support

        Args:
            model_file. serialized model weights file

        Returns:
            status: success of fail
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
            status: success of fail
        """
        try:
            self.model.save(f'{self.name}.h5')
        except ModelSavingError:
            return False
        return True
