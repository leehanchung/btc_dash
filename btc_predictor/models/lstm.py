from typing import Tuple
import numpy as np
import tensorflow as tf
from btc_predictor.models import BaseModel


class LSTM_Model(tf.keras.Model):
    """Two layer LSTM model using Tensorflow API. Can either be compiled
    using Keras API or using tf.GradientTape() to train. This is
    equivalent to Keras Functional API:

        inputs = Input(shape=input_shape)
        x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.LSTM(64)(x)
        outputs = tf.keras.layers.Dense(1)(x)

        simple_lstm_model = Model(inputs=inputs,
                                  outputs=outputs,
                                  name="univariate_lstm")

    Or, Keras Sequential API:
        simple_lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(128,
                                 input_shape=input_shape,
                                 return_sequences=True),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1)
        ])
    """
    def __init__(self,
                 *,
                 input_shape: Tuple[int, int],
                 dropout: float,
                 num_forward: int):
        super(LSTM_Model, self).__init__()
        self.lstm_input = tf.keras.layers.LSTM(128,
                                               input_shape=input_shape,
                                               return_sequences=True)
        self.lstm_hidden = tf.keras.layers.LSTM(64)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense = tf.keras.layers.Dense(num_forward)

    @tf.function
    def call(self, x):
        # x = self.inputs(inputs)
        x = self.lstm_input(x)
        x = self.dropout(x)
        x = self.lstm_hidden(x)
        x = self.dropout(x)
        out = self.dense(x)
        return out


class LSTMModel(BaseModel):
    """Wrapper class to convert LSTM_Model BaseModel API
    """
    RANDOM_SEED = 78

    def __init__(self, model: LSTM_Model):
        self.model = model
        self.TRAIN_SIZE = 1680
        self.VAL_SIZE = 180
        self.WINDOW_SIZE = 16
        self.BATCH_SIZE = 256
        self.EPOCHS = 15
        self.EVALUATION_INTERVAL = 64
        self.VALIDATION_STEPS = 64
        self.WALK_FORWARD = 30

    def predict(self):
        raise NotImplementedError

    def fit(self):

        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError
