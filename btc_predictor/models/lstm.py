from typing import Tuple
import tensorflow as tf


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
