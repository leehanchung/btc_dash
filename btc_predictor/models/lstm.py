from typing import Tuple, Dict, Callable
import numpy as np
import tensorflow as tf
from btc_predictor.models import BaseModel
from btc_predictor.datasets import DataReader
from btc_predictor.utils import print_metrics


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

    def __init__(self, *,  model: Callable, model_args: Dict = None):
        super().__init__(model=model,
                         model_args=model_args)

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

    def fit(self, *, data: DataReader) -> None:
        """Function that accept input training data and train the model

        Args:
            data: Features required by the model to generate a
            prediction. Numpy array of shape (1, n) where n is the dimension
            of the feature vector.

        Returns:
            None
        """
        # load data
        df = data.pd
        df['log_ret'] = np.log(df.Close) - np.log(df.Close.shift(1))
        df.dropna(inplace=True)

        # preprocess
        time_series_data = np.diff(df['log_ret'].to_numpy()).astype('float32')
        train = time_series_data[:self.TRAIN_SIZE]
        val = time_series_data[self.TRAIN_SIZE:self.VAL_SIZE+self.TRAIN_SIZE]
        # self.test = time_series_data[self.VAL_SIZE+self.TRAIN_SIZE:]

        train_tfds = data.create_tfds_from_np(
            data=train,
            window_size=self.WINDOW_SIZE,
            batch_size=self.BATCH_SIZE
        )
        val_tfds = data.create_tfds_from_np(
            data=val,
            window_size=self.WINDOW_SIZE,
            batch_size=self.BATCH_SIZE
        )
        # test_tfds = self.data.create_tfds_from_np(
        #     data=test,
        #     window_size=self.WINDOW_SIZE,
        #     batch_size=1
        # )
        print(f'Total daily data: {df.shape[0]} days')

        lstm_model = self.model(
            input_shape=(self.WINDOW_SIZE-1, 1),
            dropout=0.4,
            num_forward=1,
        )
        lstm_model.compile(
            optimizer='adam',
            loss='mse'
        )

        train_history = lstm_model.fit(
            train_tfds,
            epochs=self.EPOCHS,
            steps_per_epoch=self.EVALUATION_INTERVAL,
            validation_data=val_tfds,
            validation_steps=self.VALIDATION_STEPS
        )

        self.history = train_history
        return None

    def eval(self, *, data: DataReader):
        """Function that accept input training data and train the model

        Args:
            data: Features required by the model to generate a
            prediction. Numpy array of shape (1, n) where n is the dimension
            of the feature vector.

        Returns:
            eval_scores: a tuple of MSE and MAE scores.
        """

        test_y_true = np.array([])
        test_y_pred = np.array([])
        for x, y in self.test_tfds.take(self.WALK_FORWARD):
            test_y_true = np.append(test_y_true, y[0].numpy())
            test_y_pred = np.append(test_y_pred, self.modle.predict(x)[0])

        print_metrics(y_true=test_y_true, y_pred=test_y_pred)
        return None
