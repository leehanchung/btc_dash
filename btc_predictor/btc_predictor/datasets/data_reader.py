import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import tensorflow as tf

_logger = logging.getLogger(__name__)


class DataReadingError(Exception):
    pass


class DataReader:
    """Load either 1 minute data with no header and columns representing
    [posix_timestamp, price, volumne in btc] in parquet format or daily
    btcusd data with header in csv format.

    Data can be accessed with .pd for pandas format or .tfds for tensorflow
    dataset format.
    """

    def __init__(self, *, data_file: str) -> None:
        self.datafile = data_file

        if data_file.split(".")[-1] == "csv":
            self.source_type = "csv"
        elif data_file.split(".")[-1] == "parquet":
            self.source_type = "parquet"
        else:
            raise DataReadingError("Invalid datatype")

        # Read the data into pandas dataframe
        if self.source_type == "csv":
            self.data = self.read_csv(csv_file=data_file)
        else:
            self.data = self.read_parquet(parquet_file=data_file)

        self.data_file = data_file
        self.__name__ = f"{self.data_file}"

    def read_csv(self, *, csv_file: str) -> None:
        """Read parquet data file using pyarrow into a pandas dataframe

        Args:
            csv_file: name of the csv file.

        Returns:
            Pandas dataframe containing data in the csv file

        """
        data = pd.read_csv(csv_file, thousands=",")

        data["Date"] = pd.to_datetime(data["Date"])
        data = data.sort_values(by="Date")
        data.set_index("Date", inplace=True)
        return data

    @property
    def pd(self) -> pd.DataFrame:
        """Returns the dataset in pandas dataframe format

        Args:
            None

        Returns:
            Pandas dataframe containing data in the parquet file

        """
        return self.data

    @staticmethod
    def create_tfds_from_np(
        *,
        data: np.ndarray,
        window_size: int = 31,
        shift_size: int = 1,
        stride_size: int = 1,
        batch_size: int = 15,
    ) -> tf.data.Dataset:
        """Generates tf.data dataset using a given numpy array using tf.data API

        Args:
            data: np.ndarray data in numpy array, to be flattened
            window_size: size of the moving window
            shift_size: step size of the moving window,
                e.g. [0, 1, 2, 3, 4, 5, 6] with shift 2 and window 3
                -> [0, 1, 2], [2, 3, 4], ...
            stride_size: sampling size of the moving window,
                e.g., [0, 1, 2, 3, 4, 5, 6] with stride 2 and window 3
                -> [0, 2, 4], [1, 3, 5], ...
            batch_size: batch size of the created data

        Returns:
            tf.data.Dataset

        """
        data = tf.data.Dataset.from_tensor_slices(data.reshape(-1, 1))
        data = data.window(
            size=window_size,
            shift=shift_size,
            stride=stride_size,
            drop_remainder=True,
        )
        data = data.flat_map(
            lambda window: window.batch(window_size, drop_remainder=True)
        )
        data = data.map(
            lambda window: (window[:-1], tf.reshape(window[-1:], []))
        )
        data = data.shuffle(batch_size).batch(batch_size).cache().repeat()

        return data


class BitfinexCandlesAPI:
    def __init__(
        self,
        *,
        period: str = "1m",
        symbol: str = "tBTCUSD",
        section: str = "hist",
    ):
        self.period = period
        self.symbol = symbol
        self.section = section
        self.url = (
            "https://api-pub.bitfinex.com/v2/candles/trade"
            f":{period}:{symbol}/{section}"
        )
        self.start_time = None
        self.end_time = None
        self.data = None

    def load(
        self, start_time: int = 1610000000000, limit: int = 10000
    ) -> None:
        """Loads data from Bitfinex Candles API given an Unix timestamp[ms].
        Currently Bitfinex limits number of candle requested to 10,000, so
        we default limit to 10,000.

        For simplicity sake, setting start_time to 1610000000000 for now.

        Detailed Bitfinex API doc is here:
        https://docs.bitfinex.com/reference#rest-public-candles

        Args:
            start_time (int): Unix timestamp[ms]
            limit (int, optional): number of candles requested. Defaults to
                                   10000.
        """
        self.start_time = datetime.fromtimestamp(start_time // 1000).strftime(
            "%Y%m%d"
        )
        self.end_time = datetime.fromtimestamp(
            (start_time + limit) // 1000
        ).strftime("%Y%m%d")

        params = {"start": start_time, "limit": limit, "sort": 1}
        response = requests.get(self.url, params=params)

        payload = json.loads(response.content)
        columns = ["timestamp", "open", "close", "high", "low", "volume"]
        data = pd.DataFrame(payload, columns=columns)
        data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")

        self.data = data

    @staticmethod
    def create_tfds_from_np(
        *,
        data: np.ndarray,
        window_size: int = 31,
        shift_size: int = 1,
        stride_size: int = 1,
        batch_size: int = 15,
    ) -> tf.data.Dataset:
        """Generates tf.data dataset using a given numpy array using tf.data API

        Args:
            data: np.ndarray data in numpy array, to be flattened
            window_size: size of the moving window
            shift_size: step size of the moving window,
                e.g. [0, 1, 2, 3, 4, 5, 6] with shift 2 and window 3
                -> [0, 1, 2], [2, 3, 4], ...
            stride_size: sampling size of the moving window,
                e.g., [0, 1, 2, 3, 4, 5, 6] with stride 2 and window 3
                -> [0, 2, 4], [1, 3, 5], ...
            batch_size: batch size of the created data

        Returns:
            tf.data.Dataset

        """
        data = tf.data.Dataset.from_tensor_slices(data.reshape(-1, 1))
        data = data.window(
            size=window_size,
            shift=shift_size,
            stride=stride_size,
            drop_remainder=True,
        )
        data = data.flat_map(
            lambda window: window.batch(window_size, drop_remainder=True)
        )
        data = data.map(
            lambda window: (window[:-1], tf.reshape(window[-1:], []))
        )
        data = data.shuffle(batch_size).batch(batch_size).cache().repeat()

        return data

    @property
    def pd(self) -> pd.DataFrame:
        """Returns the dataset in pandas dataframe format

        Args:
            None

        Returns:
            Pandas dataframe containing data in the parquet file

        """
        return self.data
