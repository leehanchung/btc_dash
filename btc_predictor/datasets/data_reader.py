# from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf


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

    def read_parquet(self, parquet_file: str) -> None:
        """Read parquet data file using pyarrow into a pandas dataframe

        Args:
            parquet_file: name of the parquet file.

        Returns:
            Pandas dataframe containing data in the parquet file

        """
        return pq.read_table(source=parquet_file).to_pandas()

    @property
    def pd(self) -> pd.DataFrame:
        """Returns the dataset in pandas dataframe format

        Args:
            None

        Returns:
            Pandas dataframe containing data in the parquet file

        """
        return self.data

    @property
    def tfds(self) -> tf.data.Dataset:
        """Returns the dataset in tf.data format

        Args:
            None

        Returns:
            Pandas dataframe containing data in the parquet file

        """
        raise NotImplementedError

    @staticmethod
    def create_tfds_from_np(*,
                            data: np.ndarray,
                            window_size: int = 31,
                            shift_size: int = 1,
                            stride_size: int = 1,
                            batch_size: int = 15) -> tf.data.Dataset:
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
        data = data.window(size=window_size,
                           shift=shift_size,
                           stride=stride_size,
                           drop_remainder=True)
        data = data.flat_map(lambda window: window.batch(window_size,
                                                         drop_remainder=True))
        data = data.map(lambda window: (window[:-1],
                                        tf.reshape(window[-1:], [])))
        data = data.cache().shuffle(batch_size).batch(batch_size).repeat()

        return data
