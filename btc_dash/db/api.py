import pathlib
import pandas as pd
import numpy as np


CSV_FILE = (
    pathlib.Path(__file__).resolve().parent.joinpath("btcusd.csv").resolve()
)
OUT_FILE = (
    pathlib.Path(__file__)
    .resolve()
    .parent.joinpath("btcusd_predict.csv")
    .resolve()
)
OOS_START = 1500


def get_ohlcv_data(start, end):
    """Query OHLCV data rows between two ranges

    Args:
        start: start row id
        end: end row id

    Returns:
        pandas dataframe object

    """
    df = pd.read_csv(CSV_FILE)
    df.Date = pd.to_datetime(df.Date)
    df = df.sort_values(by="Date")
    df.set_index("Date", inplace=True)

    df.Open = df.Open.str.replace(",", "")
    df.High = df.High.str.replace(",", "")
    df.Low = df.Low.str.replace(",", "")
    df.Close = df.Close.str.replace(",", "")
    df.Volume = df.Volume.str.replace(",", "")
    df["Market Cap"] = df["Market Cap"].str.replace(",", "")
    df.Volume = df.Volume.replace("-", np.nan)
    df = df.apply(pd.to_numeric)

    if (OOS_START + end) > df.shape[0]:
        return df.tail(50)
    else:
        s = OOS_START + start
        e = OOS_START + end
        return df.iloc[s:e, :]
