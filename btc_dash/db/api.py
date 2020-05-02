import pathlib
import sqlite3
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
    """
	Query wind data rows between two ranges
	:params start: start row id
	:params end: end row id
	:returns: pandas dataframe object 
	"""

    # con = sqlite3.connect(str(DB_FILE))
    # statement = f'SELECT Speed, SpeedError, Direction FROM Wind WHERE rowid > "{start}" AND rowid <= "{end}";'
    # df = pd.read_sql_query(statement, con)
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
        return df.iloc[OOS_START + start : OOS_START + end, :]
