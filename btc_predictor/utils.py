import sys, os
from pathlib import Path
import pandas as pd
import numpy as np

def preproc(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Convenience function for cleaning up .csv file scrapped from Coinmarketcap.com
	
	Parameters
	==========
	df : input dataframe, with columns as {Date, Open, High, Low, Close, Volume, 
	and Market Cap. Everything in text format. Unknown values are marked as '-'
	
	Returns
	==========
	df : original dataframe cleaned and parsed into numerics. Unknown as NaN.
	
	"""
	df.Date = pd.to_datetime(df.Date)
	df = df.sort_values(by = 'Date')
	df.set_index('Date', inplace=True)
	
	df.Open = df.Open.str.replace(",", "")
	df.High = df.High.str.replace(",", "")
	df.Low = df.Low.str.replace(",", "")
	df.Close = df.Close.str.replace(",", "")
	df.Volume = df.Volume.str.replace(",", "")
	df["Market Cap"] = df["Market Cap"].str.replace(",", "")
	df.Volume = df.Volume.replace("-", np.nan)
	
	df = df.apply(pd.to_numeric)
	
	return df