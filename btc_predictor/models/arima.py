import numpy as np
import pandas as pd
import talib


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix

import joblib

def arima(df: pd.DataFrame) -> pd.DataFrame:
	return df