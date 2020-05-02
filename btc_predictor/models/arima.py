# import numpy as np
import pandas as pd

from statsmodels.tsa.arima_model import ARIMA

# from statsmodels.tsa.arima_model import ARIMAResults
# from sklearn.metrics import accuracy_score, mean_absolute_error
# from sklearn.metrics import mean_squared_error, confusion_matrix

# import joblib


def arima(df: pd.DataFrame) -> pd.DataFrame:

    start = 1700
    end = 2031
    window = 60
    y_rolling_arima = pd.Series([])
    periods = range(start - window, end, 9)

    ptrain = clone.iloc[(p - window) : p, :]["log_ret"]
    ptest = clone.iloc[p : p + 9, :]["log_ret"]
    model = ARIMA(ptrain, order=(3, 1, 0), freq="D").fit(disp=0)
    predict = model.predict(ptest.index[0], ptest.index[-1], dynamic=True)
    y_rolling_arima = y_rolling_arima.append(predict)

    return df
