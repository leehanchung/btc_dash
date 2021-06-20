from typing import Dict

import pandas as pd
import pmdarima as pm
from pmdarima.arima import auto_arima


def stepwise_arima(*, df: pd.Series, config: Dict) -> pm.arima.arima.ARIMA:
    """Read parquet data file using pyarrow into a pandas dataframe

    Args:
        df: endogenous variable in the form of pandas dataframe/series
        config: configuration dictionary for model search parameters

    Returns:
        PMD ARIMA model, ready to be fitted.

    """
    return auto_arima(
        df,
        start_p=0,
        start_q=0,
        max_d=5,
        max_p=16,
        max_q=5,
        m=12,
        scoring="mse",
        start_P=0,
        max_order=20,
        random_state=78,
        seasonal=False,
        d=1,
        D=1,
        trace=True,
        information_criterion="aic",
        error_action="ignore",
        stationary=False,
        suppress_warnings=True,
        with_intercept=False,
        stepwise=True,
        maxiter=100,
        n_jobs=24,
    )
