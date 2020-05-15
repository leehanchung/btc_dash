from typing import List
import datetime as dt

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt


def show_plot(*,
              plot_data: List[np.ndarray],
              delta: int = 0,
              title: str,
              labels: List[str] = ['History', 'True Future', 'Prediction'],
              marker: List[str] = ['.-', 'rx', 'go']) -> plt:
    """Plot time series historical and walk-fowards data on the same chart.

    Args:
        plot_data: List of numpy arrays to be plotted, with historical data at
            as the first element
        delta: Walk-forward delta to be plotted
        title: title of the chart
        label: labels of the individual series
        market: markers of the individual series

    Returns:
        A matplotlib pyplot chart object
    """
    def create_time_steps(length):
        return list(range(-length, 0))

    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            if not delta:
                # plot single step forward in the future
                plt.plot(future,
                         plot_data[i].flatten(),
                         marker[i],
                         markersize=10,
                         label=labels[i])
            else:
                # plot multi step forward in the future
                plt.plot(range(future),
                         plot_data[i].flatten(),
                         marker[i],
                         markersize=10,
                         label=labels[i])
        else:
            # plot the historical data
            plt.plot(time_steps,
                     plot_data[i].flatten(),
                     marker[i],
                     label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+10)])
    plt.xlabel('Time-Step')

    return plt


def print_metrics(*,
                  y_true: List[np.ndarray],
                  y_pred: List[np.ndarray]) -> None:
    """Convenience function for displaying RMSE and directional accuracy

    Args:
        y_true: numpy array, shape (N,)
        y_pred: numpy array, shape (N,)

    Returns:
        Original dataframe cleaned and parsed into numerics. Unknown as NaN.
    """
    mse = np.sqrt(mean_squared_error(y_true, y_pred))
    accuracy = accuracy_score(np.sign(y_true), np.sign(y_pred))
    print(f"Prediction RMSE: {mse:.4f}, directional accuracy: {accuracy:.4f}")


def preproc(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function for cleaning up .csv file scrapped from
    Coinmarketcap.com and bitstamp.

    Args:
        df : input dataframe, with columns as
        {Date, Open, High, Low, Close, Volume}.
        Everything in text format. Unknown values are marked as '-'

    Returns:
        Original dataframe cleaned and parsed into numerics. Unknown as NaN.
    """
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

    return df


def cv_score(
    clf,
    X,
    y,
    sample_weight,
    scoring="neg_log_loss",
    t1=None,
    cv=None,
    cvGen=None,
    pctEmbargo=None,
):
    if scoring not in ["neg_log_loss", "accuracy"]:
        raise Exception("wrong scoring method.")
    from sklearn.metrics import log_loss
    from clfSequential import PurgedKFold  # ???????

    if cvGen is None:
        cvGen = PurgedKFold(
            n_splits=cv, t1=t1, pctEmbargo=pctEmbargo
        )  # purged
    score = []
    for train, test in cvGen.split(X=X):
        fit = clf.fit(
            X=X.iloc[train, :],
            y=y.iloc[train],
            sample_weight=sample_weight.iloc[train].values,
        )
        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X.iloc[test, :])
            score_ = -log_loss(
                y.iloc[test],
                prob,
                sample_weight=sample_weight.iloc[test].values,
                labels=clf.classes_,
            )
        else:
            prob = fit.predict(X.iloc[test, :])
            score_ = accuracy_score(
                y.iloc[test],
                prob,
                sample_weights=sample_weight.iloc[test].values,
            )
        score.append(score_)
    return np.array(score)


def get_current_time():
    """ Helper function to get the current time in seconds. """

    now = dt.datetime.now()
    total_time = (now.hour * 3600) + (now.minute * 60) + (now.second)
    return total_time
