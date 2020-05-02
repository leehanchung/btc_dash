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


def cvScore(
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
    from sklearn.metrics import log_loss, accuracy_score
    from clfSequential import PurgedKFold  # ???????

    if cvGen is None:
        cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged
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
                pred,
                sample_weights=sample_weight.iloc[test].values,
            )
        score.append(score_)
    return np.array(score)


def get_current_time():
    """ Helper function to get the current time in seconds. """

    now = dt.datetime.now()
    total_time = (now.hour * 3600) + (now.minute * 60) + (now.second)
    return total_time
