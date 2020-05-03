import numpy as np
from dash.dependencies import Input, Output
from dash.dependencies import State
import plotly.graph_objs as go
from sklearn.metrics import confusion_matrix

from btc_dash.db import get_ohlcv_data
from btc_dash import config


def register_confusion_callback(app):

    @app.callback(
        Output("confusion-matrix", "figure"),
        [Input("btcusd-ohlcv-update", "n_intervals")],
        [State("btcusd-ohlcv", "figure"),],
    )
    def gen_confusion_matrix(interval, ohlcv_figure):
        """
        Genererate confusion matrix of prediction directions.

        :params interval: upadte the graph based on an interval
        :params ohlcv_figure: current ohlcv chart, not used. LOL.
        """

        # hack to wrap interval around available data.  OOS starts at 1500, df has a
        # total of 2274 rows after processing to wrap around 2274-1500 ~ 750. Reset
        # prediction data to empty df.
        interval = interval % 750

        df = get_ohlcv_data(interval - 50, interval)
        df["log_ret"] = np.log(df.Close) - np.log(df.Close.shift(1))

        if config.df_pred.shape[0] < 30:
            p = config.df_pred.shape[0]
            cm = confusion_matrix(
                np.sign(df.log_ret.tail(p).values),
                np.sign(config.df_pred.pred_log_ret.tail(p).values),
            )
            # print(len(cm))
            if len(cm) == 0 or len(cm) == 1:
                cm = [[1, 1], [1, 1]]

            cm = np.array(cm) / p
        else:
            cm = confusion_matrix(
                np.sign(df.log_ret.tail(30).values),
                np.sign(config.df_pred.pred_log_ret.tail(30).values),
            )
            cm = np.array(cm) / 30

        # generate text for confusion metics. dont know how to display
        # text on plotly go
        cm_text = np.around(cm, decimals=2)

        data = go.Heatmap(
            z=cm,
            x=["Predicted Down", "Predicted Up"],
            y=["Actual Up", "Actual Down"],
            zmin=0.0,
            zmax=1.0,
            opacity=0.8,
        )

        layout = go.Layout(
            height=350,
            plot_bgcolor=config.app_color["graph_bg"],
            paper_bgcolor=config.app_color["graph_bg"],
            font={"color": "#fff"},
            autosize=True,
            hovermode="closest",
            legend={
                # "orientation": "h",
                # "yanchor": "bottom",
                # "xanchor": "center",
                # "y": 1,
                # "x": 0.5,
            },
        )

        return go.Figure(data=data, layout=layout)
