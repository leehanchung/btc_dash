from dash import Dash
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from statsmodels.tsa.arima_model import ARIMA

from btc_dash.db import get_ohlcv_data
from btc_dash import config


def register_ohlcv_callback(app: Dash):
    """Wrapper function for registering callback to generate momentum
    indicator figure using plotly

    Args:
        dash app object

    Returns:
        None

    """
    @app.callback(
        Output("btcusd-ohlcv", "figure"),
        [Input("btcusd-ohlcv-update", "n_intervals")],
    )
    def gen_ohlcv(interval: int) -> go.Figure:
        """Generate OHLCV Chart for BTCUSD with predicted price overlay.

        Args:
            interval: update the graph based on an interval

        """
        # hack to wrap interval around available data.  OOS starts at 1500,
        # df has a total of 2274 rows after processing to wrap around
        # 2274-1500 ~ 750. Reset prediction data to empty df.
        interval = interval % 750

        print("interva is {}...".format(interval))

        # read data from source
        df = get_ohlcv_data(interval - 100, interval)
        df["log_ret"] = np.log(df.Close) - np.log(df.Close.shift(1))

        print("\ndata df loaded, starting prediction...\n")
        # online training and forecast.
        model = ARIMA(df.tail(60)["log_ret"], order=(3, 1, 0), freq="D",).fit(
            disp=0
        )
        pred = model.forecast()[0]

        print("\nprediction ended, writing to output df...")

        # save forecast to output dataframe. should be dB irl.
        next_dt = df.tail(1).index[0] + pd.Timedelta("1 day")
        config.df_pred.loc[next_dt] = [
            pred[0],
            (np.exp(pred) * df.tail(1).Close.values)[0],
        ]
        print("\nnext datetime is {}...".format(next_dt))
        # get index location of period.
        loc = config.df_pred.index.get_loc(next_dt) + 1
        print("\nloc is {}...".format(loc))

        # slices for the past N periods perdiction for plotting
        df_pred_plot = config.df_pred.iloc[
            slice(max(0, loc - 30), min(loc, len(df)))
        ].sort_index()
        print("\n set pred df for plotting...\n", df_pred_plot)

        # plotting ohlc candlestick
        trace_ohlc = go.Candlestick(
            x=df.tail(50).index,
            open=df["Open"].tail(50),
            close=df["Close"].tail(50),
            high=df["High"].tail(50),
            low=df["Low"].tail(50),
            opacity=0.5,
            hoverinfo="skip",
            name="BTCUSD",
        )

        # plotting prediction line
        trace_line = go.Scatter(
            x=df_pred_plot.index,
            y=df_pred_plot.pred_Close,
            line_color="yellow",
            mode="lines+markers",
            name="Predicted Close",
        )

        layout = go.Layout(
            plot_bgcolor=config.app_color["graph_bg"],
            paper_bgcolor=config.app_color["graph_bg"],
            font={"color": "#fff"},
            height=700,
            xaxis={
                "showline": False,
                "showgrid": False,
                "zeroline": False,
            },
            yaxis={
                "showgrid": True,
                "showline": True,
                "fixedrange": True,
                "zeroline": True,
                "gridcolor": config.app_color["graph_line"],
                "title": "Price (USD$)",
            },
        )

        return go.Figure(data=[trace_ohlc, trace_line], layout=layout)
