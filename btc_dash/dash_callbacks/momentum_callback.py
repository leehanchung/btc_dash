from dash import Dash
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objs as go

from btc_dash.db import get_ohlcv_data
from btc_dash import config


def register_momentum_callback(app: Dash):
    """Wrapper function for registering callback to generate momentum
    indicator figure using plotly

    Args:
        dash app object

    Returns:
        None

    """
    @app.callback(
        Output("momentum-gauge", "figure"),
        [Input("btcusd-ohlcv-update", "n_intervals")],
    )
    def gen_momentum_gauge(interval: int):
        """Generate 5 period lag RSI on BTCUSD Close and plot it as Momentum Gauge

        Args:
            interval: integer. update the graph based on an interval

        Returns:
            Plotly graph object figure.

        """
        # hack to wrap interval around available data.  OOS starts at 1500,
        # df has a total of 2274 rows after processing to wrap around
        # 2274-1500 ~ 750.
        interval = interval % 750

        # read data from source and calculate RSI.
        # RSI ranges between 0 and 100.
        df = get_ohlcv_data(interval - 6, interval)
        # rsi = int(round(talib.RSI(df.Close.values, 5)[-1]))
        # print(rsi)
        rsi2 = int(round(RSI(df.Close, 5)[-1]))
        # print(rsi2)

        # Let's subdivide RSI into 10s to reduce plotting
        # dial triangle complexity
        angle = round(rsi2, -1)
        # center of dial coordinate is 0.24 0.5. We plot left top and
        # right coordinates of a triangle
        dials_dict = {
            0: "M 0.24 0.4950 L 0.09 0.5 L 0.24 0.505 Z",
            10: "M 0.2384 0.4952 L 0.0973 0.5463 L 0.2415 0.5047 Z",
            20: "M 0.2370 0.4959 L 0.1186 0.5881 L 0.2429 0.5040 Z",
            30: "M 0.2359 0.4970 L 0.1518 0.6213 L 0.2440 0.5029 Z",
            40: "M 0.2352 0.4985 L 0.1936 0.6247 L 0.2447 0.5015 Z",
            50: "M 0.235 0.5 L 0.24 0.65 L 0.245 0.5 Z",  # confirmed)
            60: "M 0.2352 0.5015 L 0.2863 0.6426 L 0.2447 0.4984 Z",
            70: "M 0.2359 0.5029 L 0.3281 0.6213 L 0.244 0.497 Z",
            80: "M 0.2370 0.5040 L 0.3613 0.5881 L 0.2429 0.4959 Z",
            90: "M 0.2384 0.5047 L 0.3826 0.5463 L 0.2415 0.4952 Z",
            100: "M 0.24 0.505 L 0.39 0.50 L 0.24 0.495 Z",
        }

        # first we trace the dial using pie chart, hiding bottom half.
        trace1 = go.Pie(
            values=[50, 10, 10, 10, 10, 10],
            labels=["RSI Index", "HODL", "HELP", "MEH", "NICE", "FOMO"],
            domain={"x": [0, 0.48]},
            marker_colors=[
                config.app_color["graph_bg"],
                "rgb(232,226,202)",
                "rgb(226,210,172)",
                "rgb(223,189,139)",
                "rgb(223,162,103)",
                "rgb(226,126,64)",
            ],
            name="Gauge",
            hole=0.3,
            direction="clockwise",
            rotation=90,
            showlegend=False,
            hoverinfo="none",
            textinfo="label",
            textposition="inside",
        )

        # then we add numerical labels to the same pie chart
        trace2 = go.Pie(
            values=[40, 10, 10, 10, 10, 10, 10],
            labels=[".", "0", "20", "40", "60", "80", "100"],
            domain={"x": [0, 0.48]},
            marker_colors=["rgba(255, 255, 255, 0)"] * 7,
            hole=0.4,
            direction="clockwise",
            rotation=108,
            showlegend=False,
            hoverinfo="none",
            textinfo="label",
            textposition="outside",
        )

        layout = go.Layout(
            height=350,
            plot_bgcolor=config.app_color["graph_bg"],
            paper_bgcolor=config.app_color["graph_bg"],
            font={"color": "#fff"},
            autosize=True,
            margin=dict(l=200, autoexpand=True),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,),
            # this is the hand/triangle on the dial.
            # https://plot.ly/python/gauge-charts/#dial center is 0.24, 0.5.
            # 2019/08/01: ^ and the above coordinate is not exactly correct
            # so the angles and magnitutdes are off.
            shapes=[
                dict(
                    type="path",
                    path=dials_dict[angle],
                    fillcolor="rgba(44, 160, 101, 0.5)",
                    line_width=1,
                    xref="paper",
                    yref="paper",
                )
            ],
            annotations=[
                dict(
                    xref="paper",
                    yref="paper",
                    x=0.23,
                    y=0.45,
                    text=rsi2,
                    showarrow=False,
                )
            ],
        )

        return go.Figure(data=[trace1, trace2], layout=layout)


def RSI(series, period):
    """
    Custom RSI function calculating relative strength indicator (RSI) instead
    of using TA-Lib. Heroku have a hard time import TA-Lib due to gcc
    compilation errors.

    Args:
        series: pd.Series. time series data to calculate RSI
        period: int. number of periods used to calculate RSI.

    Returns:    
        rsi: float. value of relative strength indicator.

    """
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    # first value is sum of avg gains
    u[u.index[period - 1]] = np.mean(u[:period])
    u = u.drop(u.index[: (period - 1)])
    # first value is sum of avg losses
    d[d.index[period - 1]] = np.mean(d[:period])
    d = d.drop(d.index[: (period - 1)])
    rs = (
        u.ewm(com=period - 1, adjust=False).mean()
        / d.ewm(com=period - 1, adjust=False).mean()
    )

    # rs = pd.stats.moments.ewma(u, com=period-1, adjust=False) / \
    # pd.stats.moments.ewma(d, com=period-1, adjust=False)
    return 100 - 100 / (1 + rs)
