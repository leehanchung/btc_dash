import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from btc_dash import config

# set interval at 5000ms, or 5s. need 5s for everything to render.
# 8/1/19: prediction line skipping back and forth different time periods.
#         change to 10s to give heroku ample time for compute. at 13s heroku
#         still slows down dramaticall causing lines to jump back and forth.
GRAPH_INTERVAL = config.GRAPH_INTERVAL
app_color = config.app_color

"""
pred output df. Ideally this should go to some dB instead of being on this app.
"""
# df_pred = pd.DataFrame(columns=["pred_log_ret", "pred_Close"])


"""
Layout. One rows and two columns. First column is 8 width and contains the OHLC
chart. Second column is 4 width and contains prediction RMSE bar chart and
directional accuracy stacked vertically. The whole layout is sqashed between
navbar and footer rows as defined in run.py in the parent directory.

All charts have a corresponding plotting function and callback. Callback is
invoked at every interval as defined in GRAPH_INTERVAL. Interval will invoke
callback to plot OHLC chart, which will in term invoke callback to plot the
other charts.
"""
column1 = dbc.Col(
    [
        # OHLC Chart
        html.Div(
            [
                html.H6(
                    "BTCUSD ($) 50 Day Rolling Chart",
                    className="graph__title",
                ),
            ]
        ),
        dcc.Graph(
            id="btcusd-ohlcv",
            figure=go.Figure(
                layout=go.Layout(
                    plot_bgcolor=app_color["graph_bg"],
                    paper_bgcolor=app_color["graph_bg"],
                )
            ),
        ),
        dcc.Interval(
            id="btcusd-ohlcv-update",
            interval=int(GRAPH_INTERVAL),
            n_intervals=0,
        ),
    ],
    className="two-thirds column ohlcv__chart__container",
)

column2 = dbc.Col(
    [
        # BTCUSD Momentum Gauge
        html.Div(
            [
                html.Div(
                    [html.H6("MOMENTUM GAUGE", className="graph__title")],
                ),
                dcc.Graph(
                    id="momentum-gauge",
                    figure=go.Figure(
                        layout=go.Layout(
                            plot_bgcolor=app_color["graph_bg"],
                            paper_bgcolor=app_color["graph_bg"],
                        )
                    ),
                ),
            ],
            className="graph__container first",
        ),
        # Prediction Confusion Matrix
        html.Div(
            [
                html.Div(
                    [
                        html.H6(
                            "TRAILING 30 PREDICTION CONFUSION MATRIX",
                            className="graph__title",
                        )
                    ]
                ),
                dcc.Graph(
                    id="confusion-matrix",
                    figure=go.Figure(
                        layout=go.Layout(
                            plot_bgcolor=app_color["graph_bg"],
                            paper_bgcolor=app_color["graph_bg"],
                        )
                    ),
                ),
            ],
            className="graph__container second",
        ),
    ],
    width=4,
)

dashboard = dbc.Row([column1, column2])
