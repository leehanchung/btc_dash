import os, sys
import pathlib
import numpy as np
import pandas as pd
import datetime as dt
# heroku doesnt like talib, so manual written rsi function.
#import talib

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff

from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from db.api import get_ohlcv_data

from sklearn.metrics import confusion_matrix
from statsmodels.tsa.arima_model import ARIMA

from app import app

# set interval at 5000ms, or 5s. need 5s for everything to render.
# 8/1/19: prediction line skipping back and forth different time periods.
#         change to 10s to give heroku ample time for compute
GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 13000)
app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}

"""
pred output df. Ideally this should go to some dB instead of being on this app.
"""
df_pred = pd.DataFrame(columns=['pred_log_ret', 'pred_Close'])


"""
Layout. One rows and two columns. First column is 8 width and contains the OHLC 
chart. Second column is 4 width and contains prediction RMSE bar chart and 
directional accuracy stacked vertically. The whole layout is sqashed between 
navbar and footer rows as defined in run.py in the parent directory. 

All charts have a corresponding plotting function and callback. Callback is 
invoked at every interval as defined in GRAPH_INTERVAL. Interval will invoke 
callback to plot OHLC chart, which will in term invoke callback to plot the other 
charts.
"""
column1 = dbc.Col(
    [
        # OHLC Chart
		html.Div(
			[html.H6("BTCUSD ($) 50 Day Rolling Chart", className="graph__title")]
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
					[
						html.H6(
							"MOMENTUM GAUGE", className="graph__title"
						)
					]
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

layout = dbc.Row([column1, column2])


"""
Plotting functions with callbacks on every interval ticks.
"""
@app.callback(
    Output("btcusd-ohlcv", "figure"), [Input("btcusd-ohlcv-update", "n_intervals")]
)
def gen_ohlcv(interval):
	"""
	Generate OHLCV Chart for BTCUSD with predicted price overlay.
	
	:params interval: update the graph based on an interval
	
	"""
	# hack to wrap interval around available data.  OOS starts at 1500, df has a 
	# total of 2274 rows after processing to wrap around 2274-1500 ~ 750. Reset
	# prediction data to empty df.
	interval = interval % 750
	
	# read data from source
	df = get_ohlcv_data(interval - 100, interval)
	df['log_ret'] = np.log(df.Close) - np.log(df.Close.shift(1))
	
	model = ARIMA(df.tail(60)["log_ret"], order=(3,1,0), freq='D').fit(disp=0)
	pred = model.forecast()[0] 
	df_pred.loc[df.tail(1).index[0]+pd.Timedelta('1 day')] = [pred[0], (np.exp(pred)*df.tail(1).Close.values)[0]]
	
	# plotting ohlc candlestick
	trace_ohlc = go.Candlestick(
		x=df.tail(50).index,
		open=df['Open'].tail(50), 
		close=df['Close'].tail(50), 
		high=df['High'].tail(50), 
		low=df['Low'].tail(50), 
		opacity=0.5,
		hoverinfo="skip",
		name="BTCUSD",
	)
	
	# plotting prediction line
	trace_line = go.Scatter(
		x = df_pred.tail(50).index,
		y = df_pred.pred_Close.tail(50),
		line_color='yellow',
		mode="lines+markers",
		name="Predicted Close"
	)

	layout = go.Layout(
		plot_bgcolor=app_color["graph_bg"],
		paper_bgcolor=app_color["graph_bg"],
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
			"gridcolor": app_color["graph_line"],
			"title": "Price (USD$)"
		},
	)
	
	return go.Figure(data=[trace_ohlc, trace_line], layout=layout)


def RSI(series, period):
	"""
	Custom RSI function calculating relative strength indicator (RSI) instead of using 
	TA-Lib. Heroku have a hard time import TA-Lib due to gcc compilation errors.
	
	Parameters:
	===========
	series: pd.Series. time series data to calculate RSI
	period: int. number of periods used to calculate RSI.
	
	
	Output:
	==========
	rsi: float. value of relative strength indicator.
	"""
	delta = series.diff().dropna()
	u = delta * 0
	d = u.copy()
	u[delta > 0] = delta[delta > 0]
	d[delta < 0] = -delta[delta < 0]
	u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
	u = u.drop(u.index[:(period-1)])
	d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
	d = d.drop(d.index[:(period-1)])
	rs = u.ewm(com=period-1, adjust=False).mean() / \
		 d.ewm(com=period-1, adjust=False).mean()
	
	#rs = pd.stats.moments.ewma(u, com=period-1, adjust=False) / \
	#pd.stats.moments.ewma(d, com=period-1, adjust=False)
	return 100 - 100 / (1 + rs)

@app.callback(
    Output("momentum-gauge", "figure"), [Input("btcusd-ohlcv-update", "n_intervals")]
)
def gen_momentum_gauge(interval):
	"""
	Generate 5 period lag RSI on BTCUSD Close and plot it as Momentum Gauge

	Parameters:
	===========
	interval: integer. update the graph based on an interval
	
	Output: 
	===========
	Plotly graph object figure.
	"""
	
	# hack to wrap interval around available data.  OOS starts at 1500, df has a 
	# total of 2274 rows after processing to wrap around 2274-1500 ~ 750.
	interval = interval % 750

	# read data from source and calculate RSI.  RSI ranges between 0 and 100.
	df = get_ohlcv_data(interval - 6, interval)
	#rsi = int(round(talib.RSI(df.Close.values, 5)[-1]))
	#print(rsi)
	rsi2 = int(round(RSI(df.Close, 5)[-1]))
	#print(rsi2)
	
	# Let's subdivide RSI into 10s to reduce plotting dial triangle complexity
	angle = round(rsi2, -1)
	# center of dial coordinate is 0.24 0.5. We plot left top and right coordinates of a triangle
	dials_dict = { 0: 'M 0.24 0.4950 L 0.09 0.5 L 0.24 0.505 Z',
				  10: 'M 0.2384 0.4952 L 0.0973 0.5463 L 0.2415 0.5047 Z',
				  20: 'M 0.2370 0.4959 L 0.1186 0.5881 L 0.2429 0.5040 Z',
				  30: 'M 0.2359 0.4970 L 0.1518 0.6213 L 0.2440 0.5029 Z',
				  40: 'M 0.2352 0.4985 L 0.1936 0.6247 L 0.2447 0.5015 Z',
				  50: 'M 0.235 0.5 L 0.24 0.65 L 0.245 0.5 Z', # confirmed)
				  60: 'M 0.2352 0.5015 L 0.2863 0.6426 L 0.2447 0.4984 Z',
				  70: 'M 0.2359 0.5029 L 0.3281 0.6213 L 0.244 0.497 Z',
				  80: 'M 0.2370 0.5040 L 0.3613 0.5881 L 0.2429 0.4959 Z',
				  90: 'M 0.2384 0.5047 L 0.3826 0.5463 L 0.2415 0.4952 Z',
				  100: 'M 0.24 0.505 L 0.39 0.50 L 0.24 0.495 Z',
	}
	
	# first we trace the dial using pie chart, hiding bottom half.
	trace1 = go.Pie(
		values=[50, 10, 10, 10, 10, 10],
		labels=["RSI Index", "HODL", "HELP", "MEH", "NICE", "FOMO"],
		domain={"x": [0, .48]},
		marker_colors=[
				app_color["graph_bg"],
				'rgb(232,226,202)',
				'rgb(226,210,172)',
				'rgb(223,189,139)',
				'rgb(223,162,103)',
				'rgb(226,126,64)'
			],
		name="Gauge",
		hole=.3,
		direction="clockwise",
		rotation=90,
		showlegend=False,
		hoverinfo="none",
		textinfo="label",
		textposition="inside"
	)

	# then we add numerical labels to the same pie chart
	trace2 = go.Pie(
		values=[40, 10, 10, 10, 10, 10, 10],
		labels=[".", "0", "20", "40", "60", "80", "100"],
		domain={"x": [0, .48]},
		marker_colors=['rgba(255, 255, 255, 0)']*7,
		hole=.4,
		direction="clockwise",
		rotation=108,
		showlegend=False,
		hoverinfo="none",
		textinfo="label",
		textposition="outside"
	)

	layout = go.Layout(
		height = 350,
		plot_bgcolor=app_color["graph_bg"],
		paper_bgcolor=app_color["graph_bg"],
		font={"color": "#fff"},
		autosize=True,
		margin=dict(l=200, autoexpand=True),
		xaxis=dict(
			showticklabels=False,
			showgrid=False,
			zeroline=False,
		),
		yaxis=dict(
			showticklabels=False,
			showgrid=False,
			zeroline=False,
		),
		# this is the hand/triangle on the dial.
		# https://plot.ly/python/gauge-charts/#dial center is 0.24, 0.5.
		# ^ and the above coordinate is not exactly correct so the angles
		# and magnitutdes are off.
		shapes=[dict(
					type='path',
					path=dials_dict[angle],
					fillcolor='rgba(44, 160, 101, 0.5)',
					line_width=1,
					xref='paper',
					yref='paper')
		],
		annotations=[
			dict(xref='paper',
				 yref='paper',
				 x=0.23,
				 y=0.45,
				 text=rsi2,
				 showarrow=False
			)
		]
	)

	return go.Figure(data=[trace1, trace2], layout=layout)


@app.callback(
    Output("confusion-matrix", "figure"),
    [Input("btcusd-ohlcv-update", "n_intervals")],
    [
        State("btcusd-ohlcv", "figure"),
    ],
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
	df['log_ret'] = np.log(df.Close) - np.log(df.Close.shift(1))
	
	if df_pred.shape[0] < 30:
		p = df_pred.shape[0]
		cm = confusion_matrix(np.sign(df.log_ret.tail(p).values), 
							  np.sign(df_pred.pred_log_ret.tail(p).values))
		#print(len(cm))
		if len(cm) == 0 or len(cm) == 1:
			cm = [[1, 1], [1, 1]]
		
		cm = np.array(cm)/p
	else:
		cm = confusion_matrix(np.sign(df.log_ret.tail(30).values),
							  np.sign(df_pred.pred_log_ret.tail(30).values))
		cm = np.array(cm)/30
							  
	
	cm_text = np.around(cm, decimals=2)
											 
	data = go.Heatmap(
			z = cm,
			x = ["Predicted Down", "Predicted Up"],
			y = ["Actual Up", "Actual Down"],
			zmin=0.,
			zmax=1.,
			
			opacity=0.8,
	)
	
	layout = go.Layout(
		height=350,
		plot_bgcolor=app_color["graph_bg"],
		paper_bgcolor=app_color["graph_bg"],
		font={"color": "#fff"},
		autosize=True,
		hovermode="closest",
		legend={
			#"orientation": "h",
			#"yanchor": "bottom",
			#"xanchor": "center",
			#"y": 1,
			#"x": 0.5,
		},
	)
	
	return go.Figure(data=data, layout=layout)