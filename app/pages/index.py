import os, sys
import pathlib
import numpy as np
import pandas as pd
import datetime as dt

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
from db.api import get_wind_data_by_id, get_ohlcv_data, get_ohlcv_data_by_id

from sklearn.metrics import confusion_matrix
from statsmodels.tsa.arima_model import ARIMA

from app import app

# set interval at 5000ms, or 5s.
GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 5000)
app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}

# pred output df. Ideally this should go to some dB.
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
Plotting functions with callbacks.
"""
@app.callback(
    Output("btcusd-ohlcv", "figure"), [Input("btcusd-ohlcv-update", "n_intervals")]
)
def gen_ohlcv(interval):
	"""
	Generate OHLCV Chart for BTCUSD with predicted price overlay.
	
	:params interval: update the graph based on an interval
	
	"""
	print("\nStarting gen_ohlcv, interval {}...\n".format(interval))
	
	# read data from source
	df = get_ohlcv_data(interval - 100, interval)
	df['log_ret'] = np.log(df.Close) - np.log(df.Close.shift(1))
	
	model = ARIMA(df.tail(60)["log_ret"], order=(3,1,0), freq='D').fit(disp=0)
	pred = model.forecast()[0] # forecast() returns (forecast, standard error, ci), taking the first
	df_pred.loc[df.tail(1).index[0]+pd.Timedelta('1 day')] = [pred[0], (np.exp(pred)*df.tail(1).Close.values)[0]]
	
	# plotting ohlc candlestick
	trace_ohlc = go.Candlestick(
		x=df.index,
		open=df['Open'], 
		close=df['Close'], 
		high=df['High'], 
		low=df['Low'], 
		opacity=0.5,
		hoverinfo="skip",
		name="BTCUSD",
	)
	
	# plotting prediction line
	trace_line = go.Scatter(
		x = df_pred.index,
		y = df_pred.pred_Close,#y = np.exp(pred) * df.tail
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
	print("returning gen_ohlcv...")
	return go.Figure(data=[trace_ohlc, trace_line], layout=layout)


@app.callback(
    Output("momentum-gauge", "figure"), [Input("btcusd-ohlcv-update", "n_intervals")]
)
def gen_momentum_gauge(interval):
	"""
	Generate BTCUSD Momentum Gauge.

	:params interval: update the graph based on an interval
	"""

	now = dt.datetime.now()
	total_time = (now.hour * 3600) + (now.minute * 60) + (now.second)

	df = get_wind_data_by_id(total_time)
	val = df["Speed"].iloc[-1]
	direction = [0, (df["Direction"][0] - 20), (df["Direction"][0] + 20), 0]

	traces_scatterpolar = [
		{"r": [0, val, val, 0], "fillcolor": "#084E8A"},
		{"r": [0, val * 0.65, val * 0.65, 0], "fillcolor": "#B4E1FA"},
		{"r": [0, val * 0.3, val * 0.3, 0], "fillcolor": "#EBF5FA"},
	]

	data = [
		go.Scatterpolar(
			r=traces["r"],
			theta=direction,
			mode="lines",
			fill="toself",
			fillcolor=traces["fillcolor"],
			line={"color": "rgba(32, 32, 32, .6)", "width": 1},
		)
		for traces in traces_scatterpolar
	]

	layout = go.Layout(
		height=350,
		plot_bgcolor=app_color["graph_bg"],
		paper_bgcolor=app_color["graph_bg"],
		font={"color": "#fff"},
		autosize=False,
		polar={
			"bgcolor": app_color["graph_line"],
			"radialaxis": {"range": [0, 45], "angle": 45, "dtick": 10},
			"angularaxis": {"showline": False, "tickcolor": "white"},
		},
		showlegend=False,
	)

	return go.Figure(data=data, layout=layout)


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
	print("Starting gen_confusion_matrix...")
	df = get_ohlcv_data(interval - 100, interval)
	df['log_ret'] = np.log(df.Close) - np.log(df.Close.shift(1))
	#print(df.log_ret.tail(30))
	#print(df_pred.shape)#, df_pred.tail(30))
	
	if df_pred.shape[0] < 30:
		p = df_pred.shape[0]
		#print(p, df.log_ret.tail(p).shape)
		#print(df_pred.pred_log_ret.shape)
		cm = confusion_matrix(np.sign(df.log_ret.tail(p).values), 
							  np.sign(df_pred.pred_log_ret.tail(p).values))
		print(len(cm))
		if len(cm) == 0 or len(cm) == 1:
			cm = [[1, 1], [1, 1]]
		#if cm == (list([]) or list([[1]]) or list([[0]])):
		#	cm = [[0, 0], [0, 0]]
		cm = np.array(cm)/p
	else:
		cm = confusion_matrix(np.sign(df.log_ret.tail(30).values),
							  np.sign(df_pred.pred_log_ret.tail(30).values))
		cm = np.array(cm)/30
							  
	print(cm)
	cm_text = np.around(cm, decimals=2)
	#x = ["Predicted Down", "Predicted Up"],
	#y = ["Actual Up", "Actual Down"],
			
	#data = ff.create_annotated_heatmap(cm, x=x, y=y, annotation_text=cm_text)
	#annotations = go.Annotations()
	#for n, row in enumerate(cm):
		#for m, val in enumerate(row):
			#annotations.append(go.Annotation(text=str(cm[n][m]), x=x[m], y=y[n],
											 #xref='x1', yref='y1', showarrow=False))
											 
	data = go.Heatmap(
			z = cm,
			#z=[[1, 20 ],
			#	[20, 1]],
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
		#annotations=annotations,
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
	#data.layout.update(layout)
	print("Returning confusion matrix...")
	return go.Figure(data=data, layout=layout)