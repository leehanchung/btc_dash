# BTCDash: A Bitcoin Price Prediction Dashboard
[![Python 3.7.4](https://img.shields.io/badge/python-3.7.4-blue.svg)](https://www.python.org/downloads/release/python-374/)
[![Flask 2.00](https://img.shields.io/badge/flask-2.0.0-blue.svg)](https://flask.palletsprojects.com/en/1.1.x/)
[![Dash 1.20](https://img.shields.io/badge/dash-1.20.0-blue.svg)](https://github.com/plotly/dash/)
[![Plotly 4.14](https://img.shields.io/badge/plotly-4.14.3-blue.svg)](https://github.com/plotly/plotly.py)
[![Tensorflow 2.5.0](https://img.shields.io/badge/tensorflow-2.5.0-blue.svg)](https://github.com/tensorflow/tensorflow)
[![pandas 1.2.4](https://img.shields.io/badge/pandas-1.2.4-blue.svg)](https://github.com/pandas-dev/pandas)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
[![CodeFactor](https://www.codefactor.io/repository/github/leehanchung/btc_dash/badge)](https://www.codefactor.io/repository/github/leehanchung/btc_dash)
[![Coverage Status](https://coveralls.io/repos/github/leehanchung/btc_dash/badge.svg?branch=master)](https://coveralls.io/github/leehanchung/btc_dash?branch=master)

[**Website**](https://dry-shore-97069.herokuapp.com/)

*BTCDash* is a Bitcoin price prediction dashboard showcasing pseudo-real time prediction of the next period Bitcoin prices denominated in USD. 

Included on the dashboard are a 50 day BTCUSD OHLC chart with predicted price overlay line, a momentum gauge that displays the fear and greed using Relative Strength Indicator, and directional prediction accuracy over the last 30 periods. Data gathered from Bitstamp.


## Key Features

BTCDash includes a real time pseudo live prediction update chart as follows:

- BTCUSD OHLC chart with predicted price overlay
- Momentum gauge using Relative Strength Indicator
- Directional prediction confusion metrics

![demo](frontend/btc_dash/assets/btcdash1.gif)

## Infrastructure
:construction:

![infra](docs/btc_dash.png)

## Statistics

We establish our baseline using the previous period log returns for RMSE and directional accurcy calculation. 

| Baseline | RMSE | Directional Accuracy |
| ------------- |-------------:| -----:|
| Baseline      | 0.0597 | 0.4724 |
| Univariate AR(16)      | 0.0340 | 0.5333 |
| Univariate ARIMA(5, 1, 1)      | 0.0268 |   0.7143 |
| Univariate LSTM(diff=1)      | 0.0285 |   0.8000 |
