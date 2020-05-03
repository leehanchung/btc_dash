# BTCDash: A Bitcoin Price Prediction Dashboard
[![Python 3.7](https://img.shields.io/badge/python-3.7.4-blue.svg)](https://www.python.org/downloads/release/python-374/)
[![Flask 1.11](https://img.shields.io/badge/flask-1.1.1-blue.svg)](https://flask.palletsprojects.com/en/1.1.x/)
[![Dash 1.11](https://img.shields.io/badge/dash-1.11.0-blue.svg)](https://github.com/plotly/dash/)
[![Plotly 1.11](https://img.shields.io/badge/plotly-4.6.0-blue.svg)](https://github.com/plotly/plotly.py)
[![Tensorflow 2.2.0](https://img.shields.io/badge/tensorflow-2.2.0rc4-blue.svg)](https://github.com/tensorflow/tensorflow)
[![pandas 1.0.3](https://img.shields.io/badge/pandas-1.0.3-blue.svg)](https://github.com/pandas-dev/pandas)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
[![CodeFactor](https://www.codefactor.io/repository/github/leehanchung/btc_dash/badge)](https://www.codefactor.io/repository/github/leehanchung/btc_dash)


[**Website**](https://dry-shore-97069.herokuapp.com/)

*BTCDash* is a Bitcoin price prediction dashboard showcasing pseudo-real time prediction of the next period Bitcoin prices denominated in USD. 

Included on the dashboard are a 50 day BTCUSD OHLC chart with predicted price overlay line, a momentum gauge that displays the fear and greed using Relative Strength Indicator, and directional prediction accuracy over the last 30 periods.ically construct the search spaces for the hyperparameters.


## Key Features

BTCDash includes a real time pseudo live prediction update chart as follows:

- BTCUSD OHLC chart with predicted price overlay
- Momentum gauge using Relative Strength Indicator
- Directional prediction confusion metrics

![alt text](btc_dash/assets/btcdash1.gif)

## Statistics

We establish our baseline using the previous period log returns for RMSE and directional accurcy calculation. 

| Baseline | RMSE | Directional Accuracy |
| ------------- |-------------:| -----:|
| Baseline      | 0.056 | 0.4884 |
| Univariate ARIMA      | 0.051 |   0.5125 |

## TODO
- Hooking Timescale dB for historical data and prediction data.
- Walkforward performance statistics using vectorized backtester.
- More models, including fourier decomposition/FFT, multivariate ARIMA, LSTM, and WaveNet.
- Fix prediction plot time skipping/wrapping problem.
