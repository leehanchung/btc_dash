import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app

column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
# Introduction
--------------
> "Prediction is very difficult, especially if it's about the future."

>    -- Nils Bohr, Nobel laureate in Physics

For this awesome BTCUSD Price prediction app, we are using deep learning[1], artificial intelligence[2], and methods inspired by quantum mechanics[3] to make an optimized price prediction. No tea leaves were hurt during the making.



# Preprocessing
---------------
We dotted the i's and crossed the t's. We observe the Close price chart and cannot see obvious trend or seasonality. Thus we cannot extract away trend and seasonality to model the stationary white noise.

![Alt](../assets/chart.png "ACF1DIFF")

Now, we run an augmented Dicky-Fuller (ADF) test on this time series data to test if the data is statistically stationary. A stationary time series is one whose statistical properties such as mean, variance, and autocorrelation, etc, are constant over time. Most statistical forecasting methods are based on the assumption that the time series is can be rendered approximately stationary.[4] We will plot the autocorrelation function first and then conduct ADF tests. Here's what we got from the closing price that we try to predict. The null hypothesis is:

**𝐻0: BTCUSD Close price has an unit root, has a time-dependent structure and thus non-stationary.**

![Alt](../assets/acf.png "ACF")
 
The auto-correlation function looks highly dependent on its lags, e.g., today's close is dependent on yesterdays close, etc. An ADF test produces a p-value of 0.97, in other words, we cannot reject H0, meaning that the BTCUSD Price is not stationary. Thus, we take a log of Close price and calculate the first order difference, which by the way, is the log return of Close price. It will also transform the Close price to log return space. We then run through the same process again and here are the results.
 
![Alt](../assets/acf1diff.png "ACF1DIFF")

After taking the log return, autocorrelation chart looks like the current price is barely dependent on its lags. ADF test produces a p-value of 0.00, meaning we can reject H0. In other words, the time series is now stationary for us to model and forecast.



# Training
---------------
We used two metrices for our modeling - root mean squared error (RMSE) and accuracy. While mean-absolute-percentage-error works for some time series, it is not applicable here as we do not use percentage returns but log returns. We also cared a lot more about the direction of returns instead of magnitude of returns. A trade placed based on the prediction that the price to go up tomorrow will be fine if the magnitude is off but will be unprofitable if the direction is wrong. Our baseline using one day lagged Close price as predictor has a RMSE of 0.056 and accuracy of 0.4884. Yesterday's return is unsurprising a great predictor for today's return, but has a poor directional accuracy.

We ran through simple autoregression model, and searched through ARIMA model and settled in ARIMA(3,1,0).  However, we cant simply fit an ARIMA model and use it to forecast future N periods indefinitely. For example, here's our simple ARIMA forecast for all future periods, with RMSE of 0.0378 and directional accuracy of 0.565.

![Alt](../assets/arima_bad.png "ARIMA")

After several periods, it basically resort to using the last log_return as the next log_return. To combat this issue, we have to refit the model every N periods base off the last M period lag.  We choose 8 and 60. And here's the result.

![Alt](../assets/rolling_arima_good.png "ROLLARIMA")

The RMSE is 0.0512 but the directional accuraqcy is 0.5125, both are improvements to our baseline, and more importantly, it's closer to real life situations where models are continuously updated as new data comes in.



# Next Step
---------------
More features and Bigger models. Because we know everything becomes better with bigger models.

![Alt](https://imgs.xkcd.com/comics/machine_learning.png "XKCD")


# DISCLAIMER
---------------
The content of this website is for entertainment purposes only.

[1] ARIMA(3,1,0)

[2] also ARIMA(3,1,0)

[3] again, ARIMA(3,1,0)

[4] https://people.duke.edu/~rnau/411diff.htm

            """
        ),
    ],
    className="body",
)

layout = dbc.Row([column1])
