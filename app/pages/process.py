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
        
            ## Process
			
			At this superhuman BTCUSD Price prediction tool, we are using deep learning,* artificial intelligence,** and methods inspired by quantum mechanics*** to make an optimized price prediction.
			
			
			* ARIMA(3,1,0). 
			** also ARIMA(3,1,0)
			*** again, ARIMA(3,1,0) Why (3,1,0)? Because we like it.
            """
        ),

    ],
)

layout = dbc.Row([column1])