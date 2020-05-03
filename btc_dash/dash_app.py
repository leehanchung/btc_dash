import dash
import dash_bootstrap_components as dbc
from flask.app import Flask

# from btc_dash import server
from btc_dash.dash_callbacks import register_display_pages_callback
from btc_dash.dash_callbacks import register_confusion_callback
from btc_dash.dash_callbacks import register_ohlcv_callback
from btc_dash.dash_callbacks import register_momentum_callback
from btc_dash.dash_layouts import layout


def create_dash_app(*, flask_server: Flask) -> Flask:
    """Creates dash app that hooks to Flask server,  with stylesheets,
    meta tags, title, and all associated callbacks.

    Args:
        flask_server: flask.app.Flask object for initializing plotly dash.

    Returns:
        Flask with dash app hooked.

    """
    external_stylesheets = [
        # Bootswatch theme
        dbc.themes.BOOTSTRAP,
        # for social media icons
        "https://use.fontawesome.com/releases/v5.9.0/css/all.css",
    ]

    meta_tags = [
        {
            "name": "description",
            "content": (
                "BTCUSD Prediction Dashboard with real time inferencing "
                "on 5 minute delayed data, with momentum gauges and "
                "prediction confusion metrix"
            ),
        },
        {
            "name": "viewport",
         "content": "width=device-width, initial-scale=1.0",
        },
    ]

    app = dash.Dash(
        __name__,
        server=flask_server,
        external_stylesheets=external_stylesheets,
        meta_tags=meta_tags,
        routes_pathname_prefix="/",
    )

    app.config.suppress_callback_exceptions = True
    app.title = "BTCUSD Forecast"
    app.layout = layout

    register_momentum_callback(app)
    register_confusion_callback(app)
    register_ohlcv_callback(app)
    register_display_pages_callback(app)

    return app.server
