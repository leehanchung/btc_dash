from dash import Dash
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from btc_dash.dash_layouts import dashboard, process


def register_display_pages_callback(app: Dash):
    """Wrapper function for registering callback to route app to root
    or about page using Dash.

    Args:
        dash app object

    Returns:
        None

    """
    @app.callback(
        Output("page-content", "children"), [Input("url", "pathname")]
    )
    def display_page(pathname: str) -> dbc.Row:
        """
        Genererate confusion matrix of prediction directions.

        Args:
            interval: upadte the graph based on an interval
            ohlcv_figure: current ohlcv chart, not used. LOL.

        Returns:
            dbc.Row body for about and root page.
        """
        if pathname == "/":
            return dashboard
        elif pathname == "/about":
            return process
