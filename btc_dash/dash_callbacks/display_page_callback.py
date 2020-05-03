from dash.dependencies import Input, Output
from btc_dash.dash_layouts import dashboard, process


def register_display_pages_callback(app):
    @app.callback(
        Output("page-content", "children"), [Input("url", "pathname")]
    )
    def display_page(pathname):
        if pathname == "/":
            return dashboard
        elif pathname == "/about":
            return process
