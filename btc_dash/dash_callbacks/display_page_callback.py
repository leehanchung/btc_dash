from dash.dependencies import Input, Output
from btc_dash.dash_layouts import index, process


def register_display_pages_callback(app):

    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def display_page(pathname):
        if pathname == "/":
            return index.layout
        elif pathname == "/about":
            return process.layout
