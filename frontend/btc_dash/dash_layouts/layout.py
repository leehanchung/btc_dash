import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from btc_dash.dash_layouts import footer, navbar

layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        navbar,
        dbc.Container(id="page-content", className="mt-4", fluid=True),
        html.Hr(),
        footer,
    ]
)
