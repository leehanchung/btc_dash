import dash
import dash_bootstrap_components as dbc

from btc_dash import server

print("dash_app")
external_stylesheets = [
    # Bootswatch theme
    dbc.themes.BOOTSTRAP,
    # for social media icons
    "https://use.fontawesome.com/releases/v5.9.0/css/all.css",
]


meta_tags = [
    {
        "name": "description",
        "content": ("BTCUSD Prediction Dashboard with real time inferencing "
                    "on 5 minute delayed data, with momentum gauges and "
                    "prediction confusion metrix"),
    },
    {
        "name": "viewport",
        "content": "width=device-width, initial-scale=1.0",
    }
]


app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=external_stylesheets,
    meta_tags=meta_tags,
    routes_pathname_prefix='/',
)

app.config.suppress_callback_exceptions = True
app.title = "BTCUSD Forecast"
print("dash_app end")

# Imports from 3rd party libraries
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Imports from this application
# from app import app, server
# from dash_app import app
from btc_dash.dash_layouts import index, process

navbar = dbc.NavbarSimple(
    brand="BTCUSD Predictor",
    brand_href="/",
    children=[
        dbc.NavItem(dcc.Link("About", href="/about", className="nav-link")),
    ],
    sticky="top",
    color="#082255",
    dark=True,
    fluid=True,
    className="h1",
)
"""

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="../assets/Bitcoin_logo-700x155.png", height="30px")),
                        dbc.Col(dbc.NavbarBrand("Dash", className="m1-2")),
                    ],
                    #align="center",
                    no_gutters=True,
                ),
                href='/',
            ),			
            dbc.Nav(
                dbc.NavItem(dcc.Link('About', href='/about', className='nav-link')),
                navbar=True,
            ),
        ]
    ),
    expand="lg",
    dark=True,
    sticky='top',
    color="#082255",
    className='h1',
)
"""

footer = dbc.Container(
    dbc.Row(
        dbc.Col(
            html.P(
                [
                    html.Span("Hanchung Lee      ", className="mr-2"),
                    html.A(
                        html.I(className="fas fa-envelope-square mr-3"),
                        href="mailto:lee.hanchung@gmail.com",
                    ),
                    html.A(
                        html.I(className="fab fa-github-square mr-3"),
                        href="https://github.com/leehanchung/btc_dash",
                    ),
                    html.A(
                        html.I(className="fab fa-linkedin mr-3"),
                        href="https://www.linkedin.com/in/hanchunglee/",
                    ),
                    html.A(
                        html.I(className="fab fa-twitter-square mr-3"),
                        href="https://twitter.com/hanchunglee",
                    ),
                ],
                className="h1",  #'lead'
            )
        )
    ),
    fluid=True,
)

# For more explanation, see:
# Plotly Dash User Guide, URL Routing and Multiple Apps
# https://dash.plot.ly/urls
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        navbar,
        dbc.Container(id="page-content", className="mt-4", fluid=True),
        html.Hr(),
        footer,
    ]
)


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/":
        return index.layout
    elif pathname == "/about":
        return process.layout
    else:
        return dcc.Markdown("## Page not found")


# if __name__ == "__main__":
#     app.run_server(debug=True)
