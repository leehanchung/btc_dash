import dash_bootstrap_components as dbc
import dash_core_components as dcc

navbar = dbc.NavbarSimple(
    brand="BTCUSD Predictor",
    brand_href="/",
    children=[
        dbc.NavItem(
            dcc.Link(
                "About",
                href="/about",
                className="nav-link",
            )
        ),
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
                        dbc.Col(
                            html.Img(
                                src="../assets/Bitcoin_logo-700x155.png",
                                height="30px"
                            ),
                        ),
                        dbc.Col(
                            dbc.NavbarBrand(
                                "Dash",
                                className="m1-2",
                            ),
                        ),
                    ],
                    #align="center",
                    no_gutters=True,
                ),
                href='/',
            ),
            dbc.Nav(
                dbc.NavItem(
                    dcc.Link(
                        'About',
                        href='/about',
                        className='nav-link',
                    ),
                ),
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
