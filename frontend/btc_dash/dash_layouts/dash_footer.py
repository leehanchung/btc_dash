import dash_bootstrap_components as dbc
import dash_html_components as html

footer = dbc.Container(
    dbc.Row(
        dbc.Col(
            html.P(
                [
                    html.Span("Hanchung Lee      ", className="mr-2"),
                    html.A(
                        html.I(className="fab fa-github-square mr-3"),
                        href="https://github.com/leehanchung/btc_dash",
                    ),
                ],
                className="h1",
            )
        )
    ),
    fluid=True,
)
