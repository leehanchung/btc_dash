import dash_bootstrap_components as dbc
import dash_html_components as html


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
                className="h1",
            )
        )
    ),
    fluid=True,
)
