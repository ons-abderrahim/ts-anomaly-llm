"""
Plotly Dash application entry point.
Run: python src/dashboard/app.py
"""

import dash
import dash_bootstrap_components as dbc

from src.dashboard.layout import build_layout
import src.dashboard.callbacks  # noqa: F401 — registers all @callback decorators

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&display=swap",
    ],
    title="Anomaly Monitor",
    update_title=None,
)

app.layout = build_layout()
server = app.server  # expose Flask server for Gunicorn/uWSGI

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
