"""
Plotly Dash dashboard layout for real-time anomaly monitoring.
Defines the component tree; all dynamic wiring is in callbacks.py.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc


REFRESH_INTERVAL_MS = 5_000   # poll the API every 5 seconds


def build_layout() -> html.Div:
    """Return the full Dash layout tree."""
    return html.Div(
        style={"fontFamily": "'IBM Plex Mono', monospace", "backgroundColor": "#0d1117", "minHeight": "100vh"},
        children=[
            # ── Header ──────────────────────────────────────────────────
            dbc.Navbar(
                dbc.Container([
                    html.Span("🔍 Anomaly Monitor", style={
                        "color": "#58a6ff", "fontSize": "1.3rem", "fontWeight": "700",
                    }),
                    html.Span("powered by LLM explanations", style={
                        "color": "#8b949e", "fontSize": "0.8rem", "marginLeft": "12px",
                    }),
                ]),
                color="#161b22",
                dark=True,
                style={"borderBottom": "1px solid #30363d"},
            ),

            # ── Main content ─────────────────────────────────────────────
            dbc.Container(fluid=True, style={"padding": "24px"}, children=[

                # ── Control row ──────────────────────────────────────────
                dbc.Row(style={"marginBottom": "20px"}, children=[
                    dbc.Col(width=3, children=[
                        html.Label("Data Source", style={"color": "#8b949e", "fontSize": "0.75rem"}),
                        dcc.Dropdown(
                            id="source-dropdown",
                            options=[
                                {"label": "Kafka Stream", "value": "kafka"},
                                {"label": "Redis Stream", "value": "redis"},
                                {"label": "Demo (synthetic)", "value": "demo"},
                            ],
                            value="demo",
                            style={"backgroundColor": "#161b22", "color": "#c9d1d9"},
                        ),
                    ]),
                    dbc.Col(width=3, children=[
                        html.Label("Detector Model", style={"color": "#8b949e", "fontSize": "0.75rem"}),
                        dcc.Dropdown(
                            id="model-dropdown",
                            options=[
                                {"label": "Isolation Forest", "value": "isolation_forest"},
                                {"label": "LSTM Autoencoder", "value": "lstm_ae"},
                                {"label": "Transformer AE", "value": "transformer_ae"},
                            ],
                            value="isolation_forest",
                            style={"backgroundColor": "#161b22", "color": "#c9d1d9"},
                        ),
                    ]),
                    dbc.Col(width=3, children=[
                        html.Label("Threshold", style={"color": "#8b949e", "fontSize": "0.75rem"}),
                        dcc.Slider(
                            id="threshold-slider",
                            min=0.1, max=0.9, step=0.05, value=0.5,
                            marks={v / 10: str(v / 10) for v in range(1, 10, 2)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ]),
                    dbc.Col(width=3, children=[
                        html.Div(style={"paddingTop": "22px"}, children=[
                            dbc.Button("▶ Start", id="start-btn", color="success", size="sm",
                                       style={"marginRight": "8px"}),
                            dbc.Button("⏹ Stop", id="stop-btn", color="danger", size="sm"),
                        ])
                    ]),
                ]),

                # ── KPI cards ────────────────────────────────────────────
                dbc.Row(style={"marginBottom": "20px"}, children=[
                    _kpi_card("total-points-card", "Points Ingested", "0", "#58a6ff"),
                    _kpi_card("anomaly-count-card", "Anomalies Detected", "0", "#f85149"),
                    _kpi_card("anomaly-rate-card", "Anomaly Rate", "0%", "#d29922"),
                    _kpi_card("last-score-card", "Last Anomaly Score", "—", "#3fb950"),
                ]),

                # ── Time series chart ─────────────────────────────────────
                dbc.Row(style={"marginBottom": "20px"}, children=[
                    dbc.Col(width=12, children=[
                        dbc.Card(style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}, children=[
                            dbc.CardHeader("Live Time Series", style={"color": "#8b949e", "fontSize": "0.8rem"}),
                            dbc.CardBody([
                                dcc.Graph(
                                    id="ts-chart",
                                    config={"displayModeBar": False},
                                    style={"height": "320px"},
                                ),
                            ]),
                        ]),
                    ]),
                ]),

                # ── Anomaly table + explanation panel ──────────────────────
                dbc.Row(children=[
                    # Anomaly log
                    dbc.Col(width=5, children=[
                        dbc.Card(style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}, children=[
                            dbc.CardHeader("Anomaly Log", style={"color": "#8b949e", "fontSize": "0.8rem"}),
                            dbc.CardBody([
                                html.Div(id="anomaly-table", style={"maxHeight": "400px", "overflowY": "auto"}),
                            ]),
                        ]),
                    ]),

                    # LLM explanation panel
                    dbc.Col(width=7, children=[
                        dbc.Card(style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}, children=[
                            dbc.CardHeader("LLM Root-Cause Explanation", style={"color": "#8b949e", "fontSize": "0.8rem"}),
                            dbc.CardBody([
                                html.Div(
                                    id="explanation-panel",
                                    style={
                                        "color": "#c9d1d9",
                                        "fontSize": "0.9rem",
                                        "lineHeight": "1.7",
                                        "padding": "8px",
                                        "minHeight": "120px",
                                    },
                                    children="Select an anomaly row to see its explanation.",
                                ),
                                html.Hr(style={"borderColor": "#30363d"}),
                                html.Div(id="action-chips", style={"display": "flex", "flexWrap": "wrap", "gap": "8px"}),
                            ]),
                        ]),
                    ]),
                ]),
            ]),

            # ── Hidden state + intervals ─────────────────────────────────
            dcc.Store(id="series-store", data={"values": [], "timestamps": [], "anomalies": []}),
            dcc.Store(id="selected-anomaly-store"),
            dcc.Interval(id="refresh-interval", interval=REFRESH_INTERVAL_MS, n_intervals=0, disabled=True),
        ],
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _kpi_card(card_id: str, label: str, default: str, accent: str) -> dbc.Col:
    return dbc.Col(width=3, children=[
        dbc.Card(
            style={
                "backgroundColor": "#161b22",
                "border": f"1px solid {accent}33",
                "borderLeft": f"3px solid {accent}",
            },
            children=dbc.CardBody([
                html.P(label, style={"color": "#8b949e", "fontSize": "0.72rem", "marginBottom": "4px"}),
                html.H4(default, id=card_id, style={"color": accent, "fontWeight": "700", "margin": 0}),
            ]),
        ),
    ])
