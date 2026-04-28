"""
Dash callbacks wiring UI controls → API → chart + explanation panel.
"""

from __future__ import annotations

import os
import random
import time
from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
import requests
from dash import Input, Output, State, callback, html, no_update
import dash_bootstrap_components as dbc

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")


# ---------------------------------------------------------------------------
# Interval enable / disable
# ---------------------------------------------------------------------------

@callback(
    Output("refresh-interval", "disabled"),
    Input("start-btn", "n_clicks"),
    Input("stop-btn", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_interval(start_clicks, stop_clicks):
    from dash import ctx
    if ctx.triggered_id == "start-btn":
        return False     # enable
    return True          # disable


# ---------------------------------------------------------------------------
# Data fetch + store update
# ---------------------------------------------------------------------------

@callback(
    Output("series-store", "data"),
    Input("refresh-interval", "n_intervals"),
    State("series-store", "data"),
    State("model-dropdown", "value"),
    State("threshold-slider", "value"),
    State("source-dropdown", "value"),
    prevent_initial_call=True,
)
def fetch_data(n_intervals, store, model, threshold, source):
    """
    In demo mode: generate synthetic data with occasional spikes.
    In kafka/redis mode: call the detect API with fresh data.
    """
    prev_values: list = store.get("values", [])
    prev_timestamps: list = store.get("timestamps", [])
    prev_anomalies: list = store.get("anomalies", [])

    # Generate new batch of points (demo mode)
    new_values = _generate_demo_batch(n_points=20)
    now = datetime.utcnow()
    new_timestamps = [
        (now + timedelta(seconds=i)).isoformat() + "Z"
        for i in range(len(new_values))
    ]

    # Combine with history (keep last 300 points)
    all_values = (prev_values + new_values)[-300:]
    all_timestamps = (prev_timestamps + new_timestamps)[-300:]

    if len(all_values) < 10:
        return store   # not enough data yet

    # Call detect API
    try:
        resp = requests.post(
            f"{API_BASE}/detect",
            json={
                "series": all_values,
                "timestamps": all_timestamps,
                "model": model,
                "threshold": threshold,
                "metadata": {"sensor_id": "demo-sensor", "unit": "units", "domain": "iot"},
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        detected = [pt for pt in data["anomalies"] if pt["is_anomaly"]]
    except Exception:
        detected = prev_anomalies   # keep stale on error

    return {"values": all_values, "timestamps": all_timestamps, "anomalies": detected}


# ---------------------------------------------------------------------------
# Chart update
# ---------------------------------------------------------------------------

@callback(
    Output("ts-chart", "figure"),
    Output("total-points-card", "children"),
    Output("anomaly-count-card", "children"),
    Output("anomaly-rate-card", "children"),
    Output("last-score-card", "children"),
    Input("series-store", "data"),
)
def update_chart(store):
    values = store.get("values", [])
    timestamps = store.get("timestamps", [])
    anomalies = store.get("anomalies", [])

    fig = go.Figure()

    # Base series
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=values,
        mode="lines",
        line={"color": "#58a6ff", "width": 1.5},
        name="Signal",
    ))

    # Anomaly markers
    anom_x = [a["timestamp"] for a in anomalies]
    anom_y = [a["value"] for a in anomalies]
    if anom_x:
        fig.add_trace(go.Scatter(
            x=anom_x,
            y=anom_y,
            mode="markers",
            marker={"color": "#f85149", "size": 10, "symbol": "x"},
            name="Anomaly",
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#c9d1d9", "family": "IBM Plex Mono"},
        xaxis={"gridcolor": "#21262d", "showgrid": True},
        yaxis={"gridcolor": "#21262d", "showgrid": True},
        legend={"bgcolor": "rgba(0,0,0,0)"},
        margin={"l": 40, "r": 20, "t": 10, "b": 40},
    )

    total = len(values)
    anom_count = len(anomalies)
    rate = f"{anom_count / total * 100:.1f}%" if total else "0%"
    last_score = f"{anomalies[-1]['score']:.3f}" if anomalies else "—"

    return fig, str(total), str(anom_count), rate, last_score


# ---------------------------------------------------------------------------
# Anomaly table
# ---------------------------------------------------------------------------

@callback(
    Output("anomaly-table", "children"),
    Input("series-store", "data"),
)
def update_table(store):
    anomalies = store.get("anomalies", [])
    if not anomalies:
        return html.P("No anomalies detected.", style={"color": "#8b949e", "fontSize": "0.85rem"})

    rows = []
    for a in anomalies[-20:][::-1]:    # last 20, newest first
        rows.append(
            html.Div(
                id={"type": "anomaly-row", "index": a["timestamp"]},
                style={
                    "padding": "8px 12px",
                    "marginBottom": "4px",
                    "borderRadius": "4px",
                    "backgroundColor": "#1c2128",
                    "border": "1px solid #30363d",
                    "cursor": "pointer",
                    "display": "flex",
                    "justifyContent": "space-between",
                    "fontSize": "0.78rem",
                    "color": "#c9d1d9",
                },
                children=[
                    html.Span(a["timestamp"][-8:]),
                    html.Span(f"{a['value']:.3f}", style={"color": "#f0883e"}),
                    html.Span(f"score {a['score']:.2f}", style={"color": "#f85149"}),
                ],
            )
        )
    return rows


# ---------------------------------------------------------------------------
# LLM Explanation panel
# ---------------------------------------------------------------------------

@callback(
    Output("explanation-panel", "children"),
    Output("action-chips", "children"),
    Input("series-store", "data"),
    prevent_initial_call=True,
)
def update_explanation(store):
    """Auto-explain the most recent anomaly."""
    anomalies = store.get("anomalies", [])
    values = store.get("values", [])

    if not anomalies or not values:
        return "No anomalies to explain yet.", []

    latest = anomalies[-1]

    try:
        resp = requests.post(
            f"{API_BASE}/explain",
            json={
                "anomaly": {
                    "timestamp": latest["timestamp"],
                    "value": latest["value"],
                    "score": latest["score"],
                },
                "context_window": values[-21:],
                "metadata": {
                    "sensor_id": "demo-sensor",
                    "unit": "units",
                    "location": "Demo Environment",
                    "domain": "iot",
                },
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()

        explanation_text = data.get("explanation", "No explanation returned.")
        actions = data.get("suggested_actions", [])

        chips = [
            dbc.Badge(
                action,
                color="secondary",
                style={"fontSize": "0.75rem", "padding": "5px 10px", "cursor": "default"},
            )
            for action in actions
        ]
        return explanation_text, chips

    except Exception as exc:
        return f"⚠ Could not fetch explanation: {exc}", []


# ---------------------------------------------------------------------------
# Demo data generator
# ---------------------------------------------------------------------------

def _generate_demo_batch(n_points: int = 20) -> list[float]:
    """Generate a batch of synthetic sensor readings with occasional spikes."""
    base = 1.0
    noise = np.random.normal(0, 0.1, n_points)
    values = (base + noise).tolist()
    # Inject a spike with 5% probability per batch
    if random.random() < 0.05:
        idx = random.randint(0, n_points - 1)
        values[idx] = base + random.uniform(5, 12)
    return [round(v, 4) for v in values]
