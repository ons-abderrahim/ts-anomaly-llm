# Architecture

## Overview

The system is composed of two cooperating layers running behind a FastAPI gateway, with a Plotly Dash front-end for human operators.

```
┌──────────────────────────────────────────────────────────────────┐
│                        Data Sources                              │
│   IoT sensors · financial feeds · server metrics · log streams  │
└───────────────────┬──────────────────────────────────────────────┘
                    │
        ┌───────────▼───────────┐
        │  Streaming Ingestion  │
        │  Kafka / Redis Streams│
        │  → sliding window buf │
        └───────────┬───────────┘
                    │ np.ndarray window
        ┌───────────▼───────────┐
        │   Layer 1 – Detector  │
        │                       │
        │  IsolationForest  ──┐ │
        │  LSTM-AE          ──┤ │  → AnomalyResult
        │  Transformer-AE   ──┘ │     scores[], is_anomaly[]
        └───────────┬───────────┘
                    │ AnomalyResult + metadata
        ┌───────────▼───────────┐
        │   Layer 2 – Explainer │
        │                       │
        │  ContextBuilder       │  → AnomalyContext
        │  LangChain Chain      │  → ExplanationResult
        │  OpenAI GPT-4o-mini   │    .explanation (text)
        └───────────┬───────────┘    .suggested_actions[]
                    │
        ┌───────────▼───────────┐
        │  InfluxDB Persistence │
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │    FastAPI REST API   │
        │  POST /detect         │
        │  POST /explain        │
        │  GET  /models         │
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │  Plotly Dash Dashboard│
        │  Live chart · KPI     │
        │  Anomaly log          │
        │  Explanation panel    │
        └───────────────────────┘
```

## Layer 1 — Anomaly Detection

All detectors implement `BaseDetector` with three methods:

| Method | Description |
|---|---|
| `fit(X)` | Learn normal behaviour from training data |
| `score(X)` | Return per-point anomaly scores ∈ [0, 1] |
| `detect(X, timestamps, metadata)` | Score + flag + wrap in `AnomalyResult` |

Scores are normalised to [0, 1] using training-set statistics, making the threshold consistent across models.

## Layer 2 — LLM Explanation

1. `ContextBuilder` takes an `AnomalyResult` + metadata and computes baseline statistics, deviation in σ, and trend classification.
2. `AnomalyExplainer` runs two LangChain chains:
   - **Explanation chain** — domain-aware system prompt + structured human prompt → plain-English paragraph
   - **Action chain** — extracts 2–4 concrete operator actions as a JSON list
3. Domain routing: `"iot"`, `"financial"`, `"operational"`, or `"general"` selects a specialised system prompt.

## Data Flow — Streaming Mode

```
Kafka/Redis message → JSON parse → buffer append
→ buffer full (N points) → on_window callback
→ detector.detect(window) → AnomalyResult
→ for each anomaly: ContextBuilder.build() → AnomalyExplainer.explain()
→ InfluxDBWriter.write_anomaly_result() + write_explanation()
→ Dash dashboard polls API every 5 s
```

## API Routes

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/api/v1/models` | List available detectors |
| `POST` | `/api/v1/detect` | Run anomaly detection |
| `POST` | `/api/v1/explain` | Generate LLM explanation |

## Deployment

- `docker/docker-compose.yml` brings up the full stack (API, dashboard, Kafka, Redis, InfluxDB) with a single command.
- API and dashboard are independently deployable; the dashboard reads from the API over HTTP.
- For production, replace the in-request `fit()` call in `/detect` with a model registry that loads pre-trained models from disk.
