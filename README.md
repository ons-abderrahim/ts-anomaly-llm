# ts-anomaly-llm
Time Series Anomaly Detection with LLM Explanations Detect anomalies in IoT / financial / operational data and explain them in natural language


# 🔍 Time Series Anomaly Detection with LLM Explanations

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](docker/docker-compose.yml)

> **Detect anomalies in IoT, financial, and operational data — then explain them in plain English using LLMs.**

Anomaly detection models flag issues but can't explain *why* something is anomalous in terms non-engineers understand. This system bridges that gap: a two-layer pipeline that detects, then explains, making ML-powered monitoring accessible to operations teams.

---

## 🧠 How It Works

```
Streaming Data (Kafka/Redis)
        │
        ▼
┌───────────────────┐
│  Anomaly Detector │  ← Isolation Forest / LSTM-AE / Transformer
│  (Layer 1)        │    flags anomalies with score + context
└────────┬──────────┘
         │  anomaly context
         ▼
┌───────────────────┐
│   LLM Explainer   │  ← LangChain + OpenAI API
│   (Layer 2)       │    generates plain-English root-cause explanations
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Plotly Dash UI   │  ← Real-time dashboard for operators
│  + FastAPI REST   │
└───────────────────┘
```

---

## ✨ Features

- **Three anomaly detection models** — choose by use case:
  - `IsolationForest` — fast, unsupervised, great for tabular/IoT data
  - `LSTM-Autoencoder` — sequence-aware, ideal for smooth time series
  - `Transformer-AE` — best accuracy for complex multivariate series
- **LLM explanation layer** — LangChain chain with context-aware prompts converts raw anomaly scores into operator-readable narratives
- **Streaming ingestion** — Kafka and Redis Streams adapters with InfluxDB persistence
- **REST API** — FastAPI endpoints for detection, explanation, and model management
- **Real-time dashboard** — Plotly Dash UI with live anomaly feed and explanation panel
- **Pluggable architecture** — swap models, LLMs, or data sources with minimal config changes

---

## 📁 Project Structure

```
ts-anomaly-llm/
├── src/
│   ├── anomaly/               # Detection models
│   │   ├── base.py            # Abstract detector interface
│   │   ├── isolation_forest.py
│   │   ├── lstm_autoencoder.py
│   │   ├── transformer_ae.py
│   │   └── detector_factory.py
│   ├── llm/                   # Explanation layer
│   │   ├── explainer.py       # LangChain chain
│   │   ├── prompts.py         # Prompt templates
│   │   └── context_builder.py # Anomaly → prompt context
│   ├── ingestion/             # Data streaming
│   │   ├── kafka_consumer.py
│   │   ├── redis_consumer.py
│   │   └── influxdb_writer.py
│   ├── api/                   # FastAPI app
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── detect.py
│   │   │   ├── explain.py
│   │   │   └── models.py
│   │   └── schemas.py
│   └── dashboard/             # Plotly Dash UI
│       ├── app.py
│       ├── layout.py
│       └── callbacks.py
├── tests/
│   ├── test_anomaly_models.py
│   ├── test_explainer.py
│   ├── test_api.py
│   └── fixtures/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_explanation_quality.ipynb
├── configs/
│   ├── config.yaml            # Main config
│   ├── models.yaml            # Model hyperparameters
│   └── prompts.yaml           # LLM prompt templates
├── docker/
│   ├── docker-compose.yml
│   ├── Dockerfile.api
│   └── Dockerfile.dashboard
├── scripts/
│   ├── train.py               # Model training CLI
│   ├── evaluate.py            # Evaluation metrics
│   └── seed_data.py           # Generate synthetic data
├── docs/
│   ├── architecture.md
│   ├── model_comparison.md
│   └── api_reference.md
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourname/ts-anomaly-llm.git
cd ts-anomaly-llm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set your OPENAI_API_KEY and Kafka/Redis endpoints
```

### 3. Start Infrastructure

```bash
docker compose -f docker/docker-compose.yml up -d
```

### 4. Train a Model

```bash
python scripts/train.py --model lstm_ae --data data/sample_iot.csv --output models/
```

### 5. Run the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

### 6. Launch the Dashboard

```bash
python src/dashboard/app.py
# → http://localhost:8050
```

---

## 🔌 API Reference

### Detect Anomalies

```http
POST /api/v1/detect
Content-Type: application/json

{
  "series": [1.2, 1.1, 1.3, 9.8, 1.2],
  "timestamps": ["2024-01-01T00:00:00Z", ...],
  "model": "lstm_ae",
  "metadata": { "sensor_id": "temp-42", "unit": "celsius" }
}
```

**Response:**
```json
{
  "anomalies": [
    {
      "timestamp": "2024-01-01T00:03:00Z",
      "value": 9.8,
      "score": 0.94,
      "is_anomaly": true
    }
  ],
  "model_used": "lstm_ae"
}
```

### Explain an Anomaly

```http
POST /api/v1/explain
Content-Type: application/json

{
  "anomaly": { "timestamp": "...", "value": 9.8, "score": 0.94 },
  "context_window": [1.2, 1.1, 1.3, 9.8, 1.2],
  "metadata": { "sensor_id": "temp-42", "unit": "celsius", "location": "Pump Room 3" }
}
```

**Response:**
```json
{
  "explanation": "Temperature sensor temp-42 in Pump Room 3 recorded an anomalous spike to 9.8°C at 00:03 UTC — roughly 8× the recent baseline of ~1.2°C. This pattern is consistent with either a cooling system failure or a sensor calibration fault. Recommend inspecting the pump coolant loop before the next scheduled maintenance window.",
  "confidence": 0.87,
  "suggested_actions": ["Inspect coolant loop", "Cross-check with adjacent sensors"]
}
```

Full API docs available at `http://localhost:8000/docs` after startup.

---

## 🤖 Models

| Model | Best For | Training Time | Inference |
|---|---|---|---|
| `isolation_forest` | Tabular, sparse anomalies | < 1 min | < 5 ms |
| `lstm_ae` | Smooth, univariate time series | ~10 min | ~20 ms |
| `transformer_ae` | Multivariate, complex patterns | ~60 min | ~50 ms |

See [`docs/model_comparison.md`](docs/model_comparison.md) for detailed benchmarks on public datasets.

---

## 📊 Datasets & Resources

| Resource | Link |
|---|---|
| Google TODS Library | [github.com/google-research/google-research/tree/master/tods](https://github.com/google-research/google-research/tree/master/tods) |
| Luminol (LinkedIn) | [github.com/linkedin/luminol](https://github.com/linkedin/luminol) |
| Kats by Meta | [github.com/facebookresearch/Kats](https://github.com/facebookresearch/Kats) |
| NAB Benchmark | [github.com/numenta/NAB](https://github.com/numenta/NAB) |

---

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 🐳 Docker Deployment

```bash
docker compose -f docker/docker-compose.yml up --build
```

Services started:
- `api` → FastAPI on port `8000`
- `dashboard` → Plotly Dash on port `8050`
- `kafka` → Kafka broker on port `9092`
- `redis` → Redis on port `6379`
- `influxdb` → InfluxDB on port `8086`

---

## 🗺️ Roadmap

- [ ] Add OCSVM and DeepSVDD detectors
- [ ] Multi-language explanation support
- [ ] Slack / PagerDuty alert integrations
- [ ] Model drift detection & auto-retraining
- [ ] Batch CSV upload endpoint
