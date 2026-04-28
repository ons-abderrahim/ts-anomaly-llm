#!/usr/bin/env python
"""
Generate synthetic time series CSV files for testing and development.

Produces three domain-specific datasets:
  - data/sample_iot.csv       — temperature sensor with occasional spikes
  - data/sample_financial.csv — price series with flash-crash pattern
  - data/sample_operational.csv — server latency with traffic surge

Usage:
    python scripts/seed_data.py
    python scripts/seed_data.py --out data/ --points 2000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_iot(n: int, seed: int = 42) -> pd.DataFrame:
    """Temperature sensor: sinusoidal baseline + Gaussian noise + spike anomalies."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8 * np.pi, n)
    values = 20 + 5 * np.sin(t) + rng.normal(0, 0.3, n)

    # Inject 5 spike anomalies
    for idx in rng.integers(50, n - 50, size=5):
        values[idx] += rng.uniform(15, 30)

    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min")
    return pd.DataFrame({
        "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "value": np.round(values, 4),
        "sensor_id": "temp-01",
        "unit": "celsius",
        "location": "Pump Room 3",
    })


def generate_financial(n: int, seed: int = 7) -> pd.DataFrame:
    """Stock price: geometric Brownian motion + flash-crash anomalies."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0001, 0.01, n)
    prices = 100.0 * np.exp(np.cumsum(returns))

    # Inject 3 flash-crash events (sudden drop + recovery)
    for idx in rng.integers(100, n - 10, size=3):
        prices[idx] *= rng.uniform(0.6, 0.75)    # drop 25–40%
        prices[idx + 1] *= rng.uniform(0.9, 1.1) # partial recovery

    timestamps = pd.date_range("2024-01-01 09:30", periods=n, freq="1min")
    return pd.DataFrame({
        "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "value": np.round(prices, 4),
        "sensor_id": "AAPL",
        "unit": "USD",
        "location": "NASDAQ",
    })


def generate_operational(n: int, seed: int = 13) -> pd.DataFrame:
    """Server latency: low baseline with periodic traffic surge anomalies."""
    rng = np.random.default_rng(seed)
    values = 50 + rng.exponential(5, n)  # baseline ~50 ms

    # Inject 4 traffic surge events (elevated latency windows)
    for idx in rng.integers(50, n - 30, size=4):
        duration = rng.integers(5, 20)
        window = slice(int(idx), int(idx) + duration)
        values[window] += rng.uniform(200, 600)

    timestamps = pd.date_range("2024-01-01", periods=n, freq="30s")
    return pd.DataFrame({
        "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "value": np.round(values, 2),
        "sensor_id": "api-gateway-latency",
        "unit": "ms",
        "location": "us-east-1",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/", help="Output directory")
    parser.add_argument("--points", type=int, default=1000, help="Number of data points")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    datasets = {
        "sample_iot.csv": generate_iot(args.points),
        "sample_financial.csv": generate_financial(args.points),
        "sample_operational.csv": generate_operational(args.points),
    }

    for fname, df in datasets.items():
        path = out / fname
        df.to_csv(path, index=False)
        print(f"  ✅ {path}  ({len(df)} rows)")

    print(f"\nGenerated {len(datasets)} datasets in '{args.out}'")


if __name__ == "__main__":
    main()
