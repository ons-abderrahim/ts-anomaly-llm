#!/usr/bin/env python
"""
CLI script to train an anomaly detection model on a CSV file and save the result.

Usage:
    python scripts/train.py --model lstm_ae --data data/sample.csv --output models/
    python scripts/train.py --model isolation_forest --data data/iot.csv --threshold 0.4
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ is on the path when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.anomaly import get_detector


def parse_args():
    parser = argparse.ArgumentParser(description="Train an anomaly detection model")
    parser.add_argument("--model", required=True,
                        choices=["isolation_forest", "lstm_ae", "transformer_ae"],
                        help="Detector model to train")
    parser.add_argument("--data", required=True,
                        help="Path to training CSV file (must have a 'value' column)")
    parser.add_argument("--value-col", default="value",
                        help="Name of the numeric value column in the CSV")
    parser.add_argument("--output", default="models/",
                        help="Directory to save the trained model")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Anomaly score threshold (0–1)")
    # Model-specific hyperparams
    parser.add_argument("--window-size", type=int, default=30,
                        help="Sliding window size (LSTM-AE / Transformer-AE only)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (LSTM-AE / Transformer-AE only)")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="LSTM hidden dim or Transformer d_model")
    return parser.parse_args()


def load_data(path: str, value_col: str) -> np.ndarray:
    df = pd.read_csv(path)
    if value_col not in df.columns:
        raise ValueError(
            f"Column '{value_col}' not found. Available columns: {list(df.columns)}"
        )
    values = df[value_col].dropna().values.astype(float)
    print(f"  Loaded {len(values)} samples from '{path}'")
    return values


def main():
    args = parse_args()

    print(f"\n{'='*55}")
    print(f"  Training: {args.model.upper()}")
    print(f"  Data:     {args.data}")
    print(f"  Threshold:{args.threshold}")
    print(f"{'='*55}\n")

    # Load data
    X = load_data(args.data, args.value_col)

    # Build detector kwargs
    kwargs: dict = {"threshold": args.threshold}
    if args.model in ("lstm_ae", "transformer_ae"):
        kwargs.update({
            "window_size": args.window_size,
            "epochs": args.epochs,
        })
    if args.model == "lstm_ae":
        kwargs["hidden_dim"] = args.hidden_dim
    if args.model == "transformer_ae":
        kwargs["d_model"] = args.hidden_dim

    # Train
    detector = get_detector(args.model, **kwargs)
    print("Training...")
    detector.fit(X)
    print("Training complete.\n")

    # Evaluate on training data
    result = detector.detect(X)
    print(f"  Training-set anomaly rate: {result.anomaly_count}/{len(X)} "
          f"({result.anomaly_count/len(X)*100:.2f}%)")

    # Save
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{args.model}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(detector, f)

    print(f"\n✅ Model saved to: {model_path}")


if __name__ == "__main__":
    main()
