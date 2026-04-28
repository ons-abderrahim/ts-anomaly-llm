"""
LSTM Autoencoder anomaly detector.
Best for: smooth univariate or low-dimensional multivariate time series.
Learns normal patterns; high reconstruction error → anomaly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseDetector


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class _LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM autoencoder."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            reconstruction: (batch, seq_len, input_dim)
        """
        _, (h, c) = self.encoder(x)                        # encode
        h_rep = h[-1].unsqueeze(1).repeat(1, x.size(1), 1) # repeat context
        out, _ = self.decoder(h_rep)                        # decode
        return out


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class LSTMAEDetector(BaseDetector):
    """
    LSTM Autoencoder–based anomaly detector.

    The model is trained to reconstruct normal windows.  At inference time,
    the per-window mean squared reconstruction error is used as the anomaly
    score and normalised to [0, 1] using the training error distribution.

    Args:
        window_size:   Length of sliding windows fed into the LSTM.
        hidden_dim:    LSTM hidden units.
        num_layers:    LSTM depth.
        dropout:       Dropout between LSTM layers (only applied if num_layers > 1).
        epochs:        Training epochs.
        lr:            Adam learning rate.
        batch_size:    Mini-batch size for training.
        threshold:     Anomaly score cutoff in [0, 1].
        device:        "cuda", "mps", or "cpu".  Auto-detected if None.
    """

    def __init__(
        self,
        window_size: int = 30,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
        threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        super().__init__(threshold=threshold)
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self._model: Optional[_LSTMAutoencoder] = None
        self._train_errors: Optional[np.ndarray] = None   # for score normalisation

    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "LSTMAEDetector":
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[:, None]

        input_dim = X.shape[1]
        self._model = _LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        windows = self._make_windows(X)                    # (N, window, input_dim)
        dataset = TensorDataset(torch.tensor(windows))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self._model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                recon = self._model(batch)
                loss = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}  loss={total_loss/len(loader):.4f}")

        # Store training errors for normalisation
        self._train_errors = self._reconstruction_errors(X)
        self._err_mean = self._train_errors.mean()
        self._err_std = self._train_errors.std() + 1e-8

        self._is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[:, None]

        errors = self._reconstruction_errors(X)
        # Normalise using training distribution → sigmoid squash to [0,1]
        z = (errors - self._err_mean) / self._err_std
        scores = 1 / (1 + np.exp(-z))
        return scores

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_windows(self, X: np.ndarray) -> np.ndarray:
        """Sliding window segmentation."""
        n = len(X)
        windows = [X[i: i + self.window_size] for i in range(n - self.window_size + 1)]
        return np.stack(windows)                           # (N, window_size, input_dim)

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Per-timestep mean reconstruction error."""
        windows = self._make_windows(X)                    # (N, W, D)
        tensor = torch.tensor(windows).to(self.device)

        self._model.eval()
        with torch.no_grad():
            recon = self._model(tensor).cpu().numpy()

        errors_per_window = ((windows - recon) ** 2).mean(axis=(1, 2))  # (N,)

        # Map window errors back to timestep errors (last window that covers each step)
        n = len(X)
        timestep_errors = np.zeros(n)
        counts = np.zeros(n)
        for i, err in enumerate(errors_per_window):
            timestep_errors[i: i + self.window_size] += err
            counts[i: i + self.window_size] += 1
        counts = np.maximum(counts, 1)
        return timestep_errors / counts

    def save(self, path: str):
        """Save model weights."""
        torch.save(self._model.state_dict(), path)

    def load(self, path: str, input_dim: int = 1):
        """Load model weights."""
        self._model = _LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        self._model.load_state_dict(torch.load(path, map_location=self.device))
        self._is_fitted = True
