"""
Transformer Autoencoder anomaly detector.
Best for: complex multivariate time series requiring long-range dependencies.
Uses masked self-attention to reconstruct normal patterns.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseDetector


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                               # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer AE
# ---------------------------------------------------------------------------

class _TransformerAE(nn.Module):
    """Encoder-only Transformer that reconstructs its input."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder_proj = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            reconstruction: (batch, seq_len, input_dim)
        """
        z = self.pos_enc(self.input_proj(x))
        z = self.encoder(z)
        return self.decoder_proj(z)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class TransformerAEDetector(BaseDetector):
    """
    Transformer Autoencoder–based anomaly detector.

    Trains on normal windows; flags points where reconstruction error
    exceeds the learned threshold.

    Args:
        window_size:        Sliding window length.
        d_model:            Transformer embedding dimension.
        nhead:              Number of attention heads.
        num_encoder_layers: Depth of transformer encoder.
        dim_feedforward:    FFN hidden size.
        dropout:            Dropout probability.
        epochs:             Training epochs.
        lr:                 Adam learning rate.
        batch_size:         Mini-batch size.
        threshold:          Anomaly score cutoff in [0, 1].
        device:             Compute device ("cuda" / "mps" / "cpu").
    """

    def __init__(
        self,
        window_size: int = 60,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        epochs: int = 80,
        lr: float = 5e-4,
        batch_size: int = 64,
        threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        super().__init__(threshold=threshold)
        self.window_size = window_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self._model: Optional[_TransformerAE] = None
        self._err_mean: float = 0.0
        self._err_std: float = 1.0

    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "TransformerAEDetector":
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[:, None]

        input_dim = X.shape[1]
        assert self.d_model % self.nhead == 0, "d_model must be divisible by nhead"

        self._model = _TransformerAE(
            input_dim=input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self.device)

        windows = self._make_windows(X)
        dataset = TensorDataset(torch.tensor(windows))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
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
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}  loss={total_loss/len(loader):.5f}")

        errors = self._reconstruction_errors(X)
        self._err_mean = errors.mean()
        self._err_std = errors.std() + 1e-8
        self._is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[:, None]
        errors = self._reconstruction_errors(X)
        z = (errors - self._err_mean) / self._err_std
        return np.clip(1 / (1 + np.exp(-z)), 0, 1)

    # ------------------------------------------------------------------

    def _make_windows(self, X: np.ndarray) -> np.ndarray:
        return np.stack([X[i: i + self.window_size] for i in range(len(X) - self.window_size + 1)])

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        windows = self._make_windows(X)
        tensor = torch.tensor(windows).to(self.device)

        self._model.eval()
        with torch.no_grad():
            recon = self._model(tensor).cpu().numpy()

        per_window = ((windows - recon) ** 2).mean(axis=(1, 2))
        n = len(X)
        errors = np.zeros(n)
        counts = np.zeros(n)
        for i, err in enumerate(per_window):
            errors[i: i + self.window_size] += err
            counts[i: i + self.window_size] += 1
        return errors / np.maximum(counts, 1)
