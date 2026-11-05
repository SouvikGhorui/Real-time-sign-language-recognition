"""The sign classifier and a helper for loading it for inference."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .features import FEATURE_DIM, prepare_clip, sample_frames, to_vector, uniform_indices


class ConvBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class SignClassifier(nn.Module):
    """Temporal convolutions for local motion, a BiGRU for order, attention to pool.

    Input: (batch, frames, FEATURE_DIM). Output: (batch, classes) logits.
    Normalisation statistics are stored in the model, so they travel with the checkpoint.
    """

    def __init__(self, num_classes: int, input_dim: int = FEATURE_DIM,
                 hidden: int = 192, dropout: float = 0.4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("std", torch.ones(input_dim))
        self.embed = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.conv = nn.Sequential(ConvBlock(hidden, 5, dropout), ConvBlock(hidden, 5, dropout))
        self.gru = nn.GRU(hidden, hidden // 2, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.attention = nn.Linear(hidden, 1)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        x = self.embed(x)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.gru(x)
        weights = torch.softmax(self.attention(x), dim=1)
        return self.classifier((weights * x).sum(dim=1))


class SignPredictor:
    """Loads a checkpoint and classifies raw landmark segments."""

    def __init__(self, checkpoint: Path):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.glosses: list[str] = state["glosses"]
        self.num_frames: int = state["num_frames"]
        self.model = SignClassifier(len(self.glosses), hidden=state["hidden"], dropout=state["dropout"])
        self.model.load_state_dict(state["model"])
        self.model.eval()

    @torch.inference_mode()
    def predict(self, raw: np.ndarray, aspect: float, top_k: int = 5) -> list[tuple[str, float]]:
        """(T, 218) -> [(gloss, probability), ...], best first. Empty if unusable."""
        clip = prepare_clip(raw, aspect)
        if clip is None:
            return []
        sampled = sample_frames(clip, uniform_indices(len(clip.pose), self.num_frames))
        inputs = torch.from_numpy(to_vector(sampled)).unsqueeze(0)
        probabilities = torch.softmax(self.model(inputs), dim=1)[0]
        values, indices = probabilities.topk(min(top_k, len(self.glosses)))
        return [(self.glosses[i], float(v)) for v, i in zip(values, indices)]
