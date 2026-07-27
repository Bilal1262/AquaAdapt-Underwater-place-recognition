"""Projection from DINOv2 features to AquaAdapt descriptors."""

import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):
    """384 → 512 → GELU → Dropout → 256 with L2 normalization."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 512, output_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(features), dim=-1)


class ResidualAdapterHead(nn.Module):
    """DINO-preserving residual adapter initialized as the identity mapping."""

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        adapter_scale: float = 0.1,
    ):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        # Zero initialization makes the initial descriptor exactly normalized
        # raw DINOv2. Training therefore learns a correction instead of
        # replacing the foundation descriptor with a random projection.
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.adapter_scale = float(adapter_scale)
        self.output_dim = input_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = F.normalize(features, dim=-1)
        correction = self.adapter(raw)
        return F.normalize(raw + self.adapter_scale * correction, dim=-1)
