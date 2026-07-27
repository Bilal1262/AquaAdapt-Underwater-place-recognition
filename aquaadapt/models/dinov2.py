"""Official PyTorch Hub DINOv2 ViT-S/14 wrapper."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class DINOv2Backbone(nn.Module):
    """DINOv2 wrapper with explicit pooling and controlled unfreezing."""

    def __init__(
        self,
        torch_home: str,
        pooling: str = "cls",
        freeze: bool = True,
        unfreeze_last_n_blocks: int = 0,
        expected_dim: int = 384,
    ):
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError(f"Unsupported pooling: {pooling}")
        os.environ.setdefault("TORCH_HOME", torch_home)
        try:
            self.model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14",
                pretrained=True, trust_repo=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "DINOv2 loading failed. Check network access or pre-populate TORCH_HOME "
                f"({torch_home}) with the official facebookresearch/dinov2 Hub repository and weights. "
                f"Original error: {exc}"
            ) from exc
        self.pooling = pooling
        self.output_dim = int(getattr(self.model, "embed_dim", expected_dim))
        if self.output_dim != expected_dim:
            raise ValueError(f"dinov2_vits14 feature dimension is {self.output_dim}, expected {expected_dim}")
        self.configure_trainability(freeze, unfreeze_last_n_blocks)

    @property
    def blocks(self) -> Any:
        return self.model.blocks

    def configure_trainability(self, freeze: bool, unfreeze_last_n_blocks: int) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = not freeze
        if freeze and unfreeze_last_n_blocks:
            for block in self.blocks[-unfreeze_last_n_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
        self._fully_frozen = not any(parameter.requires_grad for parameter in self.model.parameters())
        if self._fully_frozen:
            self.model.eval()

    def train(self, mode: bool = True) -> "DINOv2Backbone":
        super().train(mode)
        if self._fully_frozen:
            self.model.eval()
        return self

    def forward(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        features = self.model.forward_features(images)
        if self.pooling == "cls":
            descriptor = features["x_norm_clstoken"]
        else:
            descriptor = features["x_norm_patchtokens"].mean(dim=1)
        return F.normalize(descriptor, dim=-1) if normalize else descriptor

