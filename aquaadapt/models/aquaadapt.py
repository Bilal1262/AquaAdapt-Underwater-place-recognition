"""Combined DINOv2 backbone and learned projection head."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from aquaadapt.models.dinov2 import DINOv2Backbone
from aquaadapt.models.projection import ProjectionHead, ResidualAdapterHead


class AquaAdaptModel(nn.Module):
    MODES = {"projection_head_only", "partial_backbone_finetuning"}

    def __init__(self, cfg: dict[str, Any], mode: str = "projection_head_only"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Adapted model mode must be one of {sorted(self.MODES)}")
        model_cfg = cfg["model"]
        unfreeze = int(model_cfg["unfreeze_last_n_blocks"]) if mode == "partial_backbone_finetuning" else 0
        self.backbone = DINOv2Backbone(
            cfg["paths"]["torch_home"], model_cfg["pooling"], True, unfreeze,
            int(model_cfg["backbone_dim"]),
        )
        adapter_type = str(model_cfg.get("adapter_type", "mlp"))
        if adapter_type == "residual":
            descriptor_dim = int(model_cfg["descriptor_dim"])
            if descriptor_dim != self.backbone.output_dim:
                raise ValueError(
                    "Residual adapter requires model.descriptor_dim to equal "
                    f"the backbone dimension ({self.backbone.output_dim})"
                )
            self.projection = ResidualAdapterHead(
                self.backbone.output_dim,
                int(model_cfg["projection_hidden_dim"]),
                float(model_cfg["projection_dropout"]),
                float(model_cfg.get("adapter_scale", 0.1)),
            )
        elif adapter_type == "mlp":
            self.projection = ProjectionHead(
                self.backbone.output_dim, int(model_cfg["projection_hidden_dim"]),
                int(model_cfg["descriptor_dim"]), float(model_cfg["projection_dropout"]),
            )
        else:
            raise ValueError("model.adapter_type must be 'mlp' or 'residual'")
        self.adapter_type = adapter_type
        self.mode_name = mode

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            with torch.no_grad():
                return self.backbone(images, normalize=False)
        return self.backbone(images, normalize=False)

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.project_features(self.extract_features(images))

    def optimizer_groups(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = [{
            "params": list(self.projection.parameters()),
            "lr": float(cfg["training"]["head_learning_rate"]),
            "name": "projection_head",
        }]
        backbone_parameters = [p for p in self.backbone.parameters() if p.requires_grad]
        if backbone_parameters:
            groups.append({
                "params": backbone_parameters,
                "lr": float(cfg["training"]["backbone_learning_rate"]),
                "name": "backbone",
            })
        return groups
