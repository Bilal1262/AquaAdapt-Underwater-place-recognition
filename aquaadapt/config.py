"""YAML configuration loading and validation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "project": {"name": "aquaadapt", "seed": 42, "run_name": "mclab1"},
    "paths": {
        "bag": "/mnt/windows/datasets/ntnu_underwater/subset-mclab/mclab_1/mclab_1.bag",
        "tum": "/mnt/windows/datasets/ntnu_underwater/subset-mclab/mclab_1/mclab_1_baseline.tum",
        "calibrations": "/mnt/windows/datasets/ntnu_underwater/calibrations",
        "processed_root": "/mnt/windows/datasets/ntnu_underwater/processed",
        "torch_home": "/mnt/windows/datasets/model_cache/torch",
    },
    "extraction": {
        "camera_topic": "auto", "sample_rate_hz": 2.0, "max_frames": None,
        "jpeg_quality": 95, "overwrite": False, "resume": True,
        "pose_max_time_difference_sec": 0.10, "timestamp_offset_sec": 0.0,
    },
    "images": {"model_size": 224, "preserve_original": True},
    "model": {
        "backbone": "dinov2_vits14", "backbone_dim": 384,
        "descriptor_dim": 256, "pooling": "cls", "freeze_backbone": True,
        "unfreeze_last_n_blocks": 0, "projection_hidden_dim": 512,
        "projection_dropout": 0.1, "adapter_type": "mlp",
        "adapter_scale": 0.1,
    },
    "training": {
        "epochs": 20, "batch_size": 32, "num_workers": 4,
        "head_learning_rate": 1e-3, "backbone_learning_rate": 1e-5,
        "weight_decay": 1e-4, "temperature": 0.07,
        "temporal_positive_window_sec": 1.0,
        "temporal_positive_max_pose_distance_m": 1.0,
        "use_temporal_positive": True, "mixed_precision": True,
        "balance_trajectories": False,
        "early_stopping_patience": 5, "gradient_clip_norm": 1.0,
        "gradient_accumulation": 1,
    },
    "splits": {
        "policy": "chronological", "train": 0.60, "guard1": 0.02,
        "validation": 0.18, "guard2": 0.02, "test": 0.18,
    },
    "evaluation": {
        "pose_positive_radius_m": 1.5, "temporal_exclusion_sec": 10.0,
        "recalls": [1, 5, 10], "query_stride": 2, "database_stride": 2,
        "allow_progress_fallback": False,
    },
    "robustness": {
        "severity_levels": [0, 1, 2, 3],
        "corruptions": ["low_light", "color_attenuation", "haze", "blur", "marine_snow"],
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate values that would otherwise lead to subtle failures."""
    errors: list[str] = []
    ext = cfg["extraction"]
    model = cfg["model"]
    splits = cfg["splits"]
    training_manifests = cfg["paths"].get("training_manifests")
    if training_manifests is not None and (
        not isinstance(training_manifests, list) or len(training_manifests) < 2
    ):
        errors.append("paths.training_manifests must contain at least two manifest paths")
    if float(ext["sample_rate_hz"]) <= 0:
        errors.append("extraction.sample_rate_hz must be > 0")
    if not 1 <= int(ext["jpeg_quality"]) <= 100:
        errors.append("extraction.jpeg_quality must be in [1, 100]")
    if float(ext["pose_max_time_difference_sec"]) < 0:
        errors.append("extraction.pose_max_time_difference_sec must be >= 0")
    if int(cfg["images"]["model_size"]) <= 0:
        errors.append("images.model_size must be > 0")
    if model["backbone"] != "dinov2_vits14":
        errors.append("model.backbone must be dinov2_vits14 (larger models are intentionally unsupported)")
    if int(model["backbone_dim"]) != 384:
        errors.append("model.backbone_dim must be 384 for dinov2_vits14")
    if model["pooling"] not in {"cls", "mean"}:
        errors.append("model.pooling must be 'cls' or 'mean'")
    if model["adapter_type"] not in {"mlp", "residual"}:
        errors.append("model.adapter_type must be mlp or residual")
    if model["adapter_type"] == "residual" and int(model["descriptor_dim"]) != int(model["backbone_dim"]):
        errors.append("residual adapter requires descriptor_dim == backbone_dim")
    if float(model["adapter_scale"]) <= 0:
        errors.append("model.adapter_scale must be > 0")
    if int(model["unfreeze_last_n_blocks"]) not in {0, 1, 2}:
        errors.append("model.unfreeze_last_n_blocks must be 0, 1, or 2")
    if splits["policy"] not in {"chronological", "all_test"}:
        errors.append("splits.policy must be chronological or all_test")
    if abs(sum(float(splits[k]) for k in ("train", "guard1", "validation", "guard2", "test")) - 1) > 1e-6:
        errors.append("chronological split fractions must sum to 1")
    if not 0 < float(cfg["training"]["temperature"]):
        errors.append("training.temperature must be > 0")
    if any(int(k) < 1 for k in cfg["evaluation"]["recalls"]):
        errors.append("evaluation.recalls must contain positive integers")
    if errors:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))


def load_config(path: str | Path | None, quick: bool = False) -> dict[str, Any]:
    """Load a configuration and fill backward-compatible defaults."""
    supplied: dict[str, Any] = {}
    if path:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration not found: {config_path}")
        supplied = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(supplied, dict):
            raise ValueError(f"Configuration root must be a mapping: {config_path}")
    cfg = _merge(DEFAULTS, supplied)
    if quick:
        cfg["extraction"].update({"sample_rate_hz": 1.0, "max_frames": 300})
        cfg["training"].update({"epochs": 1, "batch_size": min(8, int(cfg["training"]["batch_size"])), "num_workers": 0})
        cfg["robustness"]["severity_levels"] = [0, 2]
    validate_config(cfg)
    return cfg
