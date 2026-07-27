"""Controlled clean-database/corrupted-query robustness benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aquaadapt.augmentations.pipeline import apply_corruption
from aquaadapt.data.datasets import ManifestImageDataset
from aquaadapt.evaluation.place_recognition import evaluate_arrays
from aquaadapt.models.aquaadapt import AquaAdaptModel
from aquaadapt.models.dinov2 import DINOv2Backbone
from aquaadapt.retrieval.descriptors import classical_enhancement, descriptor_directory
from aquaadapt.retrieval.index import normalize_descriptors
from aquaadapt.training.checkpointing import load_checkpoint


def _model_for_method(cfg: dict[str, Any], method: str, checkpoint: str | Path | None) -> torch.nn.Module:
    if method in {"raw_dinov2", "enhanced_dinov2"}:
        return DINOv2Backbone(
            cfg["paths"]["torch_home"], cfg["model"]["pooling"], True, 0,
            int(cfg["model"]["backbone_dim"]),
        )
    if checkpoint is None:
        raise ValueError("AquaAdapt robustness evaluation requires --checkpoint")
    state = load_checkpoint(checkpoint)
    model = AquaAdaptModel(cfg, str(state.get("mode", "projection_head_only")))
    model.load_state_dict(state["model_state"])
    return model


def _encode_corrupted(
    cfg: dict[str, Any], metadata: pd.DataFrame, model: torch.nn.Module,
    method: str, corruption: str, severity: int,
) -> np.ndarray:
    seed = int(cfg["project"]["seed"])

    def enhancer(image: np.ndarray) -> np.ndarray:
        degraded = apply_corruption(image, corruption, severity, seed)
        return classical_enhancement(degraded) if method == "enhanced_dinov2" else degraded

    dataset = ManifestImageDataset(metadata, int(cfg["images"]["model_size"]), enhancer)
    device = next(model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _ in tqdm(loader, desc=f"{method} {corruption} s{severity}", leave=False):
            chunks.append(model(images.to(device)).cpu().numpy())
    return normalize_descriptors(np.concatenate(chunks))


def run_robustness(
    cfg: dict[str, Any], checkpoint: str | Path | None,
    methods: list[str] | None = None, limit: int | None = None,
) -> Path:
    methods = methods or ["raw_dinov2", "enhanced_dinov2", "aquaadapt"]
    output = Path("results") / str(cfg["project"]["run_name"])
    output.mkdir(parents=True, exist_ok=True)
    missing_caches: list[str] = []
    for method in methods:
        clean_dir = descriptor_directory(cfg, method)
        required = (clean_dir / "descriptors.npy", clean_dir / "descriptor_metadata.csv")
        absent = [str(path) for path in required if not path.is_file()]
        if absent:
            missing_caches.extend(absent)
    if missing_caches:
        run_name = str(cfg["project"]["run_name"])
        missing_lines = "\n  - ".join(missing_caches)
        hint = (
            "\nRun the extraction/baseline/encoding pipeline first. For the mclab_2 "
            "transfer configuration, use:\n"
            "  bash scripts/run_mclab2_transfer.sh"
            if run_name == "mclab2_transfer_5hz"
            else
            "\nRun `aquaadapt baseline` for raw/enhanced methods and "
            "`aquaadapt encode` for AquaAdapt before robustness."
        )
        raise FileNotFoundError(
            "Robustness requires clean descriptor caches for every requested method. "
            f"Missing:\n  - {missing_lines}{hint}"
        )
    rows: list[dict[str, Any]] = []
    for method in methods:
        clean_dir = descriptor_directory(cfg, method)
        clean = np.load(clean_dir / "descriptors.npy")
        metadata = pd.read_csv(clean_dir / "descriptor_metadata.csv")
        if limit is not None and limit < len(metadata):
            # Preserve the chronological/split span instead of truncating away the test block.
            selected = np.linspace(0, len(metadata) - 1, int(limit), dtype=int)
            clean, metadata = clean[selected], metadata.iloc[selected].reset_index(drop=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _model_for_method(cfg, method, checkpoint).to(device).eval()
        for corruption in cfg["robustness"]["corruptions"]:
            for severity in cfg["robustness"]["severity_levels"]:
                corrupted = clean if int(severity) == 0 else _encode_corrupted(
                    cfg, metadata, model, method, str(corruption), int(severity)
                )
                metrics, _ = evaluate_arrays(clean, metadata, cfg, corrupted)
                stability = float(np.mean(np.sum(normalize_descriptors(clean) * corrupted, axis=1)))
                metrics["descriptor_cosine_stability"] = stability
                for metric, value in metrics.items():
                    if not isinstance(value, (bool, np.bool_)) and isinstance(
                        value, (int, float, np.integer, np.floating)
                    ):
                        rows.append({
                            "method": method, "corruption": corruption, "severity": int(severity),
                            "metric": metric, "value": value,
                            "eligible_query_count": metrics["eligible_queries"],
                        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    destination = output / "robustness_results.csv"
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(
            "Robustness produced no rows despite complete descriptor caches; "
            "check that the descriptor metadata is non-empty."
        )
    drop_rows: list[dict[str, Any]] = []
    for (method, corruption, metric), group in frame[
        frame["metric"].isin(["Recall@1", "Recall@5", "MRR"])
    ].groupby(["method", "corruption", "metric"]):
        clean_values = group.loc[group["severity"] == 0, "value"].astype(float)
        if clean_values.empty or not np.isfinite(clean_values.iloc[0]):
            continue
        clean_value = float(clean_values.iloc[0])
        for _, row in group.iterrows():
            value = float(row["value"])
            drop_rows.append({
                "method": method, "corruption": corruption, "severity": int(row["severity"]),
                "metric": f"relative_drop_{metric}",
                "value": clean_value - value if np.isfinite(value) else np.nan,
                "eligible_query_count": row["eligible_query_count"],
            })
    if drop_rows:
        frame = pd.concat([frame, pd.DataFrame(drop_rows)], ignore_index=True)
    frame.to_csv(destination, index=False)
    for metric in (
        "Recall@1", "Recall@5", "MRR", "descriptor_cosine_stability",
        "relative_drop_Recall@1", "evaluation_coverage",
    ):
        subset = frame[frame["metric"] == metric]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for (method, corruption), group in subset.groupby(["method", "corruption"]):
            ax.plot(group["severity"], group["value"], marker="o", label=f"{method}/{corruption}")
        ax.set(xlabel="severity", ylabel=metric, title=f"{metric} versus corruption severity")
        ax.legend(fontsize=6, ncol=2); fig.tight_layout()
        fig.savefig(output / f"robustness_{metric.replace('@', 'at').lower()}.png", dpi=160)
        plt.close(fig)
    return destination
