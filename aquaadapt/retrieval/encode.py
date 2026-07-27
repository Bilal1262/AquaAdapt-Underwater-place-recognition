"""Adapted descriptor encoding."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aquaadapt.data.datasets import ManifestImageDataset
from aquaadapt.models.aquaadapt import AquaAdaptModel
from aquaadapt.retrieval.descriptors import descriptor_directory
from aquaadapt.retrieval.index import normalize_descriptors
from aquaadapt.training.checkpointing import load_checkpoint


def encode_adapted(
    cfg: dict[str, Any], manifest_path: str | Path, checkpoint: str | Path,
    limit: int | None = None, overwrite: bool = False,
) -> Path:
    output = descriptor_directory(cfg, "aquaadapt")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "descriptors.npy"
    metadata = pd.read_csv(manifest_path)
    if limit is not None:
        metadata = metadata.iloc[:limit].copy()
    checkpoint_path = Path(checkpoint)
    manifest_digest = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps({
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": manifest_digest,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_mtime_ns": checkpoint_path.stat().st_mtime_ns,
        "checkpoint_size": checkpoint_path.stat().st_size,
        "rows": len(metadata), "model_size": cfg["images"]["model_size"],
    }, sort_keys=True).encode()).hexdigest()
    config_path = output / "descriptor_config.json"
    if destination.is_file() and config_path.is_file() and not overwrite:
        cached = json.loads(config_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            return output
    state = load_checkpoint(checkpoint)
    mode = str(state.get("mode", "projection_head_only"))
    model = AquaAdaptModel(cfg, mode)
    model.load_state_dict(state["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    loader = DataLoader(
        ManifestImageDataset(metadata, int(cfg["images"]["model_size"])),
        batch_size=int(cfg["training"]["batch_size"]), shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
    )
    collected: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _ in tqdm(loader, desc="encode aquaadapt"):
            collected.append(model(images.to(device)).cpu().numpy())
    descriptors = normalize_descriptors(np.concatenate(collected))
    if not np.all(np.isfinite(descriptors)) or not np.allclose(np.linalg.norm(descriptors, axis=1), 1, atol=1e-4):
        raise ValueError("Adapted descriptors failed finite/unit-norm validation")
    np.save(destination, descriptors)
    metadata.to_csv(output / "descriptor_metadata.csv", index=False)
    (output / "descriptor_config.json").write_text(json.dumps({
        "method": "aquaadapt", "checkpoint": str(checkpoint),
        "dimension": int(descriptors.shape[1]), "mode": mode, "fingerprint": fingerprint,
    }, indent=2), encoding="utf-8")
    return output
