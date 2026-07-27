"""Cached descriptor extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aquaadapt.data.datasets import ManifestImageDataset
from aquaadapt.models.dinov2 import DINOv2Backbone
from aquaadapt.retrieval.index import normalize_descriptors


def classical_enhancement(image: np.ndarray) -> np.ndarray:
    """Gray-world white balance, luminance CLAHE, and mild gamma correction."""
    import cv2

    values = image.astype(np.float32)
    means = values.mean(axis=(0, 1))
    scale = means.mean() / np.maximum(means, 1e-6)
    balanced = np.clip(values * scale, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gamma = 0.9
    table = np.clip((np.arange(256) / 255.0) ** gamma * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(enhanced, table)


def descriptor_directory(cfg: dict[str, Any], method: str) -> Path:
    return Path(cfg["paths"]["processed_root"]) / "descriptors" / str(cfg["project"]["run_name"]) / method


def encode_raw(
    cfg: dict[str, Any], manifest_path: str | Path, method: str,
    limit: int | None = None, overwrite: bool = False,
) -> Path:
    """Extract and cache normalized raw/enhanced DINOv2 descriptors."""
    output = descriptor_directory(cfg, method)
    output.mkdir(parents=True, exist_ok=True)
    descriptor_path = output / "descriptors.npy"
    manifest = pd.read_csv(manifest_path)
    if limit is not None:
        manifest = manifest.iloc[:limit].copy()
    manifest_digest = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps({
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": manifest_digest,
        "method": method, "model_size": cfg["images"]["model_size"],
        "pooling": cfg["model"]["pooling"], "rows": len(manifest),
    }, sort_keys=True).encode()).hexdigest()
    config_path = output / "descriptor_config.json"
    if descriptor_path.is_file() and config_path.is_file() and not overwrite:
        cached = json.loads(config_path.read_text(encoding="utf-8"))
        if cached.get("manifest_fingerprint") == fingerprint:
            return output
    enhancer = classical_enhancement if method == "enhanced_dinov2" else None
    dataset = ManifestImageDataset(manifest, int(cfg["images"]["model_size"]), enhancer)
    loader = DataLoader(
        dataset, batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"]["num_workers"]), shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = DINOv2Backbone(
        cfg["paths"]["torch_home"], cfg["model"]["pooling"], True, 0,
        int(cfg["model"]["backbone_dim"]),
    ).to(device).eval()
    collected: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _ in tqdm(loader, desc=f"encode {method}"):
            collected.append(backbone(images.to(device)).cpu().numpy())
    descriptors = normalize_descriptors(np.concatenate(collected))
    np.save(descriptor_path, descriptors)
    manifest.to_csv(output / "descriptor_metadata.csv", index=False)
    (output / "descriptor_config.json").write_text(json.dumps({
        "method": method, "backbone": cfg["model"]["backbone"],
        "dimension": int(descriptors.shape[1]), "manifest_fingerprint": fingerprint,
    }, indent=2), encoding="utf-8")
    if enhancer and len(manifest):
        import cv2
        original = cv2.imread(str(manifest.iloc[0]["image_path"]))
        if original is not None:
            cv2.imwrite(str(output / "enhancement_before.jpg"), original)
            cv2.imwrite(str(output / "enhancement_after.jpg"), enhancer(original))
    return output
