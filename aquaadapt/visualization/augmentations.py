"""Augmentation contact sheets."""

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from aquaadapt.augmentations.pipeline import apply_corruption


def visualize_augmentations(manifest: str | Path, output: str | Path, cfg: dict[str, Any]) -> Path:
    frame = pd.read_csv(manifest)
    if frame.empty:
        raise ValueError("Cannot visualize augmentations from an empty manifest")
    image = cv2.imread(str(frame.iloc[len(frame) // 2]["image_path"]))
    if image is None:
        raise FileNotFoundError(frame.iloc[len(frame) // 2]["image_path"])
    corruptions = list(cfg["robustness"]["corruptions"])
    levels = [0, 1, 2, 3]
    figure, axes = plt.subplots(len(corruptions), len(levels) + 1, figsize=(15, 3 * len(corruptions)))
    for row, name in enumerate(corruptions):
        axes[row, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title(f"{name}\noriginal")
        axes[row, 0].axis("off")
        for column, level in enumerate(levels, 1):
            degraded = apply_corruption(image, name, level, int(cfg["project"]["seed"]))
            axes[row, column].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
            axes[row, column].set_title(f"severity {level}")
            axes[row, column].axis("off")
    figure.tight_layout()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination

