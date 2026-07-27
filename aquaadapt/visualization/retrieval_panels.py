"""Retrieval panel generation."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def retrieval_panel(query: str, retrieved: list[str], labels: list[str], output: str | Path) -> None:
    paths = [query, *retrieved]
    titles = ["query", *labels]
    fig, axes = plt.subplots(1, len(paths), figsize=(3 * len(paths), 3))
    for ax, path, title in zip(axes, paths, titles):
        image = cv2.imread(path)
        if image is not None:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)
