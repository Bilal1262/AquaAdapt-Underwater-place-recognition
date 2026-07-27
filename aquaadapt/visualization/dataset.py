"""Dataset timestamp, association, and evaluation-sampling plots."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def plot_dataset_diagnostics(frame: pd.DataFrame, output_dir: str | Path, cfg: dict[str, Any]) -> None:
    """Save required dataset inspection plots from a completed manifest."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(frame["relative_time_sec"], bins=min(50, max(10, len(frame) // 5)))
    ax.set(xlabel="relative image timestamp [s]", ylabel="frame count", title="Image timestamp distribution")
    fig.tight_layout(); fig.savefig(output / "image_timestamp_distribution.png", dpi=160); plt.close(fig)

    valid_errors = frame.loc[frame["pose_valid"].astype(bool), "pose_time_difference_sec"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(valid_errors, bins=30)
    ax.set(xlabel="nearest-pose absolute time error [s]", ylabel="frame count", title="Pose association errors")
    fig.tight_layout(); fig.savefig(output / "pose_association_errors.png", dpi=160); plt.close(fig)

    test = frame.loc[(frame["split"] == "test") & frame["pose_valid"].astype(bool)]
    query_stride = max(1, int(cfg["evaluation"]["query_stride"]))
    database_stride = max(1, int(cfg["evaluation"]["database_stride"]))
    query = test.iloc[1::query_stride]
    database = test.iloc[::database_stride]
    fig, ax = plt.subplots(figsize=(7, 6))
    if not database.empty:
        ax.scatter(database["tx"], database["ty"], s=18, label="database")
    if not query.empty:
        ax.scatter(query["tx"], query["ty"], s=18, marker="x", label="query")
    ax.set(xlabel="X [m]", ylabel="Y [m]", title="Sampled test query/database poses")
    ax.legend(); fig.tight_layout()
    fig.savefig(output / "query_database_poses.png", dpi=160); plt.close(fig)
