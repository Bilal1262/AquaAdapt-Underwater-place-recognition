"""Ablation result assembly without invented values."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ABLATIONS = [
    ("Raw DINOv2", False, False, False, 0),
    ("ordinary_augmentations", True, False, False, 0),
    ("underwater_augmentations", True, False, True, 0),
    ("AquaAdapt_temporal", True, True, True, 0),
    ("AquaAdapt_unfreeze1", True, True, True, 1),
]


def assemble_ablations(cfg: dict[str, Any]) -> Path:
    output = Path("results") / str(cfg["project"]["run_name"])
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name, same, temporal, underwater, blocks in ABLATIONS:
        result_path = output / "ablations" / name / "evaluation.json"
        metrics = json.loads(result_path.read_text()) if result_path.is_file() else {}
        rows.append({
            "method": name, "same-frame positive": same, "temporal positive": temporal,
            "underwater augmentation": underwater, "unfrozen blocks": blocks,
            "trainable parameters": metrics.get("trainable_parameters", "NA"),
            "Recall@1": metrics.get("Recall@1", "NA"), "Recall@5": metrics.get("Recall@5", "NA"),
            "MRR": metrics.get("MRR", "NA"), "FPS": metrics.get("FPS", "NA"),
        })
    destination = output / "ablation_results.csv"
    pd.DataFrame(rows).to_csv(destination, index=False)
    return destination

