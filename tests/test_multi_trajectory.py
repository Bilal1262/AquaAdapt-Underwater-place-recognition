from pathlib import Path

import pandas as pd
import torch

from aquaadapt.data.manifest import build_combined_manifest, combined_manifest_path
from aquaadapt.training.trainer import _positive_mask


def _cfg(tmp_path: Path, manifests: list[Path]) -> dict:
    return {
        "project": {
            "run_name": "combined",
            "evaluation_regime": "multi_trajectory_training",
        },
        "paths": {
            "processed_root": str(tmp_path / "processed"),
            "training_manifests": [str(path) for path in manifests],
        },
        "splits": {
            "policy": "chronological",
            "train": 0.5,
            "guard1": 0.0,
            "validation": 0.5,
            "guard2": 0.0,
            "test": 0.0,
        },
        "training": {
            "use_temporal_positive": True,
            "temporal_positive_window_sec": 1.0,
            "temporal_positive_max_pose_distance_m": 1.0,
        },
        "evaluation": {"query_stride": 2, "database_stride": 2},
    }


def test_combined_manifest_preserves_trajectory_boundaries(tmp_path: Path) -> None:
    manifests = []
    for trajectory in ("mclab_1", "mclab_2"):
        path = tmp_path / f"{trajectory}.csv"
        pd.DataFrame(
            {
                "trajectory_id": [trajectory] * 4,
                "timestamp_sec": [0.0, 1.0, 2.0, 3.0],
                "relative_time_sec": [0.0, 1.0, 2.0, 3.0],
                "image_path": [f"/{trajectory}/{index}.jpg" for index in range(4)],
                "pose_valid": [True] * 4,
                "pose_time_difference_sec": [0.01] * 4,
                "tx": [0.0, 1.0, 2.0, 3.0],
                "ty": [0.0] * 4,
            }
        ).to_csv(path, index=False)
        manifests.append(path)

    cfg = _cfg(tmp_path, manifests)
    output, summary = build_combined_manifest(cfg, manifests)

    assert output == combined_manifest_path(cfg)
    assert summary["trajectory_count"] == 2
    combined = pd.read_csv(output)
    assert combined.groupby("trajectory_id")["split"].apply(list).to_dict() == {
        "mclab_1": ["train", "train", "validation", "validation"],
        "mclab_2": ["train", "train", "validation", "validation"],
    }


def test_temporal_positives_do_not_cross_trajectories() -> None:
    manifest = pd.DataFrame(
        {
            "trajectory_id": ["mclab_1", "mclab_2"],
            "timestamp_sec": [10.0, 10.1],
            "pose_valid": [True, True],
            "tx": [0.0, 0.0],
            "ty": [0.0, 0.0],
            "tz": [0.0, 0.0],
        }
    )
    cfg = {
        "training": {
            "use_temporal_positive": True,
            "temporal_positive_window_sec": 1.0,
            "temporal_positive_max_pose_distance_m": 1.0,
        }
    }

    mask = _positive_mask(
        2, torch.device("cpu"), torch.tensor([0, 1]), manifest, cfg
    )

    assert mask[0, 1].item() is False
    assert mask[0, 3].item() is False
    assert mask[0, 2].item() is True


def test_spatial_revisit_is_positive_after_temporal_exclusion() -> None:
    manifest = pd.DataFrame(
        {
            "trajectory_id": ["mclab_1", "mclab_1", "mclab_2"],
            "timestamp_sec": [0.0, 30.0, 30.0],
            "pose_valid": [True, True, True],
            "tx": [0.0, 0.5, 0.5],
            "ty": [0.0, 0.0, 0.0],
            "tz": [0.0, 0.0, 0.0],
        }
    )
    cfg = {
        "training": {
            "use_temporal_positive": False,
            "use_spatial_positives": True,
            "spatial_positive_radius_m": 1.5,
            "spatial_positive_temporal_exclusion_sec": 10.0,
        }
    }
    mask = _positive_mask(
        3, torch.device("cpu"), torch.tensor([0, 1, 2]), manifest, cfg
    )

    assert mask[0, 1].item() is True
    assert mask[0, 4].item() is True
    assert mask[0, 2].item() is False
    assert mask[0, 5].item() is False
