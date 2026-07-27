"""Output path conventions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def topic_slug(topic: str) -> str:
    """Convert a ROS topic to a stable directory-safe name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", topic.strip("/")) or "camera"


def trajectory_id(cfg: dict[str, Any]) -> str:
    return Path(cfg["paths"]["bag"]).stem


def trajectory_root(cfg: dict[str, Any], topic: str) -> Path:
    return Path(cfg["paths"]["processed_root"]) / trajectory_id(cfg) / topic_slug(topic)


def results_root(cfg: dict[str, Any]) -> Path:
    return Path("results") / str(cfg["project"]["run_name"])

