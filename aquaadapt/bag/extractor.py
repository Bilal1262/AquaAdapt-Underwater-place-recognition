"""Resumable timestamp-based image extraction."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader
from tqdm import tqdm

from aquaadapt.bag.image_decoder import decode_image
from aquaadapt.bag.inspect import inspect_bag
from aquaadapt.paths import topic_slug, trajectory_id, trajectory_root
from aquaadapt.trajectory.association import associate_timestamps, timestamp_compatibility
from aquaadapt.trajectory.tum import parse_tum

LOG = logging.getLogger(__name__)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False, encoding="utf-8") as handle:
        temp_path = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temp_path, path)


def _contact_sheet(paths: list[Path], output: Path, columns: int = 5) -> None:
    selected = [paths[index] for index in np.linspace(
        0, len(paths) - 1, min(20, len(paths)), dtype=int
    )] if paths else []
    thumbs: list[np.ndarray] = []
    for path in selected:
        image = cv2.imread(str(path))
        if image is not None:
            image = cv2.resize(image, (240, 135), interpolation=cv2.INTER_AREA)
            thumbs.append(image)
    if not thumbs:
        return
    blank = np.zeros_like(thumbs[0])
    while len(thumbs) % columns:
        thumbs.append(blank)
    rows = [np.hstack(thumbs[i:i + columns]) for i in range(0, len(thumbs), columns)]
    cv2.imwrite(str(output), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])


def _write_parquet_if_available(frame: pd.DataFrame, path: Path) -> bool:
    try:
        frame.to_parquet(path, index=False)
        return True
    except (ImportError, ValueError):
        LOG.warning("Parquet engine unavailable; metadata.csv remains the authoritative manifest")
        return False


def extract_images(cfg: dict[str, Any], quick: bool = False, limit: int | None = None, overwrite: bool = False) -> dict[str, Any]:
    """Extract a sampled camera stream and associate each frame with a TUM pose."""
    bag_path = Path(cfg["paths"]["bag"])
    processed = Path(cfg["paths"]["processed_root"])
    trajectory_dir = processed / trajectory_id(cfg)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    report_path = trajectory_dir / "bag_report.json"
    requested_topic = str(cfg["extraction"]["camera_topic"])
    if report_path.is_file() and requested_topic == "auto":
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = inspect_bag(bag_path, trajectory_dir, requested_topic)
    topic = str(report["selected_camera_topic"]) if requested_topic == "auto" else requested_topic
    root = trajectory_root(cfg, topic)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "metadata.csv"
    failed_path = root / "failed_frames.csv"

    if overwrite and metadata_path.exists():
        LOG.warning("Overwrite requested: metadata will be rebuilt; existing images are replaced only as sampled")
        previous = pd.DataFrame()
    elif metadata_path.is_file():
        previous = pd.read_csv(metadata_path)
    else:
        previous = pd.DataFrame()
    existing = set(previous["bag_timestamp_ns"].astype(np.int64).tolist()) if not previous.empty else set()

    rate = 1.0 if quick else float(cfg["extraction"]["sample_rate_hz"])
    max_frames = 300 if quick else cfg["extraction"].get("max_frames")
    if limit is not None:
        max_frames = limit if max_frames is None else min(int(max_frames), int(limit))
    interval_ns = int(round(1e9 / rate))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    first_sample_ns: int | None = None
    last_sample_ns: int | None = None
    started = time.perf_counter()

    with AnyReader([bag_path]) as reader:
        connections = [connection for connection in reader.connections if connection.topic == topic]
        if not connections:
            raise ValueError(f"Camera topic is absent from bag: {topic}")
        source_type = connections[0].msgtype
        progress = tqdm(total=max_frames, desc=f"extract {topic}", unit="frame") if max_frames else tqdm(desc=f"extract {topic}", unit="frame")
        for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
            timestamp_ns = int(timestamp_ns)
            if last_sample_ns is not None and timestamp_ns - last_sample_ns < interval_ns:
                continue
            if first_sample_ns is None:
                first_sample_ns = timestamp_ns
            last_sample_ns = timestamp_ns
            image_path = images_dir / f"{timestamp_ns}.jpg"
            if (
                not overwrite and bool(cfg["extraction"].get("resume", True))
                and timestamp_ns in existing and image_path.is_file() and image_path.stat().st_size > 0
            ):
                if max_frames is not None and len(rows) + len(existing) >= int(max_frames):
                    break
                continue
            try:
                message = reader.deserialize(rawdata, connection.msgtype)
                decoded = decode_image(message, connection.msgtype)
                ok = cv2.imwrite(
                    str(image_path), decoded.image,
                    [cv2.IMWRITE_JPEG_QUALITY, int(cfg["extraction"]["jpeg_quality"])],
                )
                if not ok:
                    raise ValueError("cv2.imwrite returned false")
                rows.append({
                    "trajectory_id": trajectory_id(cfg), "camera_topic": topic,
                    "frame_index": 0, "bag_timestamp_ns": timestamp_ns,
                    "timestamp_sec": timestamp_ns / 1e9,
                    "relative_time_sec": (timestamp_ns - first_sample_ns) / 1e9,
                    "image_path": str(image_path), "width": int(decoded.image.shape[1]),
                    "height": int(decoded.image.shape[0]), "original_encoding": decoded.encoding,
                    "source_message_type": source_type,
                })
                progress.update(1)
            except (ValueError, TypeError, RuntimeError, OSError) as exc:
                LOG.warning("Skipping frame %d: %s", timestamp_ns, exc)
                failures.append({"bag_timestamp_ns": timestamp_ns, "error": str(exc)})
            total_considered = len(rows) + (len(previous) if not previous.empty else 0)
            if max_frames is not None and total_considered >= int(max_frames):
                break
        progress.close()

    combined = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True)
    if combined.empty:
        raise RuntimeError(f"No images were extracted from {topic}")
    combined = combined.drop_duplicates("bag_timestamp_ns", keep="last").sort_values("bag_timestamp_ns").reset_index(drop=True)
    combined["frame_index"] = np.arange(len(combined))
    combined["relative_time_sec"] = combined["timestamp_sec"] - float(combined["timestamp_sec"].iloc[0])

    trajectory = parse_tum(cfg["paths"]["tum"])
    compatible, ranges = timestamp_compatibility(combined["timestamp_sec"].to_numpy(), trajectory.timestamps)
    offset = float(cfg["extraction"].get("timestamp_offset_sec", 0.0))
    if not compatible and offset == 0:
        LOG.warning("Timestamp ranges appear incompatible (%s); no arbitrary offset will be applied", ranges)
    association = associate_timestamps(
        combined["timestamp_sec"].to_numpy(), trajectory,
        float(cfg["extraction"]["pose_max_time_difference_sec"]), offset,
    )
    pose_idx = association.indices
    combined["pose_valid"] = association.valid
    combined["pose_timestamp_sec"] = trajectory.timestamps[pose_idx]
    combined["pose_time_difference_sec"] = association.time_differences
    for column, values in zip(
        ("tx", "ty", "tz", "qx", "qy", "qz", "qw"),
        np.hstack((trajectory.translations[pose_idx], trajectory.quaternions[pose_idx])).T,
    ):
        combined[column] = values
    combined.loc[~combined["pose_valid"], ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]] = np.nan
    if "split" not in combined:
        combined["split"] = ""
    _atomic_csv(combined, metadata_path)
    parquet = _write_parquet_if_available(combined, root / "metadata.parquet")
    pd.DataFrame(failures, columns=["bag_timestamp_ns", "error"]).to_csv(failed_path, index=False)
    _contact_sheet([Path(path) for path in combined["image_path"]], root / "sample_contact_sheet.jpg")
    summary = {
        "trajectory_id": trajectory_id(cfg), "camera_topic": topic,
        "output_directory": str(root), "sample_rate_hz": rate,
        "frame_count": int(len(combined)), "new_frames": len(rows),
        "failed_frame_count": len(failures), "metadata_parquet_written": parquet,
        "elapsed_sec": time.perf_counter() - started,
        "extraction_rate_fps": len(rows) / max(time.perf_counter() - started, 1e-9),
        "association": association.summary(), "timestamp_ranges": ranges,
        "timestamp_offset_sec": offset,
    }
    (root / "extraction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
