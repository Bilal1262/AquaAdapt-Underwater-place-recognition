"""Fast ROS1 bag inventory and RGB-camera selection."""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from rosbags.highlevel import AnyReader

from aquaadapt.bag.image_decoder import decode_image

LOG = logging.getLogger(__name__)
IMAGE_TYPES = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}
REJECT_WORDS = ("depth", "disparity", "mask", "segmentation", "thermal")


def is_image_connection(connection: Any) -> bool:
    return connection.msgtype in IMAGE_TYPES


def _connection_times(connection: Any) -> tuple[int | None, int | None]:
    indexes = getattr(connection.owner, "indexes", {}).get(connection.id, [])
    if not indexes:
        return None, None
    return int(indexes[0].time), int(indexes[-1].time)


def _inspect_image_samples(reader: AnyReader, connection: Any, count: int = 3) -> dict[str, Any]:
    info: dict[str, Any] = {"valid_samples": 0}
    serialized_sizes: list[int] = []
    for conn, _, rawdata in reader.messages(connections=[connection]):
        serialized_sizes.append(len(rawdata))
        try:
            msg = reader.deserialize(rawdata, conn.msgtype)
            decoded = decode_image(msg, conn.msgtype)
            info.update({
                "width": int(decoded.image.shape[1]),
                "height": int(decoded.image.shape[0]),
                "encoding": str(getattr(msg, "encoding", decoded.encoding)),
                "compression_format": str(getattr(msg, "format", "")),
            })
            info["valid_samples"] += 1
        except (ValueError, TypeError, RuntimeError) as exc:
            info.setdefault("sample_errors", []).append(str(exc))
        if info["valid_samples"] >= count:
            break
    if serialized_sizes:
        info["average_serialized_message_bytes"] = float(sum(serialized_sizes) / len(serialized_sizes))
    return info


def select_rgb_topic(report: dict[str, Any], override: str = "auto") -> tuple[str, str]:
    """Select the highest-count plausible visible camera stream."""
    images = report.get("image_topics", [])
    if override != "auto":
        matches = [row for row in images if row["topic"] == override]
        if not matches:
            raise ValueError(f"Requested camera topic {override!r} is not a decodable image topic")
        return override, "explicit camera-topic override"
    plausible = [
        row for row in images
        if not any(word in row["topic"].lower() for word in REJECT_WORDS)
        and int(row.get("valid_samples", 0)) > 0
    ]
    if not plausible:
        candidates = ", ".join(row["topic"] for row in images) or "none"
        raise ValueError(f"No plausible RGB image topic found. Image candidates: {candidates}")
    plausible.sort(
        key=lambda row: (
            1 if str(row.get("encoding", "")).lower() in {"rgb8", "bgr8", "rgba8", "bgra8"} else 0,
            0 if "compressed" in row["message_type"].lower() else 1,
            int(row["message_count"]),
        ),
        reverse=True,
    )
    selected = plausible[0]
    reason = (
        f"plausible visible image topic with {selected['message_count']} messages and "
        f"{selected['valid_samples']} valid decoded samples"
    )
    return str(selected["topic"]), reason


def inspect_bag(
    bag_path: str | Path,
    output_dir: str | Path | None = None,
    camera_topic: str = "auto",
) -> dict[str, Any]:
    """Inspect a bag and optionally persist JSON, CSV, and text reports."""
    bag = Path(bag_path)
    if not bag.is_file():
        raise FileNotFoundError(f"ROS bag not found: {bag}")
    rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    with AnyReader([bag]) as reader:
        for connection in reader.connections:
            first, last = _connection_times(connection)
            duration = ((last - first) / 1e9) if first is not None and last is not None else 0.0
            row = {
                "topic": connection.topic,
                "message_type": connection.msgtype,
                "message_count": int(connection.msgcount),
                "first_timestamp_ns": first,
                "last_timestamp_ns": last,
                "first_timestamp_sec": first / 1e9 if first else None,
                "last_timestamp_sec": last / 1e9 if last else None,
                "approx_frequency_hz": (connection.msgcount - 1) / duration if duration > 0 else None,
                "serialized_bytes_estimate": None,
            }
            if is_image_connection(connection):
                sample_info = _inspect_image_samples(reader, connection)
                if sample_info.get("average_serialized_message_bytes"):
                    row["serialized_bytes_estimate"] = int(
                        float(sample_info["average_serialized_message_bytes"]) * connection.msgcount
                    )
                image_row = {**row, **sample_info}
                image_rows.append(image_row)
            rows.append(row)
        report: dict[str, Any] = {
            "bag_path": str(bag), "bag_size_bytes": bag.stat().st_size,
            "start_timestamp_ns": int(reader.start_time), "end_timestamp_ns": int(reader.end_time),
            "duration_sec": float(reader.duration / 1e9), "connections": rows,
            "image_topics": image_rows,
        }
    selected, reason = select_rgb_topic(report, camera_topic)
    report["selected_camera_topic"] = selected
    report["selection_reason"] = reason
    if output_dir:
        save_bag_report(report, output_dir)
    return report


def save_bag_report(report: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "bag_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = report["connections"]
    with (output / "bag_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["topic"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"Bag: {report['bag_path']}",
        f"Size: {report['bag_size_bytes']} bytes",
        f"Duration: {report['duration_sec']:.3f} s",
        f"Selected RGB topic: {report['selected_camera_topic']}",
        f"Reason: {report['selection_reason']}",
        "",
        "topic | type | count | frequency_hz | first_ns | last_ns",
    ]
    for row in rows:
        hz = row["approx_frequency_hz"]
        frequency = f"{hz:.3f}" if hz is not None else "NA"
        lines.append(
            f"{row['topic']} | {row['message_type']} | {row['message_count']} | "
            f"{frequency} | {row['first_timestamp_ns']} | {row['last_timestamp_ns']}"
        )
    (output / "bag_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
