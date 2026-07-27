#!/usr/bin/env python3
"""Optional ROS Noetic/cv_bridge fallback for unusual image encodings.

Run only in a sourced ROS Noetic environment:
  python3 scripts/ros_noetic_extract_fallback.py BAG TOPIC OUTPUT_DIR --max-frames 300
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

try:
    import rosbag
    from cv_bridge import CvBridge
except ImportError as exc:
    raise SystemExit(
        "ROS Noetic fallback dependencies are unavailable. Source /opt/ros/noetic/setup.bash "
        "and install ros-noetic-cv-bridge, or use the default `aquaadapt extract` command."
    ) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag"); parser.add_argument("topic"); parser.add_argument("output_dir")
    parser.add_argument("--sample-rate-hz", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    output = Path(args.output_dir); images = output / "images"; images.mkdir(parents=True, exist_ok=True)
    bridge = CvBridge(); rows = []; last_ns = None; interval = int(1e9 / args.sample_rate_hz)
    with rosbag.Bag(args.bag, "r") as bag:
        for index, (_, message, timestamp) in enumerate(bag.read_messages(topics=[args.topic])):
            timestamp_ns = int(timestamp.to_nsec())
            if last_ns is not None and timestamp_ns - last_ns < interval:
                continue
            last_ns = timestamp_ns
            image = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            path = images / f"{timestamp_ns}.jpg"
            if not cv2.imwrite(str(path), image):
                continue
            rows.append({"camera_topic": args.topic, "frame_index": len(rows), "bag_timestamp_ns": timestamp_ns,
                         "timestamp_sec": timestamp_ns / 1e9, "image_path": str(path),
                         "width": image.shape[1], "height": image.shape[0],
                         "original_encoding": message.encoding, "source_message_type": message._type})
            if args.max_frames and len(rows) >= args.max_frames:
                break
    with (output / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
