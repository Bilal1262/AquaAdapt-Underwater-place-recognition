"""Decode ROS Image and CompressedImage messages without ROS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DecodedImage:
    """Visible OpenCV image and source encoding."""

    image: np.ndarray
    encoding: str


_RAW_FORMATS: dict[str, tuple[np.dtype, int]] = {
    "bgr8": (np.dtype("u1"), 3),
    "rgb8": (np.dtype("u1"), 3),
    "mono8": (np.dtype("u1"), 1),
    "bgra8": (np.dtype("u1"), 4),
    "rgba8": (np.dtype("u1"), 4),
    "bayer_rggb8": (np.dtype("u1"), 1),
    "bayer_bggr8": (np.dtype("u1"), 1),
    "bayer_gbrg8": (np.dtype("u1"), 1),
    "bayer_grbg8": (np.dtype("u1"), 1),
    "mono16": (np.dtype("u2"), 1),
    "16uc1": (np.dtype("u2"), 1),
    "32fc1": (np.dtype("f4"), 1),
}

_BAYER_CODES = {
    "bayer_rggb8": cv2.COLOR_BAYER_BG2BGR,
    "bayer_bggr8": cv2.COLOR_BAYER_RG2BGR,
    "bayer_gbrg8": cv2.COLOR_BAYER_GR2BGR,
    "bayer_grbg8": cv2.COLOR_BAYER_GB2BGR,
}


def _visible_8bit(array: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(finite, (1, 99)) if finite.size else (0.0, 0.0)
    if hi <= lo:
        return np.zeros(finite.shape, dtype=np.uint8)
    return np.clip((finite - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def decode_raw_image(message: Any) -> DecodedImage:
    """Decode a raw ROS image while respecting endianness and padded rows."""
    encoding = str(message.encoding).lower()
    if encoding not in _RAW_FORMATS:
        raise ValueError(f"Unsupported raw image encoding: {message.encoding!r}")
    dtype, channels = _RAW_FORMATS[encoding]
    height, width, step = int(message.height), int(message.width), int(message.step)
    if height <= 0 or width <= 0 or step <= 0:
        raise ValueError(f"Invalid image geometry: {width}x{height}, step={step}")
    itemsize = dtype.itemsize
    required = height * step
    payload = memoryview(message.data)
    if len(payload) < required:
        raise ValueError(f"Image payload is short: expected {required}, got {len(payload)}")
    byteorder = ">" if bool(message.is_bigendian) else "<"
    source_dtype = dtype.newbyteorder(byteorder) if itemsize > 1 else dtype
    row_values = step // itemsize
    wanted_values = width * channels
    if row_values < wanted_values:
        raise ValueError(f"Image step {step} is smaller than packed row size {wanted_values * itemsize}")
    raw = np.frombuffer(payload[:required], dtype=source_dtype).reshape(height, row_values)
    pixels = raw[:, :wanted_values].reshape(height, width, channels) if channels > 1 else raw[:, :width]
    if itemsize > 1:
        pixels = pixels.astype(dtype, copy=False)

    if encoding == "rgb8":
        output = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    elif encoding == "rgba8":
        output = cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
    elif encoding == "bgra8":
        output = cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    elif encoding in _BAYER_CODES:
        output = cv2.cvtColor(pixels, _BAYER_CODES[encoding])
    elif encoding in {"mono16", "16uc1", "32fc1"}:
        output = _visible_8bit(pixels)
    else:
        output = np.ascontiguousarray(pixels)
    return DecodedImage(output, encoding)


def decode_compressed_image(message: Any) -> DecodedImage:
    """Decode a ROS compressed image with OpenCV."""
    buffer = np.frombuffer(message.data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("OpenCV could not decode compressed image payload")
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    encoding = str(getattr(message, "format", "compressed")) or "compressed"
    return DecodedImage(image, encoding)


def decode_image(message: Any, msgtype: str) -> DecodedImage:
    """Dispatch raw or compressed ROS image decoding."""
    return decode_compressed_image(message) if "CompressedImage" in msgtype else decode_raw_image(message)

