"""Small helpers around rosbags.AnyReader."""

from pathlib import Path

from rosbags.highlevel import AnyReader


def open_bag(path: str | Path) -> AnyReader:
    """Return an unopened AnyReader for a single bag path."""
    bag = Path(path)
    if not bag.is_file():
        raise FileNotFoundError(f"ROS bag not found: {bag}")
    return AnyReader([bag])

