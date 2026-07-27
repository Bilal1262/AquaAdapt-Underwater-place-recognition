"""Trajectory plots."""

from pathlib import Path

import matplotlib.pyplot as plt

from aquaadapt.trajectory.tum import TumTrajectory


def plot_trajectory(trajectory: TumTrajectory, output: str | Path) -> Path:
    xyz = trajectory.translations
    figure = plt.figure(figsize=(14, 4))
    ax_xy = figure.add_subplot(131)
    ax_xz = figure.add_subplot(132)
    ax_3d = figure.add_subplot(133, projection="3d")
    ax_xy.plot(xyz[:, 0], xyz[:, 1]); ax_xy.set(xlabel="X [m]", ylabel="Y [m]", title="XY")
    ax_xz.plot(xyz[:, 0], xyz[:, 2]); ax_xz.set(xlabel="X [m]", ylabel="Z [m]", title="XZ")
    ax_3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2]); ax_3d.set_title("XYZ")
    for ax, dims in ((ax_xy, (0, 1)), (ax_xz, (0, 2))):
        ax.scatter(*xyz[0, list(dims)], marker="o", label="start")
        ax.scatter(*xyz[-1, list(dims)], marker="x", label="end")
        ax.legend()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(destination, dpi=160); plt.close(figure)
    return destination

