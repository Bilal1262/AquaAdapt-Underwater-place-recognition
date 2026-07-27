#!/usr/bin/env python3
"""Generate a publication-ready AquaAdapt architecture using real project imagery."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from aquaadapt.augmentations.pipeline import apply_corruption


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets"
DATA_ROOT = Path(
    os.environ.get(
        "AQUAADAPT_DATA_ROOT",
        "/mnt/windows/datasets/ntnu_underwater",
    )
)

MCLAB1 = DATA_ROOT / (
    "processed/mclab_1/alphasense_driver_ros_cam0/"
    "images/1725639605740139511.jpg"
)
MCLAB2 = DATA_ROOT / (
    "processed/mclab_2/alphasense_driver_ros_cam0/"
    "images/1725640099934219906.jpg"
)
FJORD1 = DATA_ROOT / (
    "processed/fjord_1/alphasense_driver_ros_cam0/"
    "images/1700604887984044160.jpg"
)
FJORD2 = DATA_ROOT / (
    "processed/fjord_2/alphasense_driver_ros_cam0/"
    "images/1700601715036241088.jpg"
)
RETRIEVAL = ROOT / "docs/assets/fjord2_haze_retrieval_example.png"

NAVY = "#10233f"
BLUE = "#1768d2"
BLUE_LIGHT = "#edf5ff"
TEAL = "#078f88"
TEAL_LIGHT = "#edfafa"
VIOLET = "#7157c8"
VIOLET_LIGHT = "#f5f1ff"
INK = "#172033"
MUTED = "#526174"
GREEN = "#159447"
RED = "#d83a3a"
LINE = "#25364d"
WHITE = "#ffffff"


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read required architecture asset: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def crop_fill(image: np.ndarray, aspect: float) -> np.ndarray:
    """Center-crop an image to width/height == aspect."""
    height, width = image.shape[:2]
    current = width / height
    if current > aspect:
        target = int(round(height * aspect))
        left = max(0, (width - target) // 2)
        return image[:, left : left + target]
    target = int(round(width / aspect))
    top = max(0, (height - target) // 2)
    return image[top : top + target]


def rounded(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str = WHITE,
    edge: str = BLUE,
    linewidth: float = 1.3,
    radius: float = 0.012,
    linestyle: str = "-",
    zorder: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=linestyle,
        transform=ax.transAxes,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = LINE,
    width: float = 1.5,
    dashed: bool = False,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=width,
            color=color,
            linestyle="--" if dashed else "-",
            connectionstyle=connectionstyle,
            zorder=5,
        )
    )


def section(
    ax: plt.Axes,
    number: str,
    title: str,
    y: float,
    height: float,
    color: str,
    light: str,
) -> None:
    rounded(ax, 0.018, y, 0.964, height, face=light, edge=color, linewidth=0.9, radius=0.014)
    rounded(ax, 0.027, y + 0.018, 0.095, height - 0.036, face=WHITE, edge=color, linewidth=1.0)
    ax.text(
        0.074,
        y + height - 0.047,
        number,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=WHITE,
        bbox=dict(boxstyle="circle,pad=0.38", facecolor=color, edgecolor="none"),
        zorder=8,
    )
    ax.text(
        0.074,
        y + height * 0.43,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=color,
        linespacing=1.18,
        zorder=8,
    )


def image_card(
    ax: plt.Axes,
    image: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
    color: str,
    *,
    dashed: bool = False,
) -> None:
    outer = rounded(
        ax,
        x,
        y,
        width,
        height,
        face=WHITE,
        edge=color,
        linewidth=1.35,
        linestyle="--" if dashed else "-",
        radius=0.009,
        zorder=2,
    )
    label_height = min(0.041, height * 0.35)
    pad = 0.005
    extent = (x + pad, x + width - pad, y + label_height, y + height - pad)
    target_aspect = (width - 2 * pad) / max(height - label_height - pad, 1e-4)
    artist = ax.imshow(
        crop_fill(image, target_aspect),
        extent=extent,
        transform=ax.transAxes,
        interpolation="lanczos",
        zorder=3,
        aspect="auto",
    )
    artist.set_clip_path(outer)
    ax.text(
        x + 0.009,
        y + label_height * 0.66,
        title,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.1,
        fontweight="bold",
        color=INK,
        zorder=6,
    )
    ax.text(
        x + width - 0.008,
        y + label_height * 0.33,
        subtitle,
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=6.2,
        color=MUTED,
        zorder=6,
    )


def text_card(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    color: str,
    *,
    badge: str | None = None,
) -> None:
    rounded(ax, x, y, width, height, face=WHITE, edge=color, linewidth=1.35, radius=0.009)
    title_size = 7.4 if badge and len(title) > 16 else 9.0
    ax.text(
        x + 0.012,
        y + height - 0.023,
        title,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=INK,
        zorder=6,
    )
    ax.text(
        x + 0.012,
        y + height - 0.048,
        body,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color=MUTED,
        linespacing=1.25,
        zorder=6,
    )
    if badge:
        ax.text(
            x + width - 0.011,
            y + height - 0.022,
            badge,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=6.2,
            fontweight="bold",
            color=WHITE,
            bbox=dict(boxstyle="round,pad=0.26", facecolor=color, edgecolor="none"),
            zorder=7,
        )


def split_manifest_card(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    rounded(ax, x, y, width, height, face=WHITE, edge=BLUE, linewidth=1.35, radius=0.009)
    ax.text(
        x + 0.012,
        y + height - 0.025,
        "Combined manifest",
        transform=ax.transAxes,
        fontsize=8.6,
        fontweight="bold",
        color=INK,
        va="center",
    )
    total = 4998
    values = [3798, 200, 1000]
    colors = [BLUE, "#91a0b5", TEAL]
    labels = ["train 3,798", "guard 200", "val 1,000"]
    left = x + 0.012
    usable = width - 0.024
    bar_y = y + 0.041
    for value, color in zip(values, colors):
        segment = usable * value / total
        ax.add_patch(
            FancyBboxPatch(
                (left, bar_y),
                segment,
                0.014,
                boxstyle="round,pad=0,rounding_size=0.004",
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
                zorder=5,
            )
        )
        left += segment
    ax.text(
        x + 0.012,
        y + 0.020,
        " • ".join(labels),
        transform=ax.transAxes,
        fontsize=6.0,
        color=MUTED,
        va="center",
    )
    ax.text(
        x + width - 0.011,
        y + height - 0.025,
        "4,998 frames",
        transform=ax.transAxes,
        fontsize=6.3,
        fontweight="bold",
        color=BLUE,
        ha="right",
        va="center",
    )


def descriptor_bars(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    values = np.array([0.20, 0.68, 0.38, 0.89, 0.54, 0.31, 0.76, 0.46, 0.96, 0.58])
    gap = width * 0.025
    bar_w = (width - gap * (len(values) - 1)) / len(values)
    for index, value in enumerate(values):
        bar_h = height * float(value)
        ax.add_patch(
            FancyBboxPatch(
                (x + index * (bar_w + gap), y),
                bar_w,
                bar_h,
                boxstyle="round,pad=0,rounding_size=0.002",
                transform=ax.transAxes,
                facecolor=TEAL if index % 2 else BLUE,
                edgecolor="none",
                zorder=6,
            )
        )


def retrieval_triptych(
    ax: plt.Axes,
    panel: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    rounded(ax, x, y, width, height, face=WHITE, edge=VIOLET, linewidth=1.25, radius=0.009)
    ax.text(
        x + 0.012,
        y + height - 0.021,
        "Held-out Fjord2 example · haze severity 2",
        transform=ax.transAxes,
        fontsize=7.8,
        fontweight="bold",
        color=INK,
        va="center",
    )
    # Exact native-resolution crops from the generated result:
    # corrupted query, raw top-1, and AquaAdapt top-1.
    crops = [
        panel[210:531, 20:446],
        panel[210:531, 468:896],
        panel[546:867, 468:896],
    ]
    captions = [
        ("Corrupted query", "test input", VIOLET),
        ("Raw DINOv2", "14.11 m · incorrect", RED),
        ("AquaAdapt V2", "0.82 m · positive", GREEN),
    ]
    pad = 0.012
    gap = 0.010
    card_w = (width - 2 * pad - 2 * gap) / 3
    image_y = y + 0.016
    image_h = height - 0.052
    for index, (crop, (title, sub, color)) in enumerate(zip(crops, captions)):
        left = x + pad + index * (card_w + gap)
        extent = (left, left + card_w, image_y, image_y + image_h)
        ax.imshow(
            crop_fill(crop, card_w / image_h),
            extent=extent,
            transform=ax.transAxes,
            cmap="gray",
            interpolation="lanczos",
            zorder=4,
            aspect="auto",
        )
        ax.add_patch(
            FancyBboxPatch(
                (left, image_y),
                card_w,
                image_h,
                boxstyle="round,pad=0,rounding_size=0.004",
                transform=ax.transAxes,
                facecolor="none",
                edgecolor=color,
                linewidth=1.8,
                zorder=5,
            )
        )
        ax.text(
            left + 0.004,
            image_y + 0.004,
            f"{title}\n{sub}",
            transform=ax.transAxes,
            fontsize=5.7,
            fontweight="bold",
            color=WHITE,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#07111ccc", edgecolor="none"),
            zorder=7,
        )


def robustness_card(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    rounded(ax, x, y, width, height, face=WHITE, edge=VIOLET, linewidth=1.25, radius=0.009)
    ax.text(
        x + 0.012,
        y + height - 0.021,
        "Held-out robustness · Fjord2",
        transform=ax.transAxes,
        fontsize=7.8,
        fontweight="bold",
        color=INK,
        va="center",
    )
    groups = [
        ("clean", 0.3534, 0.3564),
        ("macro · severity 3", 0.2812, 0.3353),
    ]
    base_y = y + 0.024
    max_h = height - 0.066
    starts = [x + 0.035, x + width * 0.52]
    bar_w = width * 0.075
    for start, (label, raw, aqua) in zip(starts, groups):
        for offset, value, color in ((0.0, raw, "#8090a5"), (bar_w * 1.25, aqua, TEAL)):
            ax.add_patch(
                FancyBboxPatch(
                    (start + offset, base_y),
                    bar_w,
                    max_h * value,
                    boxstyle="round,pad=0,rounding_size=0.003",
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor="none",
                    zorder=5,
                )
            )
            ax.text(
                start + offset + bar_w / 2,
                base_y + max_h * value + 0.005,
                f"{value * 100:.1f}",
                transform=ax.transAxes,
                fontsize=6.0,
                fontweight="bold",
                color=color,
                ha="center",
                va="bottom",
            )
        ax.text(
            start + bar_w,
            base_y - 0.010,
            label,
            transform=ax.transAxes,
            fontsize=5.8,
            color=MUTED,
            ha="center",
            va="top",
        )
    ax.text(
        x + width - 0.010,
        y + 0.013,
        "Recall@1  ·  raw  /  AquaAdapt",
        transform=ax.transAxes,
        fontsize=5.8,
        color=MUTED,
        ha="right",
    )


def main() -> None:
    for path in (MCLAB1, MCLAB2, FJORD1, FJORD2, RETRIEVAL):
        if not path.is_file():
            raise FileNotFoundError(path)

    mclab1 = read_rgb(MCLAB1)
    mclab2 = read_rgb(MCLAB2)
    fjord1 = read_rgb(FJORD1)
    fjord2 = read_rgb(FJORD2)
    retrieval = read_rgb(RETRIEVAL)
    clean_bgr = cv2.cvtColor(mclab1, cv2.COLOR_RGB2BGR)
    low = cv2.cvtColor(
        apply_corruption(clean_bgr, "low_light", 2, 42), cv2.COLOR_BGR2RGB
    )
    snow = cv2.cvtColor(
        apply_corruption(clean_bgr, "marine_snow", 2, 43), cv2.COLOR_BGR2RGB
    )
    augmented = np.concatenate((low, snow), axis=1)

    fig = plt.figure(figsize=(16, 10), facecolor="#fbfdff")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.966,
        "AquaAdapt · Real-Data System Architecture",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=26,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        0.5,
        0.929,
        "Self-supervised underwater place recognition with DINOv2 ViT-S/14",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        color=MUTED,
    )

    # 1 · Data preparation
    section(ax, "1", "DATA\nPREPARATION", 0.706, 0.188, BLUE, BLUE_LIGHT)
    image_card(ax, mclab1, 0.145, 0.735, 0.135, 0.128, "MCLab 1", "training · cam0", BLUE)
    image_card(ax, mclab2, 0.293, 0.735, 0.135, 0.128, "MCLab 2", "training · cam0", BLUE)
    arrow(ax, (0.430, 0.799), (0.454, 0.799))
    text_card(
        ax,
        0.458,
        0.735,
        0.145,
        0.128,
        "Frame + pose association",
        "ROS1 images @ 5 Hz\nTUM nearest-pose matching\ncamera fixed to cam0",
        BLUE,
        badge="5 Hz",
    )
    arrow(ax, (0.604, 0.799), (0.625, 0.799))
    split_manifest_card(ax, 0.629, 0.735, 0.162, 0.128)
    image_card(
        ax,
        fjord1,
        0.817,
        0.735,
        0.145,
        0.128,
        "Fjord 1",
        "training · cam0",
        BLUE,
    )
    ax.text(
        0.889,
        0.718,
        "THREE TRAINING TRAJECTORIES · BALANCED SAMPLING",
        transform=ax.transAxes,
        fontsize=5.7,
        fontweight="bold",
        color=BLUE,
        ha="center",
    )

    # 2 · Model training
    section(ax, "2", "MODEL\nTRAINING", 0.393, 0.292, TEAL, TEAL_LIGHT)
    image_card(ax, mclab1, 0.145, 0.530, 0.175, 0.125, "Original frame", "immutable dataset image", TEAL)
    image_card(
        ax,
        augmented,
        0.348,
        0.530,
        0.215,
        0.125,
        "One controlled corrupted view",
        "one degradation per sample",
        TEAL,
        dashed=True,
    )
    arrow(ax, (0.321, 0.592), (0.345, 0.592), color=TEAL)
    arrow(ax, (0.235, 0.526), (0.235, 0.488), color=TEAL)
    arrow(ax, (0.455, 0.526), (0.310, 0.488), color=TEAL, connectionstyle="arc3,rad=0.12")

    text_card(
        ax,
        0.145,
        0.414,
        0.195,
        0.070,
        "DINOv2 ViT-S/14",
        "frozen feature encoder · CLS token · 384-D",
        TEAL,
        badge="384-D",
    )
    arrow(ax, (0.342, 0.449), (0.365, 0.449), color=TEAL)
    text_card(
        ax,
        0.369,
        0.414,
        0.155,
        0.070,
        "Residual adapter",
        "384 → 512 → 384 · scaled residual · L2 norm",
        TEAL,
    )
    arrow(ax, (0.526, 0.449), (0.550, 0.449), color=TEAL)
    text_card(
        ax,
        0.554,
        0.414,
        0.152,
        0.070,
        "AquaAdapt descriptor",
        "DINO-preserving robust representation",
        TEAL,
        badge="384-D",
    )
    descriptor_bars(ax, 0.570, 0.424, 0.116, 0.018)
    arrow(ax, (0.708, 0.449), (0.733, 0.449), color=TEAL)
    text_card(
        ax,
        0.737,
        0.414,
        0.225,
        0.070,
        "Multi-objective adaptation",
        "InfoNCE + DINO geometry preservation\n+ clean/corrupt consistency",
        TEAL,
        badge="SELF-SUPERVISED",
    )

    # 3 · Retrieval and evaluation
    section(ax, "3", "RETRIEVAL +\nEVALUATION", 0.094, 0.278, VIOLET, VIOLET_LIGHT)
    image_card(ax, fjord2, 0.145, 0.276, 0.130, 0.073, "Fjord2 query", "held-out", VIOLET)
    arrow(ax, (0.278, 0.312), (0.300, 0.312), color=VIOLET)
    text_card(
        ax,
        0.304,
        0.276,
        0.135,
        0.073,
        "AquaAdapt encoder",
        "frozen checkpoint\nnormalized 384-D query",
        VIOLET,
    )
    arrow(ax, (0.441, 0.312), (0.464, 0.312), color=VIOLET)
    text_card(
        ax,
        0.468,
        0.276,
        0.135,
        0.073,
        "Descriptor database",
        "clean reference frames\n+ pose metadata",
        VIOLET,
    )
    arrow(ax, (0.605, 0.312), (0.628, 0.312), color=VIOLET)
    text_card(
        ax,
        0.632,
        0.276,
        0.135,
        0.073,
        "Cosine retrieval",
        "FAISS / NumPy ranking\nwith temporal exclusion",
        VIOLET,
    )
    arrow(ax, (0.769, 0.312), (0.792, 0.312), color=VIOLET)
    text_card(
        ax,
        0.796,
        0.276,
        0.166,
        0.073,
        "Pose-based evaluation",
        "Recall@1/5/10 · MRR\ntranslation error · coverage",
        VIOLET,
        badge="1.5 m",
    )
    retrieval_triptych(ax, retrieval, 0.145, 0.119, 0.440, 0.132)
    robustness_card(ax, 0.608, 0.119, 0.354, 0.132)

    rounded(ax, 0.205, 0.025, 0.590, 0.047, face=WHITE, edge=BLUE, linewidth=1.25, radius=0.012)
    ax.text(
        0.5,
        0.049,
        "TRAIN  ·  MCLab 1 + MCLab 2 + Fjord 1     →     TEST  ·  Fjord 2 (held-out trajectory)",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=BLUE,
    )
    ax.text(
        0.972,
        0.018,
        "Final held-out Fjord2 results from the three-trajectory V2 checkpoint (best epoch 24).",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=5.6,
        color=MUTED,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "aquaadapt_architecture"
    fig.savefig(stem.with_suffix(".png"), dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(stem.with_suffix(".svg"), facecolor=fig.get_facecolor())
    plt.close(fig)
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
