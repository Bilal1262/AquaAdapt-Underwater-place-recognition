#!/usr/bin/env python3
"""Generate the compact figures embedded in the public README."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"
ASSETS = ROOT / "docs" / "assets"

BLUE = "#1769aa"
ORANGE = "#e86f00"
GREEN = "#159a68"
GRAY = "#607080"
GRID = "#d9e1e8"


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)


def training_figure() -> Path:
    frame = pd.read_csv(RESULTS / "training_log_v2.csv")
    epochs = frame["epoch"].to_numpy(dtype=int) + 1
    best_index = int(frame["validation_loss"].idxmin())
    best_epoch = int(epochs[best_index])
    best_loss = float(frame.loc[best_index, "validation_loss"])

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), constrained_layout=True)

    axis = axes[0]
    axis.plot(epochs, frame["train_loss"], color=BLUE, linewidth=2.4, label="Training")
    axis.plot(
        epochs,
        frame["validation_loss"],
        color=ORANGE,
        linewidth=2.4,
        label="Validation",
    )
    axis.scatter([best_epoch], [best_loss], color=GREEN, s=65, zorder=5)
    axis.annotate(
        f"best epoch {best_epoch}\nvalidation loss {best_loss:.3f}",
        xy=(best_epoch, best_loss),
        xytext=(-92, 35),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": GREEN},
        fontsize=9,
        color=GREEN,
    )
    axis.set(
        title="Optimization converges without validation collapse",
        xlabel="Epoch",
        ylabel="Loss",
    )
    axis.legend(frameon=False, ncol=2)
    _style_axis(axis)

    axis = axes[1]
    axis.plot(
        epochs,
        frame["contrastive_loss"],
        color=BLUE,
        linewidth=2.2,
        label="Contrastive",
    )
    axis.plot(
        epochs,
        frame["preservation_loss"],
        color=ORANGE,
        linewidth=2.2,
        label="DINO geometry preservation",
    )
    axis.plot(
        epochs,
        frame["consistency_loss"],
        color=GREEN,
        linewidth=2.2,
        label="Clean/corrupt consistency",
    )
    axis.set(
        title="Balanced residual-adapter objectives",
        xlabel="Epoch",
        ylabel="Component loss",
    )
    axis.legend(frameon=False)
    _style_axis(axis)

    figure.suptitle(
        "AquaAdapt V2 training · MCLab1 + MCLab2 + Fjord1",
        fontsize=15,
        fontweight="bold",
    )
    output = ASSETS / "training_curves_v2.png"
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    return output


def robustness_figure() -> Path:
    frame = pd.read_csv(RESULTS / "fjord2_robustness.csv")
    recall = frame[frame["metric"].eq("Recall@1")].copy()

    macro = recall.pivot_table(
        index="severity", columns="method", values="value", aggfunc="mean"
    ).sort_index()
    severe = recall[recall["severity"].eq(3)].pivot(
        index="corruption", columns="method", values="value"
    )
    severe["gain"] = severe["aquaadapt"] - severe["raw_dinov2"]
    severe = severe.sort_values("gain")

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 4.8),
        gridspec_kw={"width_ratios": [1.02, 1.18]},
        constrained_layout=True,
    )

    axis = axes[0]
    severity = macro.index.to_numpy(dtype=int)
    axis.plot(
        severity,
        100 * macro["raw_dinov2"],
        color=GRAY,
        marker="o",
        markersize=7,
        linewidth=2.5,
        label="Raw DINOv2",
    )
    axis.plot(
        severity,
        100 * macro["aquaadapt"],
        color=BLUE,
        marker="o",
        markersize=7,
        linewidth=2.8,
        label="AquaAdapt V2",
    )
    axis.fill_between(
        severity,
        100 * macro["raw_dinov2"].to_numpy(),
        100 * macro["aquaadapt"].to_numpy(),
        color=BLUE,
        alpha=0.10,
    )
    for level in severity[1:]:
        gain = 100 * (
            float(macro.loc[level, "aquaadapt"])
            - float(macro.loc[level, "raw_dinov2"])
        )
        axis.annotate(
            f"+{gain:.1f} pp",
            (level, 100 * float(macro.loc[level, "aquaadapt"])),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color=BLUE,
            fontweight="bold",
            fontsize=9,
        )
    axis.set(
        title="Macro Recall@1 remains stronger as degradation grows",
        xlabel="Corruption severity",
        ylabel="Recall@1 (%)",
        xticks=severity,
    )
    axis.legend(frameon=False)
    _style_axis(axis)

    axis = axes[1]
    labels = [name.replace("_", " ").title() for name in severe.index]
    positions = np.arange(len(labels))
    height = 0.34
    raw_values = 100 * severe["raw_dinov2"].to_numpy()
    adapted_values = 100 * severe["aquaadapt"].to_numpy()
    axis.barh(
        positions - height / 2,
        raw_values,
        height,
        color=GRAY,
        label="Raw DINOv2",
    )
    axis.barh(
        positions + height / 2,
        adapted_values,
        height,
        color=BLUE,
        label="AquaAdapt V2",
    )
    for position, value, gain in zip(
        positions, adapted_values, 100 * severe["gain"].to_numpy(), strict=True
    ):
        axis.text(
            value + 0.45,
            position + height / 2,
            f"+{gain:.1f}",
            va="center",
            color=BLUE,
            fontsize=9,
            fontweight="bold",
        )
    axis.set(
        title="Every severe corruption improves",
        xlabel="Severity-3 Recall@1 (%) · label shows gain in pp",
        yticks=positions,
        yticklabels=labels,
    )
    axis.set_xlim(0, max(adapted_values.max(), raw_values.max()) + 6)
    _style_axis(axis)

    figure.suptitle(
        "Held-out Fjord2 robustness · 665 eligible loop-closure queries",
        fontsize=15,
        fontweight="bold",
    )
    output = ASSETS / "fjord2_robustness_overview.png"
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    return output


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for output in (training_figure(), robustness_figure()):
        print(output)


if __name__ == "__main__":
    main()
