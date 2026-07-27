"""Generic result plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training(log_csv: str | Path, output: str | Path) -> None:
    frame = pd.read_csv(log_csv)
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    if "validation_loss" in frame:
        axes[0].plot(frame["epoch"], frame["validation_loss"], label="validation")
    axes[0].set(xlabel="epoch", ylabel="loss"); axes[0].legend()
    axes[1].plot(frame["epoch"], frame["learning_rate"]); axes[1].set(xlabel="epoch", ylabel="learning rate")
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)

