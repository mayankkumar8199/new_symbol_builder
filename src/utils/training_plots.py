from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt


def save_training_plot(
    history: Iterable[Mapping[str, float]],
    output_path: Path,
    title: str,
) -> None:
    history = list(history)
    if not history:
        return

    epochs = [entry.get("epoch", idx + 1) for idx, entry in enumerate(history)]
    train_loss = [entry.get("loss") for entry in history]
    train_acc = [entry.get("acc") for entry in history]
    val_acc = [entry.get("val_acc") for entry in history if entry.get("val_acc") is not None]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_title(title)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(epochs, train_loss, color="tab:red", marker="o", label="Train Loss")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:blue")
    ax2.plot(epochs, train_acc, color="tab:blue", marker="s", label="Train Acc")
    if len(val_acc) == len(epochs):
        ax2.plot(epochs, val_acc, color="tab:green", marker="^", linestyle="--", label="Val Acc")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="lower right")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
