from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def save_evaluation(targets, predictions, class_names, prefix: str, metrics_dir: Path, figure_dir: Path):
    report = classification_report(targets, predictions, target_names=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(metrics_dir / f"{prefix}_classification_report.csv")
    display = ConfusionMatrixDisplay.from_predictions(targets, predictions, display_labels=class_names, cmap="Blues")
    display.ax_.set_title(prefix.replace("_", " "))
    display.figure_.tight_layout(); display.figure_.savefig(figure_dir / f"{prefix}_confusion.png", dpi=160)
    plt.close(display.figure_)


def save_history(history, prefix: str, metrics_dir: Path, figure_dir: Path):
    frame = pd.DataFrame(history); frame.to_csv(metrics_dir / f"{prefix}_history.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(frame.epoch, frame.train_loss, label="train"); axes[0].plot(frame.epoch, frame.val_loss, label="validation")
    axes[1].plot(frame.epoch, frame.train_accuracy, label="train"); axes[1].plot(frame.epoch, frame.val_accuracy, label="validation")
    for axis in axes: axis.grid(True); axis.legend(); axis.set_xlabel("Epoch")
    figure.tight_layout(); figure.savefig(figure_dir / f"{prefix}_training.png", dpi=160); plt.close(figure)

