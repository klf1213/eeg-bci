#!/usr/bin/env python3
"""Summarize whether learned temporal attention is focused or nearly uniform."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(path: Path) -> dict | None:
    with np.load(path) as saved:
        weights = saved["attention"]
    if weights.size == 0:
        return None
    weights = np.asarray(weights, dtype=np.float64)
    bins = weights.shape[1]
    entropy = -(weights * np.log(weights.clip(1e-12))).sum(axis=1) / np.log(bins)
    uniform = 1.0 / bins
    return {
        "file": path.name,
        "subject": path.name.split("_")[0],
        "fold": int(path.stem.split("fold")[-1].split("_")[0]),
        "trials": weights.shape[0],
        "time_bins": bins,
        "mean_max_weight": weights.max(axis=1).mean(),
        "mean_normalized_entropy": entropy.mean(),
        "mean_abs_deviation_from_uniform": np.abs(weights - uniform).mean(),
        "mean_trial_weight_std": weights.std(axis=0).mean(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Experiment directory under artifacts/runs.")
    parser.add_argument("--output", type=Path, help="Output CSV (defaults inside the run metrics directory).")
    args = parser.parse_args()
    metrics = args.run / "metrics"
    rows = [row for path in sorted(metrics.glob("*_predictions.npz")) if (row := summarize(path))]
    if not rows:
        raise SystemExit(f"No saved attention weights found in {metrics}")
    frame = pd.DataFrame(rows)
    output = args.output or metrics / "attention_diagnostics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.groupby("subject")[[
        "mean_max_weight", "mean_normalized_entropy",
        "mean_abs_deviation_from_uniform", "mean_trial_weight_std",
    ]].mean().to_string())
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
