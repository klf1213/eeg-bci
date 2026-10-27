#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_run(run_dir: Path) -> pd.DataFrame:
    manifest_path = run_dir / "manifest.yaml"
    results_path = run_dir / "metrics" / "subject_results.csv"
    if not manifest_path.exists() or not results_path.exists():
        raise FileNotFoundError(f"Incomplete experiment directory: {run_dir}")
    with manifest_path.open(encoding="utf-8") as file:
        manifest = yaml.safe_load(file)
    frame = pd.read_csv(results_path)
    frame.insert(0, "run", run_dir.name)
    frame.insert(1, "model", manifest["model"])
    frame.insert(2, "window", manifest["window"])
    frame.insert(3, "split_strategy", manifest["split_strategy"])
    return frame


def main():
    parser = argparse.ArgumentParser(description="Compare completed EEG-BCI experiment runs.")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    comparison = pd.concat([load_run(path) for path in args.runs], ignore_index=True)
    comparison = comparison.sort_values(
        ["split_strategy", "window", "subject", "model"]
    ).reset_index(drop=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(args.output, index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
