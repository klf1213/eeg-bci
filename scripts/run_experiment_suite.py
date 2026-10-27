#!/usr/bin/env python3
"""Run a reproducible model/window/split/seed experiment matrix."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["temporal-mean-hires", "attention-hires", "attention-positional-residual"])
    parser.add_argument("--windows", nargs="+", choices=["standard", "wide"], default=["standard", "wide"])
    parser.add_argument("--split-strategies", nargs="+", choices=["trial", "run"], default=["trial", "run"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--subjects", nargs="+", default=["all"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()

    for model in args.models:
        for window in args.windows:
            for split in args.split_strategies:
                for seed in args.seeds:
                    run_name = f"{model}_{window}_{split}_seed{seed}"
                    command = [
                        sys.executable, str(ROOT / "scripts/train_cv.py"),
                        "--model", model, "--window", window,
                        "--split-strategy", split, "--seed", str(seed),
                        "--subjects", *args.subjects,
                        "--data-dir", str(args.data_dir),
                        "--output-dir", str(args.output_dir),
                        "--run-name", run_name,
                    ]
                    print(f"\n=== {run_name} ===", flush=True)
                    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
