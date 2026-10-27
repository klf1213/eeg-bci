#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_bci.data import inspect_mat_file


def main():
    parser = argparse.ArgumentParser(description="Inspect BCI MATLAB training files.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    args = parser.parse_args()
    files = sorted(args.data_dir.glob("A*T.mat"))
    if not files:
        raise FileNotFoundError(f"No A*T.mat files found in {args.data_dir}")
    for path in files:
        print(inspect_mat_file(path))


if __name__ == "__main__":
    main()

