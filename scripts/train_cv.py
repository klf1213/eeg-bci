#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import yaml

from eeg_bci.data import load_subject
from eeg_bci.datasets import make_loader
from eeg_bci.evaluation import save_evaluation, save_history
from eeg_bci.models import MODEL_NAMES, build_model
from eeg_bci.preprocessing import preprocess_trials, standardize_splits
from eeg_bci.splitting import make_splits
from eeg_bci.training import evaluate, fit
from eeg_bci.utils import artifact_dirs, get_device, load_config, set_seed


def _cache_fingerprint(source: Path, window: dict, data_config: dict) -> tuple[str, dict]:
    source_stat = source.stat()
    metadata = {
        "preprocessing_version": 2,
        "source": str(source.resolve()),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "window": window,
        "data": data_config,
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16], metadata


def load_preprocessed(subject, data_dir, cache_dir, config, window_name):
    window = config["windows"][window_name]
    source = data_dir / f"{subject}T.mat"
    fingerprint, metadata = _cache_fingerprint(source, window, config["data"])
    cache = cache_dir / f"{subject}_{fingerprint}.npz"
    if cache.exists():
        with np.load(cache) as saved:
            return saved["x"], saved["y"], saved["run_ids"]
    data = config["data"]
    x, y, run_ids = load_subject(
        source, window["tmin"], window["tmax"], data["n_channels"],
        data["remove_artifacts"], return_groups=True,
    )
    x = preprocess_trials(x, data["original_sampling_rate"], data["target_sampling_rate"], data["low_frequency"], data["high_frequency"])
    np.savez_compressed(cache, x=x, y=y, run_ids=run_ids, metadata=json.dumps(metadata, sort_keys=True))
    return x, y, run_ids


def run_subject(subject, model_name, window_name, split_strategy, data_dir, config, paths, device):
    x, y, run_ids = load_preprocessed(subject, data_dir, paths["cache"], config, window_name)
    settings = config["training"]
    fold_rows, all_targets, all_predictions = [], [], []
    split_iterator = make_splits(x, y, run_ids, settings, config["seed"], split_strategy)
    for fold, train_indices, validation_indices, test_indices in split_iterator:
        set_seed(config["seed"] + fold)
        train_x, validation_x, test_x, mean, std = standardize_splits(x[train_indices], x[validation_indices], x[test_indices])
        loaders = {
            "train": make_loader(train_x, y[train_indices], settings["batch_size"], True),
            "validation": make_loader(validation_x, y[validation_indices], settings["batch_size"]),
            "test": make_loader(test_x, y[test_indices], settings["batch_size"]),
        }
        model = build_model(model_name, x.shape[-1], config).to(device)
        prefix = f"{subject}_{model_name}_{window_name}_{split_strategy}_fold{fold}"
        history = fit(model, loaders["train"], loaders["validation"], device, settings["learning_rate"],
                      settings["weight_decay"], settings["max_epochs"], settings["patience"],
                      paths["checkpoints"] / f"{prefix}.pt")
        result = evaluate(model, loaders["test"], torch.nn.CrossEntropyLoss(), device,
                          return_attention=model_name.startswith("attention"))
        save_history(history, prefix, paths["metrics"], paths["figures"])
        np.savez_compressed(paths["metrics"] / f"{prefix}_predictions.npz", targets=result["targets"],
                            predictions=result["predictions"], probabilities=result["probabilities"],
                            attention=result.get("attention", np.empty(0)), mean=mean, std=std,
                            train_indices=train_indices, validation_indices=validation_indices,
                            test_indices=test_indices, test_run_ids=run_ids[test_indices])
        fold_rows.append({"subject": subject, "model": model_name, "fold": fold,
                          "window": window_name, "split_strategy": split_strategy,
                          "accuracy": result["accuracy"], "macro_f1": result["macro_f1"]})
        all_targets.extend(result["targets"]); all_predictions.extend(result["predictions"])
        print(f"{prefix}: accuracy={result['accuracy']:.4f}, macro-F1={result['macro_f1']:.4f}")
    prefix = f"{subject}_{model_name}_{window_name}_{split_strategy}_cv"
    save_evaluation(all_targets, all_predictions, config["class_names"], prefix, paths["metrics"], paths["figures"])
    return fold_rows


def main():
    parser = argparse.ArgumentParser(description="Run subject-wise stratified cross-validation.")
    parser.add_argument(
        "--model",
        choices=[*MODEL_NAMES, "attention-wide"],
        required=True,
    )
    parser.add_argument("--window", choices=["standard", "wide"], default=None,
                        help="Input window, independent of model architecture.")
    parser.add_argument("--split-strategy", choices=["trial", "run"], default="trial",
                        help="Use stratified trial CV or keep complete recording runs separate.")
    parser.add_argument("--subjects", nargs="+", default=["A01"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--config", default=ROOT / "configs/default.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--run-name", help="Optional experiment directory name.")
    parser.add_argument("--seed", type=int, help="Override the configured random seed.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    model_name = "attention" if args.model == "attention-wide" else args.model
    window_name = args.window or ("wide" if args.model == "attention-wide" else "standard")
    run_name = args.run_name or f"{model_name}_{window_name}_{args.split_strategy}_seed{config['seed']}"
    experiment_root = args.output_dir / "runs" / run_name
    if experiment_root.exists() and any(experiment_root.iterdir()):
        raise FileExistsError(
            f"Experiment directory already exists and is not empty: {experiment_root}. "
            "Choose a new --run-name to avoid overwriting results."
        )
    set_seed(config["seed"])
    paths = artifact_dirs(experiment_root)
    paths["cache"] = args.output_dir / "cache"
    paths["cache"].mkdir(parents=True, exist_ok=True)
    subjects = config["subjects"] if args.subjects == ["all"] else args.subjects
    manifest = {
        "model": model_name,
        "window": window_name,
        "split_strategy": args.split_strategy,
        "subjects": subjects,
        "data_dir": str(args.data_dir.resolve()),
        "device": str(get_device()),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "config": config,
    }
    with (experiment_root / "manifest.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=False)
    rows = []
    device = get_device()
    for subject in subjects:
        rows.extend(run_subject(
            subject, model_name, window_name, args.split_strategy,
            args.data_dir, config, paths, device,
        ))
    frame = pd.DataFrame(rows)
    frame.to_csv(paths["metrics"] / "fold_results.csv", index=False)
    subject_summary = frame.groupby("subject", as_index=False)[["accuracy", "macro_f1"]].mean()
    subject_summary.to_csv(paths["metrics"] / "subject_results.csv", index=False)
    overall = subject_summary[["accuracy", "macro_f1"]].agg(["mean", "std", "count"])
    overall.to_csv(paths["metrics"] / "overall_results.csv")
    print(subject_summary.to_string(index=False))
    print("\nSubject-level summary:\n", overall)


if __name__ == "__main__":
    main()
