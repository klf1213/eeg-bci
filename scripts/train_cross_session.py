#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from eeg_bci.alignment import euclidean_align
from eeg_bci.data import load_subject
from eeg_bci.datasets import make_loader
from eeg_bci.evaluation import save_evaluation, save_history
from eeg_bci.models import MODEL_NAMES, build_model
from eeg_bci.preprocessing import preprocess_trials, standardize_splits
from eeg_bci.training import evaluate, fit
from eeg_bci.utils import artifact_dirs, get_device, load_config, set_seed


def official_labels(path: Path) -> np.ndarray:
    values = loadmat(path, squeeze_me=True)["classlabel"]
    return np.asarray(values).reshape(-1).astype(np.int64) - 1


def load_session(path: Path, window: dict, config: dict, remove_artifacts: bool):
    data = config["data"]
    x, y, groups = load_subject(
        path, window["tmin"], window["tmax"], data["n_channels"],
        remove_artifacts, return_groups=True,
    )
    x = preprocess_trials(
        x, data["original_sampling_rate"], data["target_sampling_rate"],
        data["low_frequency"], data["high_frequency"],
    )
    return x, y, groups


def run_subject(
    subject,
    model_name,
    window_name,
    alignment,
    data_dir,
    labels_dir,
    config,
    paths,
    device,
):
    window = config["windows"][window_name]
    train_path = data_dir / f"{subject}T.mat"
    evaluation_path = data_dir / f"{subject}E.mat"
    label_path = labels_dir / f"{subject}E.mat"

    # Verify the post-competition MAT labels against the official release before
    # artifact rejection so trial indexing cannot silently drift.
    _, evaluation_all_y, _ = load_subject(
        evaluation_path, window["tmin"], window["tmax"],
        config["data"]["n_channels"], False, return_groups=True,
    )
    released_labels = official_labels(label_path)
    if not np.array_equal(evaluation_all_y, released_labels):
        raise ValueError(f"Official label mismatch for {subject}: {len(evaluation_all_y)} vs {len(released_labels)}")

    train_x, train_y, _ = load_session(train_path, window, config, config["data"]["remove_artifacts"])
    evaluation_x, evaluation_y, evaluation_runs = load_session(
        evaluation_path, window, config, config["data"]["remove_artifacts"]
    )

    indices = np.arange(len(train_y))
    train_indices, validation_indices = train_test_split(
        indices, test_size=config["training"]["inner_validation_size"],
        random_state=config["seed"], stratify=train_y,
    )
    train_split = train_x[train_indices]
    validation_split = train_x[validation_indices]
    alignment_arrays = {}
    if alignment == "ea":
        train_split, source_reference = euclidean_align(train_split)
        validation_split, _ = euclidean_align(validation_split, source_reference)
        evaluation_x, target_reference = euclidean_align(evaluation_x)
        alignment_arrays = {
            "source_ea_reference": source_reference,
            "target_ea_reference": target_reference,
        }

    train_x, validation_x, evaluation_x, mean, std = standardize_splits(
        train_split, validation_split, evaluation_x
    )
    settings = config["training"]
    loaders = {
        "train": make_loader(train_x, train_y[train_indices], settings["batch_size"], True),
        "validation": make_loader(validation_x, train_y[validation_indices], settings["batch_size"]),
        "evaluation": make_loader(evaluation_x, evaluation_y, settings["batch_size"]),
    }

    set_seed(config["seed"])
    model = build_model(model_name, train_x.shape[-1], config).to(device)
    prefix = f"{subject}_{model_name}_{window_name}_T-to-E"
    history = fit(
        model, loaders["train"], loaders["validation"], device,
        settings["learning_rate"], settings["weight_decay"],
        settings["max_epochs"], settings["patience"],
        paths["checkpoints"] / f"{prefix}.pt",
    )
    result = evaluate(
        model, loaders["evaluation"], torch.nn.CrossEntropyLoss(), device,
        return_attention=model_name.startswith("attention"),
    )
    save_history(history, prefix, paths["metrics"], paths["figures"])
    save_evaluation(
        result["targets"], result["predictions"], config["class_names"],
        prefix, paths["metrics"], paths["figures"],
    )
    np.savez_compressed(
        paths["metrics"] / f"{prefix}_predictions.npz",
        targets=result["targets"], predictions=result["predictions"],
        probabilities=result["probabilities"],
        attention=result.get("attention", np.empty(0)),
        mean=mean, std=std, train_indices=train_indices,
        validation_indices=validation_indices,
        evaluation_run_ids=evaluation_runs,
        **alignment_arrays,
    )
    print(f"{prefix}: accuracy={result['accuracy']:.4f}, macro-F1={result['macro_f1']:.4f}")
    return {
        "subject": subject, "model": model_name, "window": window_name,
        "alignment": alignment,
        "seed": config["seed"], "train_trials": len(train_indices),
        "validation_trials": len(validation_indices),
        "evaluation_trials": len(evaluation_y),
        "best_epoch": int(pd.DataFrame(history)["val_loss"].idxmin() + 1),
        "accuracy": result["accuracy"], "macro_f1": result["macro_f1"],
    }


def main():
    parser = argparse.ArgumentParser(description="Train on BCI IV 2a T session and test once on E session.")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument(
        "--alignment",
        choices=["none", "ea"],
        default="none",
        help="Optional Euclidean Alignment using unlabeled target-session EEG.",
    )
    parser.add_argument("--window", choices=["standard", "wide"], default="standard")
    parser.add_argument("--subjects", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--labels-dir", type=Path, default=ROOT / "data/evaluation_labels")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--run-name")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    subjects = config["subjects"] if args.subjects == ["all"] else args.subjects
    alignment_suffix = "_ea" if args.alignment == "ea" else ""
    run_name = args.run_name or (
        f"{args.model}_{args.window}_cross_session{alignment_suffix}_seed{config['seed']}"
    )
    experiment_root = args.output_dir / "runs" / run_name
    if experiment_root.exists() and any(experiment_root.iterdir()):
        raise FileExistsError(f"Non-empty experiment directory already exists: {experiment_root}")

    paths = artifact_dirs(experiment_root)
    device = get_device()
    manifest = {
        "protocol": "subject-specific T-session training to held-out E-session evaluation",
        "alignment": args.alignment,
        "model": args.model, "window": args.window, "seed": config["seed"],
        "subjects": subjects, "device": str(device),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "normalization_source": "T-session inner-training subset only",
        "model_selection_source": "T-session inner validation only",
        "evaluation_labels_used_for": "final scoring only",
        "evaluation_eeg_used_for": (
            "unsupervised EA reference and final evaluation"
            if args.alignment == "ea"
            else "final evaluation only"
        ),
        "config": config,
    }
    with (experiment_root / "manifest.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=False)

    rows = [run_subject(
        subject, args.model, args.window, args.alignment,
        args.data_dir, args.labels_dir,
        config, paths, device,
    ) for subject in subjects]
    frame = pd.DataFrame(rows)
    frame.to_csv(paths["metrics"] / "subject_results.csv", index=False)
    summary = frame[["accuracy", "macro_f1"]].agg(["mean", "std", "count"])
    summary.to_csv(paths["metrics"] / "overall_results.csv")
    print("\n", frame.to_string(index=False))
    print("\nCross-session summary:\n", summary)


if __name__ == "__main__":
    main()
