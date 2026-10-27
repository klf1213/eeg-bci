from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_bci.alignment import euclidean_align
from eeg_bci.datasets import make_loader
from eeg_bci.data import load_subject
from eeg_bci.evaluation import save_evaluation
from eeg_bci.models import MODEL_NAMES, build_model
from eeg_bci.preprocessing import preprocess_trials, standardize_splits
from eeg_bci.training import evaluate, fit, train_one_epoch
from eeg_bci.utils import artifact_dirs, get_device, load_config, set_seed


def load_t_session(path, window, config):
    x, y, _ = load_subject(
        path, window["tmin"], window["tmax"], config["data"]["n_channels"],
        config["data"]["remove_artifacts"], return_groups=True,
    )
    x = preprocess_trials(
        x,
        config["data"]["original_sampling_rate"],
        config["data"]["target_sampling_rate"],
        config["data"]["low_frequency"],
        config["data"]["high_frequency"],
    )
    return x, y


def stack_subjects(cache, subjects, alignment):
    xs, ys, groups, references = [], [], [], {}
    for subject in subjects:
        trials, labels = cache[subject]
        if alignment == "ea":
            trials, references[subject] = euclidean_align(trials)
        xs.append(trials)
        ys.append(labels)
        groups.append(np.full(len(labels), subject, dtype=object))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(groups), references


def train_fixed_epochs(model, loader, device, settings, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    rows = []
    for epoch in range(1, epochs + 1):
        loss, accuracy = train_one_epoch(model, loader, criterion, optimizer, device)
        rows.append({"epoch": epoch, "train_loss": loss, "train_accuracy": accuracy})
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  final epoch {epoch:03d}/{epochs}: loss={loss:.4f}, acc={accuracy:.4f}")
    return rows


def run_target(
    target,
    all_subjects,
    cache,
    model_name,
    window_name,
    alignment,
    config,
    paths,
    device,
    inner_splits,
):
    source_subjects = [s for s in all_subjects if s != target]
    source_x, source_y, source_groups, source_references = stack_subjects(
        cache, source_subjects, alignment
    )
    target_x, target_y = cache[target]
    alignment_arrays = {}
    if alignment == "ea":
        target_x, target_reference = euclidean_align(target_x)
        alignment_arrays = {
            "target_ea_reference": target_reference,
            "source_ea_references": np.stack(
                [source_references[subject] for subject in source_subjects]
            ),
        }
    settings = config["training"]
    splitter = GroupKFold(n_splits=inner_splits)
    fold_rows, best_epochs = [], []

    print(f"\n=== Outer target {target}; sources={source_subjects} ===")
    for fold, (train_idx, val_idx) in enumerate(splitter.split(source_x, source_y, source_groups), start=1):
        train_subjects = sorted(set(source_groups[train_idx]))
        val_subjects = sorted(set(source_groups[val_idx]))
        overlap = set(train_subjects) & set(val_subjects)
        if overlap or target in train_subjects or target in val_subjects:
            raise RuntimeError(f"Subject leakage in {target} fold {fold}: {overlap}")

        inner_train_x, inner_val_x, _, _ = standardize_splits(source_x[train_idx], source_x[val_idx])
        set_seed(config["seed"] + fold)
        train_loader = make_loader(inner_train_x, source_y[train_idx], settings["batch_size"], True)
        val_loader = make_loader(inner_val_x, source_y[val_idx], settings["batch_size"])
        model = build_model(model_name, inner_train_x.shape[-1], config).to(device)
        checkpoint = paths["checkpoints"] / f"{target}_{model_name}_{window_name}_inner{fold}.pt"
        history = fit(
            model, train_loader, val_loader, device,
            settings["learning_rate"], settings["weight_decay"],
            settings["max_epochs"], settings["patience"], checkpoint,
        )
        history_frame = pd.DataFrame(history)
        best_row = history_frame.loc[history_frame["val_loss"].idxmin()]
        best_epoch = int(best_row["epoch"])
        best_epochs.append(best_epoch)
        fold_rows.append({
            "target_subject": target,
            "inner_fold": fold,
            "train_subjects": ",".join(train_subjects),
            "validation_subjects": ",".join(val_subjects),
            "n_train_trials": len(train_idx),
            "n_validation_trials": len(val_idx),
            "best_epoch": best_epoch,
            "best_val_loss": float(best_row["val_loss"]),
            "best_val_accuracy": float(best_row["val_accuracy"]),
            "best_val_macro_f1": float(best_row["val_macro_f1"]),
        })
        history_frame.to_csv(paths["metrics"] / f"{target}_inner{fold}_history.csv", index=False)
        print(f"  inner {fold}: val={val_subjects}, best_epoch={best_epoch}, val_acc={best_row['val_accuracy']:.4f}")

    selected_epoch = max(1, int(np.median(best_epochs)))
    source_x_z, target_x_z, mean, std = standardize_splits(source_x, target_x)
    set_seed(config["seed"])
    final_loader = make_loader(source_x_z, source_y, settings["batch_size"], True)
    target_loader = make_loader(target_x_z, target_y, settings["batch_size"])
    final_model = build_model(model_name, source_x_z.shape[-1], config).to(device)
    final_history = train_fixed_epochs(final_model, final_loader, device, settings, selected_epoch)
    prefix = f"{target}_{model_name}_{window_name}_LOSO"
    torch.save(final_model.state_dict(), paths["checkpoints"] / f"{prefix}.pt")
    pd.DataFrame(final_history).to_csv(paths["metrics"] / f"{prefix}_history.csv", index=False)

    result = evaluate(final_model, target_loader, torch.nn.CrossEntropyLoss(), device)
    save_evaluation(
        result["targets"], result["predictions"], config["class_names"],
        prefix, paths["metrics"], paths["figures"],
    )
    np.savez_compressed(
        paths["metrics"] / f"{prefix}_predictions.npz",
        targets=result["targets"], predictions=result["predictions"],
        probabilities=result["probabilities"], mean=mean, std=std,
        source_subjects=np.asarray(source_subjects), target_subject=target,
        selected_epoch=selected_epoch,
        **alignment_arrays,
    )
    print(f"  TEST {target}: accuracy={result['accuracy']:.4f}, macro-F1={result['macro_f1']:.4f}, selected_epoch={selected_epoch}")
    return {
        "subject": target,
        "model": model_name,
        "window": window_name,
        "alignment": alignment,
        "seed": config["seed"],
        "source_subjects": ",".join(source_subjects),
        "inner_splits": inner_splits,
        "inner_best_epochs": ",".join(map(str, best_epochs)),
        "selected_epoch": selected_epoch,
        "train_trials": len(source_y),
        "test_trials": len(target_y),
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
    }, fold_rows


def main():
    parser = argparse.ArgumentParser(
        description="Nested leave-one-subject-out evaluation on BCI IV 2a T sessions."
    )
    parser.add_argument("--model", default="temporal-mean-hires", choices=MODEL_NAMES)
    parser.add_argument(
        "--alignment",
        choices=["none", "ea"],
        default="none",
        help="Optional transductive Euclidean Alignment using unlabeled target EEG.",
    )
    parser.add_argument("--window", default="standard", choices=["standard", "wide"])
    parser.add_argument("--targets", nargs="+", default=["all"])
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--run-name")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs
    if args.patience is not None:
        config["training"]["patience"] = args.patience
    if args.inner_splits < 2 or args.inner_splits >= len(config["subjects"]):
        raise ValueError("inner-splits must be between 2 and number_of_subjects-1")

    all_subjects = list(config["subjects"])
    targets = all_subjects if args.targets == ["all"] else args.targets
    unknown = sorted(set(targets) - set(all_subjects))
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}")
    alignment_suffix = "_ea" if args.alignment == "ea" else ""
    run_name = args.run_name or (
        f"{args.model}_{args.window}_nested_loso{alignment_suffix}_seed{config['seed']}"
    )
    experiment_root = args.output_dir / "runs" / run_name
    if experiment_root.exists() and any(experiment_root.iterdir()):
        raise FileExistsError(f"Non-empty experiment directory already exists: {experiment_root}")
    paths = artifact_dirs(experiment_root)
    device = get_device()

    window = config["windows"][args.window]
    cache = {}
    for subject in all_subjects:
        path = args.data_dir / f"{subject}T.mat"
        if not path.exists():
            raise FileNotFoundError(path)
        cache[subject] = load_t_session(path, window, config)
        print(f"Loaded {subject}: X={cache[subject][0].shape}, y={cache[subject][1].shape}")

    manifest = {
        "protocol": "nested leave-one-subject-out on T sessions",
        "alignment": args.alignment,
        "outer_test_unit": "one entirely held-out subject",
        "inner_model_selection": f"GroupKFold({args.inner_splits}) grouped by subject; median best epoch",
        "normalization_source": "inner-training subjects only; final fit uses all source subjects only",
        "target_subject_labels_used_for": "final scoring only",
        "target_subject_eeg_used_for": (
            "unsupervised EA reference and final evaluation"
            if args.alignment == "ea"
            else "final evaluation only"
        ),
        "model": args.model, "window": args.window, "seed": config["seed"],
        "subjects": all_subjects, "targets": targets, "device": str(device),
        "python": platform.python_version(), "torch": str(torch.__version__), "config": config,
    }
    with (experiment_root / "manifest.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=False)

    rows, all_fold_rows = [], []
    for target in targets:
        row, fold_rows = run_target(
            target, all_subjects, cache, args.model, args.window, args.alignment,
            config, paths, device, args.inner_splits,
        )
        rows.append(row)
        all_fold_rows.extend(fold_rows)
        pd.DataFrame(rows).to_csv(paths["metrics"] / "subject_results.csv", index=False)
        pd.DataFrame(all_fold_rows).to_csv(paths["metrics"] / "inner_fold_results.csv", index=False)

    frame = pd.DataFrame(rows)
    summary = frame[["accuracy", "macro_f1"]].agg(["mean", "std", "count"])
    summary.to_csv(paths["metrics"] / "overall_results.csv")
    print("\nPer-subject LOSO results:\n", frame.to_string(index=False))
    print("\nNested LOSO summary:\n", summary)


if __name__ == "__main__":
    main()
