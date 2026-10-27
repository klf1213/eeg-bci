from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io


def extract_trials_from_run(run, tmin=2.5, tmax=4.5, n_channels=22, remove_artifacts=True):
    continuous = np.asarray(run.X, dtype=np.float32)
    starts = np.asarray(run.trial).astype(int).reshape(-1)
    labels = np.asarray(run.y).astype(int).reshape(-1)
    sampling_rate = float(np.asarray(run.fs).squeeze())
    if not len(starts) or not len(labels):
        return np.empty((0, n_channels, 0), np.float32), np.empty(0, np.int64)
    artifacts = np.asarray(getattr(run, "artifacts", np.zeros(len(labels)))).astype(int).reshape(-1)
    if not len(artifacts):
        artifacts = np.zeros(len(labels), dtype=int)
    if not (len(starts) == len(labels) == len(artifacts)):
        raise ValueError("Trial, label, and artifact counts do not match.")
    start_offset, end_offset = round(tmin * sampling_rate), round(tmax * sampling_rate)
    trials, targets = [], []
    for start, label, artifact in zip(starts, labels, artifacts):
        first, last = start + start_offset, start + end_offset
        if (remove_artifacts and artifact) or first < 0 or last > continuous.shape[0]:
            continue
        trials.append(continuous[first:last, :n_channels].T)
        targets.append(int(label) - 1)
    if not trials:
        return np.empty((0, n_channels, end_offset - start_offset), np.float32), np.empty(0, np.int64)
    return np.stack(trials).astype(np.float32), np.asarray(targets, dtype=np.int64)


def load_subject(
    path: str | Path,
    tmin=2.5,
    tmax=4.5,
    n_channels=22,
    remove_artifacts=True,
    return_groups: bool = False,
):
    path = Path(path)
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    if "data" not in mat:
        raise KeyError(f"No 'data' variable found in {path.name}.")
    all_x, all_y, all_run_ids = [], [], []
    required = {"X", "trial", "y", "fs"}
    for run_id, run in enumerate(np.atleast_1d(mat["data"])):
        if not required.issubset(set(getattr(run, "_fieldnames", []))):
            continue
        x, y = extract_trials_from_run(run, tmin, tmax, n_channels, remove_artifacts)
        if len(x):
            all_x.append(x)
            all_y.append(y)
            all_run_ids.append(np.full(len(y), run_id, dtype=np.int64))
    if not all_x:
        raise ValueError(f"No valid trials extracted from {path.name}.")
    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    if return_groups:
        return x, y, np.concatenate(all_run_ids)
    return x, y


def inspect_mat_file(path: str | Path) -> dict:
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    runs = np.atleast_1d(mat.get("data", []))
    return {"file": Path(path).name, "runs": len(runs), "fields": list(getattr(runs[0], "_fieldnames", [])) if len(runs) else []}
