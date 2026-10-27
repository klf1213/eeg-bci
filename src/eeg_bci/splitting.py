from __future__ import annotations

import numpy as np
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)


def make_splits(x, y, run_ids, settings, seed, strategy):
    """Yield leakage-aware train, validation and test indices."""
    if strategy == "trial":
        outer = StratifiedKFold(settings["folds"], shuffle=True, random_state=seed)
        for fold, (development, test) in enumerate(outer.split(x, y), 1):
            train, validation = train_test_split(
                development,
                test_size=settings["inner_validation_size"],
                random_state=seed + fold,
                stratify=y[development],
            )
            yield fold, train, validation, test
        return

    if strategy != "run":
        raise ValueError(f"Unknown split strategy: {strategy}.")
    n_runs = len(np.unique(run_ids))
    if n_runs < settings["folds"]:
        raise ValueError(
            f"run-grouped CV needs at least {settings['folds']} runs; found {n_runs}."
        )
    outer = StratifiedGroupKFold(
        settings["folds"], shuffle=True, random_state=seed
    )
    for fold, (development, test) in enumerate(
        outer.split(x, y, groups=run_ids), 1
    ):
        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=settings["inner_validation_size"],
            random_state=seed + fold,
        )
        relative_train, relative_validation = next(
            inner.split(x[development], y[development], groups=run_ids[development])
        )
        yield (
            fold,
            development[relative_train],
            development[relative_validation],
            test,
        )
