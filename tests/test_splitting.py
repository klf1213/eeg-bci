import numpy as np

from eeg_bci.splitting import make_splits


def test_run_split_keeps_groups_disjoint():
    run_ids = np.repeat(np.arange(6), 16)
    y = np.tile(np.repeat(np.arange(4), 4), 6)
    x = np.zeros((len(y), 1), dtype=np.float32)
    settings = {"folds": 4, "inner_validation_size": 0.2}

    splits = list(make_splits(x, y, run_ids, settings, seed=42, strategy="run"))

    assert len(splits) == 4
    for _, train, validation, test in splits:
        train_runs = set(run_ids[train])
        validation_runs = set(run_ids[validation])
        test_runs = set(run_ids[test])
        assert train_runs.isdisjoint(validation_runs)
        assert train_runs.isdisjoint(test_runs)
        assert validation_runs.isdisjoint(test_runs)
