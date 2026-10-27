from types import SimpleNamespace

import numpy as np

from eeg_bci.data import extract_trials_from_run


def test_extract_trials_removes_artifacts_and_converts_labels():
    run = SimpleNamespace(
        X=np.arange(400 * 3, dtype=np.float32).reshape(400, 3),
        trial=np.array([0, 100]),
        y=np.array([1, 4]),
        fs=100,
        artifacts=np.array([0, 1]),
    )

    x, y = extract_trials_from_run(
        run, tmin=0.0, tmax=1.0, n_channels=2, remove_artifacts=True
    )

    assert x.shape == (1, 2, 100)
    np.testing.assert_array_equal(y, [0])


def test_extract_trials_can_keep_artifacts():
    run = SimpleNamespace(
        X=np.zeros((400, 3), dtype=np.float32),
        trial=np.array([0, 100]),
        y=np.array([1, 4]),
        fs=100,
        artifacts=np.array([0, 1]),
    )

    x, y = extract_trials_from_run(
        run, tmin=0.0, tmax=1.0, n_channels=2, remove_artifacts=False
    )

    assert x.shape == (2, 2, 100)
    np.testing.assert_array_equal(y, [0, 3])
