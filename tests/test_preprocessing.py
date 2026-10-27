import numpy as np

from eeg_bci.preprocessing import standardize_splits


def test_standardization_uses_training_statistics_only():
    train = np.arange(4 * 2 * 8, dtype=np.float32).reshape(4, 2, 8)
    test = np.full((2, 2, 8), 10_000, dtype=np.float32)

    standardized_train, standardized_test, mean, std = standardize_splits(train, test)

    np.testing.assert_allclose(standardized_train.mean(axis=(0, 2)), 0, atol=1e-6)
    np.testing.assert_allclose(standardized_train.std(axis=(0, 2)), 1, atol=1e-6)
    assert standardized_test.mean() > 100
    assert mean.shape == (1, 2, 1)
    assert std.shape == (1, 2, 1)
