import numpy as np
from numpy.testing import assert_allclose
from eeg_bci.alignment import euclidean_align, mean_covariance


def test_euclidean_alignment_whitens_mean_covariance():
    rng = np.random.default_rng(7)
    mixing = np.array([[2.0, .4, .1, 0.0], [.4, 1.3, .2, .1], [.1, .2, .8, .2], [0.0, .1, .2, .5]])
    x = np.einsum("cd,ndt->nct", mixing, rng.normal(size=(40, 4, 128))).astype(np.float32)
    aligned, reference = euclidean_align(x)
    assert aligned.shape == x.shape
    assert reference.shape == (4, 4)
    assert np.isfinite(aligned).all()
    assert_allclose(mean_covariance(aligned), np.eye(4), atol=2e-5)


def test_fixed_reference_is_reused_without_labels():
    rng = np.random.default_rng(9)
    source = rng.normal(size=(12, 3, 64)).astype(np.float32)
    target = rng.normal(size=(7, 3, 64)).astype(np.float32)
    _, reference = euclidean_align(source)
    aligned, returned = euclidean_align(target, reference)
    assert aligned.shape == target.shape
    assert_allclose(returned, reference)
