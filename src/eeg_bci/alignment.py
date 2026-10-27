from __future__ import annotations

import numpy as np


def mean_covariance(trials: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Return the regularized arithmetic mean covariance of EEG trials.

    Parameters
    ----------
    trials : ndarray, shape (n_trials, n_channels, n_times)
    eps : relative diagonal regularization strength
    """
    trials = np.asarray(trials, dtype=np.float64)
    if trials.ndim != 3 or trials.shape[0] == 0:
        raise ValueError("trials must have shape (n_trials, n_channels, n_times)")
    centered = trials - trials.mean(axis=-1, keepdims=True)
    covariances = np.einsum("nct,ndt->ncd", centered, centered) / max(trials.shape[-1] - 1, 1)
    reference = covariances.mean(axis=0)
    scale = float(np.trace(reference) / reference.shape[0])
    reference += eps * max(scale, np.finfo(np.float64).eps) * np.eye(reference.shape[0])
    return reference


def inverse_sqrt_spd(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Stable inverse square root for a symmetric positive definite matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    floor = eps * max(float(eigenvalues.max()), 1.0)
    eigenvalues = np.clip(eigenvalues, floor, None)
    return (eigenvectors * eigenvalues**-0.5) @ eigenvectors.T


def euclidean_align(trials: np.ndarray, reference: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Align EEG trials with R_bar^{-1/2} X.

    If reference is omitted it is estimated without labels from the supplied trials.
    """
    trials = np.asarray(trials)
    if reference is None:
        reference = mean_covariance(trials)
    whitener = inverse_sqrt_spd(reference)
    aligned = np.einsum("cd,ndt->nct", whitener, trials, optimize=True)
    return aligned.astype(np.float32, copy=False), np.asarray(reference, dtype=np.float64)
