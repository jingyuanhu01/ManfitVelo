"""PCA denoising baselines for benchmark comparisons.

These helpers intentionally keep the reconstructed matrix in the original
ambient coordinates. They are baselines for denoising, not embedding methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCAInfo:
    """Diagnostics returned by PCA denoising helpers."""

    rank: int
    explained_variance_ratio: np.ndarray
    singular_values: np.ndarray
    mean: np.ndarray
    components: np.ndarray
    center: bool
    cumulative_explained_variance: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "explained_variance_ratio": self.explained_variance_ratio,
            "singular_values": self.singular_values,
            "mean": self.mean,
            "components": self.components,
            "center": self.center,
            "cumulative_explained_variance": self.cumulative_explained_variance,
        }


def _validate_matrix(X: np.ndarray, name: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{name} must have nonzero shape")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return X


def _svd_pca(X: np.ndarray, center: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = _validate_matrix(X)
    mean = X.mean(axis=0) if center else np.zeros(X.shape[1], dtype=float)
    X_centered = X - mean
    _, singular_values, vt = np.linalg.svd(X_centered, full_matrices=False)
    squared = singular_values**2
    total = float(np.sum(squared))
    explained = squared / total if total > 0.0 else np.zeros_like(squared)
    cumulative = np.cumsum(explained)
    return X_centered, singular_values, vt, mean, cumulative


def global_pca_denoise(X, rank, center=True, return_info=True):
    """Denoise ``X`` with a fixed-rank global PCA reconstruction.

    ``rank`` must be in ``[1, min(n, D)]``. Rank 0 is intentionally disallowed:
    returning only the mean is a different baseline and would make accidental
    over-smoothing too easy to miss.
    """

    X = _validate_matrix(X)
    rank = int(rank)
    max_rank = min(X.shape)
    if rank < 1 or rank > max_rank:
        raise ValueError(f"rank must be between 1 and {max_rank}; got {rank}")

    X_centered, singular_values, vt, mean, cumulative = _svd_pca(X, bool(center))
    components = vt[:rank]
    X_hat = (X_centered @ components.T) @ components + mean

    squared = singular_values**2
    total = float(np.sum(squared))
    explained = squared / total if total > 0.0 else np.zeros_like(squared)
    info = PCAInfo(
        rank=rank,
        explained_variance_ratio=explained[:rank].copy(),
        singular_values=singular_values[:rank].copy(),
        mean=mean.copy(),
        components=components.copy(),
        center=bool(center),
        cumulative_explained_variance=cumulative[:rank].copy(),
    )
    if return_info:
        return X_hat, info.to_dict()
    return X_hat


def global_pca_denoise_variance(X, variance_threshold=0.9, center=True, return_info=True):
    """Denoise ``X`` using the smallest rank reaching a variance threshold."""

    X = _validate_matrix(X)
    variance_threshold = float(variance_threshold)
    if variance_threshold <= 0.0 or variance_threshold > 1.0:
        raise ValueError("variance_threshold must be in (0, 1]")

    _, _, _, _, cumulative = _svd_pca(X, bool(center))
    if cumulative.size == 0:
        raise ValueError("cannot select PCA rank for an empty spectrum")
    rank = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    rank = min(max(rank, 1), min(X.shape))
    X_hat, info = global_pca_denoise(X, rank, center=center, return_info=True)
    info["variance_threshold"] = variance_threshold
    info["selected_rank"] = rank
    if return_info:
        return X_hat, info
    return X_hat


def project_vectors_with_pca_info(V, info: dict[str, object]) -> np.ndarray:
    """Project vectors through the same retained PCA components as positions."""

    V = _validate_matrix(V, name="V")
    components = np.asarray(info["components"], dtype=float)
    if components.ndim != 2 or components.shape[1] != V.shape[1]:
        raise ValueError("PCA components are incompatible with V")
    return (V @ components.T) @ components


def local_pca_denoise(
    X,
    intrinsic_dim,
    n_neighbors=30,
    center=True,
    return_info=True,
    vectors=None,
):
    """Project points (and optional vectors) through pointwise local PCA bases.

    Neighbors are selected once from the observed positions.  Each covariance
    is centered at the mean of the neighbors (excluding the query point), and
    the affine point reconstruction adds that mean back.  ``vectors`` are
    projected through exactly the same pointwise tangent projectors.
    """

    from sklearn.neighbors import NearestNeighbors

    X = _validate_matrix(X)
    if vectors is not None:
        vectors = _validate_matrix(vectors, name="vectors")
        if vectors.shape != X.shape:
            raise ValueError("vectors must have the same shape as X")
    intrinsic_dim = int(intrinsic_dim)
    if intrinsic_dim < 1 or intrinsic_dim > X.shape[1]:
        raise ValueError("intrinsic_dim must be between 1 and X.shape[1]")
    n_neighbors = min(max(int(n_neighbors), intrinsic_dim + 1), X.shape[0] - 1)
    if n_neighbors < 1:
        raise ValueError("at least two points are required")

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    _, indices = nbrs.kneighbors(X)
    X_hat = np.empty_like(X)
    vectors_hat = None if vectors is None else np.empty_like(vectors)
    projectors = np.empty((X.shape[0], X.shape[1], X.shape[1]), dtype=float)
    spectra = np.zeros((X.shape[0], X.shape[1]), dtype=float)
    for i, neigh in enumerate(indices[:, 1:]):
        cloud = X[neigh]
        mean = cloud.mean(axis=0) if center else np.zeros(X.shape[1], dtype=float)
        centered = cloud - mean
        cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        basis = eigvecs[:, order[:intrinsic_dim]]
        projector = basis @ basis.T
        projectors[i] = projector
        spectra[i, : eigvals.size] = eigvals
        X_hat[i] = mean + (X[i] - mean) @ projector
        if vectors_hat is not None:
            vectors_hat[i] = vectors[i] @ projector

    info = {
        "intrinsic_dim": intrinsic_dim,
        "n_neighbors": n_neighbors,
        "mean_local_spectrum": spectra.mean(axis=0),
        "projectors": projectors,
        "neighbor_indices": indices[:, 1:].copy(),
    }
    if vectors_hat is not None:
        if return_info:
            return X_hat, vectors_hat, info
        return X_hat, vectors_hat
    if return_info:
        return X_hat, info
    return X_hat
