"""Geometry-aware and velocity-aware benchmark metrics."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


EPS = 1e-12


def _as_2d(X, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{name} must have nonzero shape")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return X


def _matching(X, Y, x_name="X", y_name="Y") -> tuple[np.ndarray, np.ndarray]:
    X = _as_2d(X, x_name)
    Y = _as_2d(Y, y_name)
    if X.shape != Y.shape:
        raise ValueError(f"{x_name} and {y_name} must have matching shape")
    return X, Y


def _safe_neighbors(X: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    n_neighbors = min(max(int(n_neighbors), 1), X.shape[0] - 1)
    if n_neighbors < 1:
        raise ValueError("at least two points are required")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    return distances[:, 1:], indices[:, 1:]


def nan_summary(values, quantiles: dict[str, float] | None = None) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    result = {
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "median": float(np.median(finite)) if finite.size else float("nan"),
    }
    for name, q in (quantiles or {}).items():
        result[name] = float(np.quantile(finite, q)) if finite.size else float("nan")
    return result


def mse_to_clean(X_hat, X_clean) -> float:
    X_hat, X_clean = _matching(X_hat, X_clean, "X_hat", "X_clean")
    return float(np.mean(np.sum((X_hat - X_clean) ** 2, axis=1)))


def rmse_to_clean(X_hat, X_clean) -> float:
    return float(np.sqrt(mse_to_clean(X_hat, X_clean)))


def mean_l2_to_clean(X_hat, X_clean) -> float:
    X_hat, X_clean = _matching(X_hat, X_clean, "X_hat", "X_clean")
    return float(np.mean(np.linalg.norm(X_hat - X_clean, axis=1)))


def _distances_to_clean_cloud(X_hat, X_clean) -> np.ndarray:
    X_hat = _as_2d(X_hat, "X_hat")
    X_clean = _as_2d(X_clean, "X_clean")
    if X_hat.shape[1] != X_clean.shape[1]:
        raise ValueError("X_hat and X_clean must have the same feature dimension")
    nbrs = NearestNeighbors(n_neighbors=1).fit(X_clean)
    distances, _ = nbrs.kneighbors(X_hat)
    return distances[:, 0]


def mean_distance_to_clean_cloud(X_hat, X_clean) -> float:
    return float(np.mean(_distances_to_clean_cloud(X_hat, X_clean)))


def median_distance_to_clean_cloud(X_hat, X_clean) -> float:
    return float(np.median(_distances_to_clean_cloud(X_hat, X_clean)))


def quantile_distance_to_clean_cloud(X_hat, X_clean, q=0.9) -> float:
    return float(np.quantile(_distances_to_clean_cloud(X_hat, X_clean), float(q)))


def local_covariance_spectrum(X, n_neighbors=30) -> np.ndarray:
    X = _as_2d(X, "X")
    _, indices = _safe_neighbors(X, n_neighbors)
    spectra = np.zeros((X.shape[0], X.shape[1]), dtype=float)
    for i, neigh in enumerate(indices):
        centered = X[neigh] - X[neigh].mean(axis=0)
        cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
        eigvals = np.linalg.eigvalsh(cov)
        spectra[i] = np.maximum(eigvals[::-1], 0.0)
    return spectra


def normal_energy_ratio(X, intrinsic_dim, n_neighbors=30) -> dict[str, float]:
    spectra = local_covariance_spectrum(X, n_neighbors=n_neighbors)
    d = min(max(int(intrinsic_dim), 1), spectra.shape[1])
    total = np.sum(spectra, axis=1) + EPS
    ratios = np.sum(spectra[:, d:], axis=1) / total
    return nan_summary(ratios, {"q90": 0.9})


def tangent_energy_ratio(X, intrinsic_dim, n_neighbors=30) -> dict[str, float]:
    spectra = local_covariance_spectrum(X, n_neighbors=n_neighbors)
    d = min(max(int(intrinsic_dim), 1), spectra.shape[1])
    total = np.sum(spectra, axis=1) + EPS
    ratios = np.sum(spectra[:, :d], axis=1) / total
    return nan_summary(ratios, {"q10": 0.1})


def local_spectral_gap(X, intrinsic_dim, n_neighbors=30) -> dict[str, float]:
    spectra = local_covariance_spectrum(X, n_neighbors=n_neighbors)
    d = min(max(int(intrinsic_dim), 1), spectra.shape[1])
    if d >= spectra.shape[1]:
        gaps = np.full(spectra.shape[0], np.nan)
    else:
        gaps = spectra[:, d - 1] / (spectra[:, d] + EPS)
    return nan_summary(gaps, {"q10": 0.1})


def effective_dimension_from_spectrum(eigenvalues, eps=EPS):
    eigenvalues = np.maximum(np.asarray(eigenvalues, dtype=float), 0.0)
    numerator = np.sum(eigenvalues, axis=-1) ** 2
    denominator = np.sum(eigenvalues**2, axis=-1) + float(eps)
    return numerator / denominator


def local_effective_dimension(X, n_neighbors=30) -> dict[str, float]:
    spectra = local_covariance_spectrum(X, n_neighbors=n_neighbors)
    deff = effective_dimension_from_spectrum(spectra)
    return nan_summary(deff, {"q10": 0.1, "q90": 0.9})


def estimate_local_projectors(X, intrinsic_dim, n_neighbors=30) -> np.ndarray:
    X = _as_2d(X, "X")
    d = min(max(int(intrinsic_dim), 1), X.shape[1])
    _, indices = _safe_neighbors(X, n_neighbors)
    projectors = np.zeros((X.shape[0], X.shape[1], X.shape[1]), dtype=float)
    for i, neigh in enumerate(indices):
        centered = X[neigh] - X[neigh].mean(axis=0)
        cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        basis = eigvecs[:, order[:d]]
        projectors[i] = basis @ basis.T
    return projectors


def tangent_projector_error(P_hat, P_true) -> dict[str, float]:
    P_hat = np.asarray(P_hat, dtype=float)
    P_true = np.asarray(P_true, dtype=float)
    if P_hat.shape != P_true.shape or P_hat.ndim != 3:
        raise ValueError("P_hat and P_true must have matching shape (n, D, D)")
    errors = np.linalg.norm(P_hat - P_true, axis=(1, 2))
    return nan_summary(errors, {"q90": 0.9})


def _cosine_rows(A, B, eps=EPS) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    denom = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    out = np.full(A.shape[0], np.nan, dtype=float)
    valid = denom > eps
    out[valid] = np.sum(A[valid] * B[valid], axis=1) / denom[valid]
    return np.clip(out, -1.0, 1.0)


def _zero_velocity_fraction(V, eps=EPS) -> float:
    norms = np.linalg.norm(V, axis=1)
    return float(np.mean(norms <= eps))


def velocity_tangent_alignment(X, V, intrinsic_dim, n_neighbors=30) -> dict[str, float]:
    X, V = _matching(X, V, "X", "V")
    projectors = estimate_local_projectors(X, intrinsic_dim, n_neighbors=n_neighbors)
    projected = np.einsum("nij,nj->ni", projectors, V)
    denom = np.linalg.norm(V, axis=1)
    align = np.full(X.shape[0], np.nan, dtype=float)
    valid = denom > EPS
    align[valid] = np.linalg.norm(projected[valid], axis=1) / denom[valid]
    result = nan_summary(align, {"q10": 0.1})
    result["zero_velocity_fraction"] = _zero_velocity_fraction(V)
    if result["zero_velocity_fraction"] > 0.1:
        result["warning"] = f"{result['zero_velocity_fraction']:.1%} of velocity vectors are near zero"
    return result


def velocity_neighbor_direction_agreement(X, V, n_neighbors=30) -> dict[str, float]:
    X, V = _matching(X, V, "X", "V")
    _, indices = _safe_neighbors(X, n_neighbors)
    scores = np.full(X.shape[0], np.nan, dtype=float)
    for i, neigh in enumerate(indices):
        displacements = X[neigh] - X[i]
        velocities = np.repeat(V[i][None, :], neigh.size, axis=0)
        cosines = _cosine_rows(velocities, displacements)
        if np.any(np.isfinite(cosines)):
            scores[i] = np.nanmax(cosines)
    result = nan_summary(scores, {"q10": 0.1})
    result["zero_velocity_fraction"] = _zero_velocity_fraction(V)
    return result


def velocity_smoothness(X, V, n_neighbors=30) -> dict[str, float]:
    X, V = _matching(X, V, "X", "V")
    _, indices = _safe_neighbors(X, n_neighbors)
    scores = np.full(X.shape[0], np.nan, dtype=float)
    for i, neigh in enumerate(indices):
        velocities = np.repeat(V[i][None, :], neigh.size, axis=0)
        cosines = _cosine_rows(velocities, V[neigh])
        if np.any(np.isfinite(cosines)):
            scores[i] = np.nanmean(cosines)
    result = nan_summary(scores)
    result["zero_velocity_fraction"] = _zero_velocity_fraction(V)
    return result


def displacement_summary(X_before, X_after, n_neighbors=30) -> dict[str, float]:
    X_before, X_after = _matching(X_before, X_after, "X_before", "X_after")
    displacements = np.linalg.norm(X_after - X_before, axis=1)
    local_distances, _ = _safe_neighbors(X_before, n_neighbors)
    local_scale = np.median(local_distances[:, -1]) + EPS
    summary = nan_summary(displacements, {"q90": 0.9})
    summary["relative_to_local_scale_mean"] = float(summary["mean"] / local_scale)
    summary["relative_to_local_scale_q90"] = float(summary["q90"] / local_scale)
    return summary


def knn_overlap(X_before, X_after, n_neighbors=30) -> dict[str, float]:
    X_before, X_after = _matching(X_before, X_after, "X_before", "X_after")
    _, before_idx = _safe_neighbors(X_before, n_neighbors)
    _, after_idx = _safe_neighbors(X_after, n_neighbors)
    scores = []
    for left, right in zip(before_idx, after_idx):
        scores.append(len(set(left).intersection(set(right))) / float(n_neighbors))
    return nan_summary(np.asarray(scores), {"q10": 0.1})


def pairwise_distance_correlation(X_before, X_after, max_points=600, random_state=0) -> float:
    X_before, X_after = _matching(X_before, X_after, "X_before", "X_after")
    n = X_before.shape[0]
    if n > int(max_points):
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(n, size=int(max_points), replace=False))
        X_before = X_before[idx]
        X_after = X_after[idx]
    d0 = pdist(X_before)
    d1 = pdist(X_after)
    if np.std(d0) <= EPS or np.std(d1) <= EPS:
        return float("nan")
    return float(np.corrcoef(d0, d1)[0, 1])


def trustworthiness_score(X_before, X_after, n_neighbors=30) -> float:
    X_before, X_after = _matching(X_before, X_after, "X_before", "X_after")
    try:
        from sklearn.manifold import trustworthiness
    except ImportError:
        warnings.warn("sklearn.manifold.trustworthiness is unavailable", RuntimeWarning)
        return float("nan")
    n_neighbors = min(max(int(n_neighbors), 1), X_before.shape[0] - 1)
    return float(trustworthiness(X_before, X_after, n_neighbors=n_neighbors))


def average_local_spectrum(X, n_neighbors=30) -> np.ndarray:
    return np.nanmean(local_covariance_spectrum(X, n_neighbors=n_neighbors), axis=0)


def metric_dict(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items() if key != "warning"}
