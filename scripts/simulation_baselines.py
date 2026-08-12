"""Shared baseline pipelines used by both formal simulation entry points."""

from __future__ import annotations

import hashlib

import numpy as np
from sklearn.neighbors import NearestNeighbors

from scripts.pca_denoisers import (
    global_pca_denoise,
    local_pca_denoise,
    project_vectors_with_pca_info,
)


EPS = 1e-12


def shared_knn_graph(X: np.ndarray, k: int) -> np.ndarray:
    X = np.asarray(X, float)
    k = min(int(k), len(X) - 1)
    raw = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)
    return np.asarray([row[row != i][:k] for i, row in enumerate(raw)], dtype=np.int64)


def neighbor_graph_hash(neighbors: np.ndarray) -> str:
    value = np.ascontiguousarray(neighbors, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def cosine_kernel_projection(
    X: np.ndarray, V: np.ndarray, neighbors: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Official cosine kernel plus density correction on a supplied graph."""
    X, V, neighbors = np.asarray(X, float), np.asarray(V, float), np.asarray(neighbors, int)
    difference = X[neighbors] - X[:, None, :]
    distance = np.linalg.norm(difference, axis=2)
    speed = np.linalg.norm(V, axis=1)
    valid = (distance > EPS) & (speed[:, None] > EPS)
    cosine = np.zeros(distance.shape)
    numerator = np.sum(V[:, None, :] * difference, axis=2)
    cosine[valid] = numerator[valid] / (distance * speed[:, None])[valid]
    coefficients = cosine - cosine.mean(axis=1, keepdims=True)
    estimate = np.einsum("nk,nkd->nd", coefficients, difference)
    return estimate, {"neighbor_graph_hash": neighbor_graph_hash(neighbors), "density_corrected": True}


def restore_noisy_speed(direction: np.ndarray, noisy_velocity: np.ndarray) -> tuple[np.ndarray, dict]:
    direction, noisy_velocity = np.asarray(direction, float), np.asarray(noisy_velocity, float)
    magnitude = np.linalg.norm(direction, axis=1)
    noisy_magnitude = np.linalg.norm(noisy_velocity, axis=1)
    fallback = magnitude <= EPS
    unit = np.zeros_like(direction)
    unit[~fallback] = direction[~fallback] / magnitude[~fallback, None]
    usable = fallback & (noisy_magnitude > EPS)
    unit[usable] = noisy_velocity[usable] / noisy_magnitude[usable, None]
    return noisy_magnitude[:, None] * unit, {"speed_restoration_fallback_fraction": float(fallback.mean())}


def global_pca_state(X: np.ndarray, V: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Centered fixed-rank PCA: Xbar+(X-Xbar)P and V P."""
    Xhat, info = global_pca_denoise(X, rank, center=True, return_info=True)
    Vhat = project_vectors_with_pca_info(V, info)
    components = np.asarray(info["components"])
    projector = components.T @ components
    return Xhat, Vhat, {**info, "projector": projector}


def downstream_velocity(X: np.ndarray, V: np.ndarray, rank: int, k: int) -> tuple[np.ndarray, dict]:
    _, Vhat, info = local_pca_denoise(X, rank, n_neighbors=k, vectors=V, return_info=True)
    return Vhat, {
        "downstream_rule": "recomputed_local_tangent_projection",
        "downstream_k": int(k),
        "downstream_graph_hash": neighbor_graph_hash(info["neighbor_indices"]),
        "projectors": info["projectors"],
        "mean_local_spectrum": info["mean_local_spectrum"],
    }


def local_pca_state(X: np.ndarray, V: np.ndarray, rank: int, k: int) -> tuple[np.ndarray, np.ndarray, dict]:
    Xhat, position_info = local_pca_denoise(X, rank, n_neighbors=k, return_info=True)
    Vhat, velocity_info = downstream_velocity(Xhat, V, rank, k)
    return Xhat, Vhat, {"position_info": position_info, **velocity_info}


JOINT_LOW_RANK_VARIANCE_THRESHOLD = 0.90


def joint_low_rank_state(
    X: np.ndarray, V: np.ndarray, *, variance_threshold: float = JOINT_LOW_RANK_VARIANCE_THRESHOLD
) -> tuple[np.ndarray, np.ndarray, dict]:
    """M3 Joint Low-Rank Denoising (Weekly Plan v1.1 section 4 "Joint Low-Rank").

    Block-normalizes position and velocity by their own Frobenius norm before
    concatenating them into one matrix, so neither block can dominate the
    rank selection purely because of its ambient scale; truncates by a fixed
    cumulative-explained-variance threshold (never by ground truth); inverts
    with an exact affine unscale (the forward map is purely linear, so no
    approximation is introduced). No neighborhood graph, no ground truth, no
    per-scenario tuning -- the threshold is the only frozen constant, shared
    across every scenario.
    """
    X, V = np.asarray(X, float), np.asarray(V, float)
    x_mean, v_mean = X.mean(axis=0), V.mean(axis=0)
    x_centered, v_centered = X - x_mean, V - v_mean
    x_scale = float(np.linalg.norm(x_centered))
    v_scale = float(np.linalg.norm(v_centered))
    x_scale = x_scale if x_scale > EPS else 1.0
    v_scale = v_scale if v_scale > EPS else 1.0
    joint = np.concatenate([x_centered / x_scale, v_centered / v_scale], axis=1)

    U, singular_values, Vt = np.linalg.svd(joint, full_matrices=False)
    energy = singular_values**2
    cumulative = np.cumsum(energy) / max(float(np.sum(energy)), EPS)
    rank = int(np.searchsorted(cumulative, variance_threshold) + 1)
    rank = min(rank, len(singular_values))

    joint_hat = (U[:, :rank] * singular_values[:rank]) @ Vt[:rank, :]
    ambient = X.shape[1]
    Xhat = joint_hat[:, :ambient] * x_scale + x_mean
    Vhat = joint_hat[:, ambient:] * v_scale + v_mean
    return Xhat, Vhat, {
        "rank": rank,
        "variance_threshold": variance_threshold,
        "explained_variance_at_rank": float(cumulative[rank - 1]),
        "n_singular_values": len(singular_values),
        "x_scale": x_scale,
        "v_scale": v_scale,
        "rank_uses_ground_truth": False,
    }
