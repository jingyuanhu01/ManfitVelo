"""Thin adapter for GraphVelo's official analytical-manifold call path.

The project deliberately vendors only the small numerical core used by the
official ``notebook/simulation/bif_ball3D.ipynb``.  This avoids GraphVelo's
AnnData/count-data wrapper and its optional single-cell dependency stack while
preserving the notebook sequence exactly: 15-nearest-neighbor indices
(including the query point, as in sklearn), cosine correlation, official row
density correction, tangent-space projection, and graph reconstruction.

Upstream: https://github.com/xing-lab-pitt/GraphVelo
Pinned commit: 0d2bb4e69b3632fe075963753efa913c51930d71
Release/package line: GraphVelo 0.1.11 (repository release v0.1.0)
Source files: graphvelo/tangent_space.py, graphvelo/graph_velocity.py
Notebook: notebook/simulation/bif_ball3D.ipynb, cells 11--12
License: BSD-3-Clause
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors

from scripts.simulation_baselines import neighbor_graph_hash


GRAPHVELO_CONFIG = {
    "n_neighbors": 15,
    "a": 1.0,
    "b": 0.0,
    "r": 1.0,
    "loss_func": "linear",
    "softmax_adjusted": False,
    "approx": False,
    "preprocessing": "raw continuous coordinates; no log/log1p, normalization, or PCA",
}

GRAPHVELO_PROVENANCE = {
    "package": "GraphVelo",
    "package_version": "0.1.11",
    "repository_release": "v0.1.0",
    "commit": "0d2bb4e69b3632fe075963753efa913c51930d71",
    "source": "https://github.com/xing-lab-pitt/GraphVelo",
    "notebook": "notebook/simulation/bif_ball3D.ipynb",
    "vendored_functions": [
        "cos_corr",
        "corr_kernel",
        "density_corrected_transition_matrix",
        "regression_phi/tangent_space_projection",
        "project_velocity",
    ],
}

GRAPHVELO_STANDARDIZATION = {
    "position_center": "coordinate-wise noisy-position mean",
    "position_scale": "median positive displacement norm over the official 15-NN graph",
    "velocity_scale": "median noisy velocity norm",
    "truth_free": True,
    "selected_by_performance": False,
    "inverse_map": "V_hat = velocity_scale * V_hat_standardized",
}


def official_neighbors(X: np.ndarray, n_neighbors: int = 15) -> np.ndarray:
    """Match the notebook's ``kneighbors(X)`` call, including self."""
    X = np.asarray(X, dtype=float)
    if len(X) < n_neighbors:
        raise ValueError("GraphVelo requires at least n_neighbors observations")
    model = NearestNeighbors(n_neighbors=int(n_neighbors)).fit(X)
    return model.kneighbors(X, return_distance=False).astype(np.int64, copy=False)


def official_cosine_kernel(X: np.ndarray, V: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """Vendored ``corr_kernel(..., corr_func=cos_corr)``."""
    X, V = np.asarray(X, float), np.asarray(V, float)
    P = np.zeros((len(X), len(X)), dtype=float)
    for i, idx in enumerate(np.asarray(neighbors, int)):
        displacement = X[idx] - X[i]
        distance = np.linalg.norm(displacement, axis=1)
        distance[distance == 0] = 1
        direction = displacement / distance[:, None]
        speed = np.linalg.norm(V[i])
        if speed == 0:
            speed = 1
        P[i, idx] = direction @ V[i] / speed
    return P


def official_density_correction(P: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """Vendored sparse-row mean subtraction used by GraphVelo."""
    corrected = np.zeros_like(np.asarray(P, float))
    for i, idx in enumerate(np.asarray(neighbors, int)):
        values = P[i, idx].copy()
        values -= values.mean()
        corrected[i, idx] = values
    return corrected


def _official_regression_phi(
    X: np.ndarray,
    V: np.ndarray,
    C: np.ndarray,
    neighbors: np.ndarray,
    i: int,
    *,
    a: float,
    b: float,
    r: float,
    loss_func: str,
) -> tuple[np.ndarray, bool, int]:
    """Literal serial form of upstream ``regression_phi``."""
    x, velocity, idx = X[i], V[i], neighbors[i]
    c = C[i, idx]
    displacement = X[idx] - x
    c_norm = np.linalg.norm(c)

    def objective(weights: np.ndarray) -> float:
        reconstructed = weights @ displacement
        residual = reconstructed - velocity
        reconstruction = residual @ residual
        if loss_func == "log":
            reconstruction = np.log(max(reconstruction, np.finfo(float).tiny))
        elif loss_func != "linear":
            raise NotImplementedError(loss_func)
        w_norm = np.linalg.norm(weights)
        similarity = c @ weights / (c_norm * w_norm) if b and c_norm and w_norm else 0.0
        regularization = weights @ weights if r else 0.0
        return float(a * reconstruction - b * similarity + r * regularization)

    def gradient(weights: np.ndarray) -> np.ndarray:
        residual = weights @ displacement - velocity
        value = 2 * a * displacement @ residual
        if loss_func == "log":
            value /= max(residual @ residual, np.finfo(float).tiny)
        w_norm = np.linalg.norm(weights)
        if b and c_norm and w_norm:
            value -= b * (c / (w_norm * c_norm) - (weights @ c) * weights / (w_norm**3 * c_norm))
        if r:
            value += 2 * r * weights
        return value

    result = minimize(objective, x0=c, jac=gradient)
    return np.asarray(result.x), bool(result.success), int(result.nit)


def official_tangent_space_projection(
    X: np.ndarray,
    V: np.ndarray,
    C: np.ndarray,
    neighbors: np.ndarray,
    *,
    a: float = 1.0,
    b: float = 0.0,
    r: float = 1.0,
    loss_func: str = "linear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Serial deterministic equivalent of upstream tangent-space projection."""
    coefficients = np.zeros((len(X), len(X)), dtype=float)
    success = np.zeros(len(X), dtype=bool)
    iterations = np.zeros(len(X), dtype=int)
    for i, idx in enumerate(neighbors):
        weights, success[i], iterations[i] = _official_regression_phi(
            X, V, C, neighbors, i, a=a, b=b, r=r, loss_func=loss_func
        )
        coefficients[i, idx] = weights
    return coefficients, success, iterations


def official_project_velocity(X: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Vendored notebook ``project_velocity(X_embedding, T)``."""
    X, coefficients = np.asarray(X, float), np.asarray(coefficients, float)
    output = np.zeros_like(X)
    for i in range(len(X)):
        idx = np.flatnonzero(coefficients[i] != 0)
        output[i] = coefficients[i, idx] @ (X[idx] - X[i])
    return output


def official_notebook_call(X: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, dict]:
    """Execute the pinned analytical-manifold notebook path directly."""
    X, V = np.asarray(X, float), np.asarray(V, float)
    neighbors = official_neighbors(X, GRAPHVELO_CONFIG["n_neighbors"])
    kernel = official_cosine_kernel(X, V, neighbors)
    corrected = official_density_correction(kernel, neighbors)
    coefficients, success, iterations = official_tangent_space_projection(
        X, V, corrected, neighbors,
        a=GRAPHVELO_CONFIG["a"], b=GRAPHVELO_CONFIG["b"],
        r=GRAPHVELO_CONFIG["r"], loss_func=GRAPHVELO_CONFIG["loss_func"],
    )
    velocity = official_project_velocity(X, coefficients)
    return velocity, {
        "neighbor_graph_hash": neighbor_graph_hash(neighbors),
        "optimizer_success_fraction": float(success.mean()),
        "optimizer_median_iterations": float(np.median(iterations)),
        **GRAPHVELO_CONFIG,
    }


def graphvelo_velocity(X: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, dict]:
    """Raw-scale official benchmark adapter; positions remain unchanged."""
    estimate, info = official_notebook_call(X, V)
    if estimate.shape != np.asarray(V).shape or not np.all(np.isfinite(estimate)):
        raise RuntimeError("official GraphVelo path returned an invalid velocity")
    return estimate, info


def noisy_standardization_scales(X: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return the fixed truth-free centering and scales for standardized TSP."""
    X, V = np.asarray(X, float), np.asarray(V, float)
    neighbors = official_neighbors(X, GRAPHVELO_CONFIG["n_neighbors"])
    distances = np.linalg.norm(X[neighbors] - X[:, None, :], axis=2)
    positive_distance = distances[distances > np.finfo(float).eps]
    positive_speed = np.linalg.norm(V, axis=1)
    positive_speed = positive_speed[positive_speed > np.finfo(float).eps]
    if not positive_distance.size or not positive_speed.size:
        raise ValueError("GraphVelo standardization requires nonzero neighbor distances and velocities")
    return X.mean(axis=0), float(np.median(positive_distance)), float(np.median(positive_speed))


def graphvelo_velocity_standardized(X: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, dict]:
    """Run the unchanged official TSP objective in fixed noisy-data units.

    The official parameters are not altered. Only a globally specified,
    truth-free unit conversion is applied before the call and inverted after
    graph velocity reconstruction.
    """
    X, V = np.asarray(X, float), np.asarray(V, float)
    center, position_scale, velocity_scale = noisy_standardization_scales(X, V)
    standardized_X = (X - center) / position_scale
    standardized_V = V / velocity_scale
    standardized_estimate, info = official_notebook_call(standardized_X, standardized_V)
    estimate = velocity_scale * standardized_estimate
    if estimate.shape != V.shape or not np.all(np.isfinite(estimate)):
        raise RuntimeError("standardized official GraphVelo path returned an invalid velocity")
    return estimate, {
        **info,
        "standardization": "median_15nn_position_distance_and_median_noisy_velocity_norm",
        "position_scale": position_scale,
        "velocity_scale": velocity_scale,
        "position_center_norm": float(np.linalg.norm(center)),
        "truth_free_standardization": True,
        "selected_by_performance": False,
    }
