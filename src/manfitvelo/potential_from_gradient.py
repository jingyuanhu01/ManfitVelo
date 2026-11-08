"""Recover scalar potential values from a fitted manifold gradient field.

The main entry point, :func:`potential_from_gradient`, solves a sparse graph
least-squares problem on fitted manifold coordinates. It is intended as a
downstream step after ``VelocityManifoldFitter.fit(return_dict=True)``:

``X_fit, V_fit -> phi, grad_phi``.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
from sklearn.neighbors import NearestNeighbors


def _validate_inputs(X, gradient) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if gradient.shape != X.shape:
        raise ValueError("gradient must have the same shape as X")
    if X.shape[0] <= 1:
        raise ValueError("At least two points are required")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")
    if not np.all(np.isfinite(gradient)):
        raise ValueError("gradient contains non-finite values")
    return X, gradient


def _knn_graph(X: np.ndarray, n_neighbors: int, bandwidth: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = min(int(n_neighbors), X.shape[0] - 1)
    if k < 1:
        raise ValueError("n_neighbors must be at least 1")

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = distances[:, 1:]
    neighbors = indices[:, 1:]

    if bandwidth is None:
        bandwidth = float(np.median(distances[:, -1])) + 1e-12
    else:
        bandwidth = float(bandwidth)
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")

    weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    return neighbors, weights, distances


def _add_edge_rows(
    rows: list[int],
    cols: list[int],
    vals: list[float],
    rhs: list[float],
    X: np.ndarray,
    gradient: np.ndarray,
    neighbors: np.ndarray,
    weights: np.ndarray,
    sign: float,
    row_start: int,
) -> int:
    row = row_start
    for i in range(X.shape[0]):
        for local_idx, j in enumerate(neighbors[i]):
            weight = float(np.sqrt(weights[i, local_idx]))
            delta_x = X[j] - X[i]
            predicted_delta_phi = sign * float(np.dot(gradient[i], delta_x))
            rows.extend([row, row])
            cols.extend([i, j])
            vals.extend([-weight, weight])
            rhs.append(weight * predicted_delta_phi)
            row += 1
    return row


def _add_cell_consistency_rows(
    rows: list[int],
    cols: list[int],
    vals: list[float],
    rhs: list[float],
    X: np.ndarray,
    gradient: np.ndarray,
    neighbors: np.ndarray,
    weights: np.ndarray,
    sign: float,
    row_start: int,
    reg: float,
) -> int:
    if reg <= 0:
        return row_start

    row = row_start
    scale = float(np.sqrt(reg))
    normalized = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-12)
    for i in range(X.shape[0]):
        predicted_offset = 0.0
        rows.append(row)
        cols.append(i)
        vals.append(scale)
        for local_idx, j in enumerate(neighbors[i]):
            alpha = float(normalized[i, local_idx])
            delta_x = X[i] - X[j]
            predicted_offset += alpha * sign * float(np.dot(gradient[j], delta_x))
            rows.append(row)
            cols.append(int(j))
            vals.append(-scale * alpha)
        rhs.append(scale * predicted_offset)
        row += 1
    return row


def _add_laplacian_rows(
    rows: list[int],
    cols: list[int],
    vals: list[float],
    rhs: list[float],
    neighbors: np.ndarray,
    weights: np.ndarray,
    row_start: int,
    reg: float,
) -> int:
    if reg <= 0:
        return row_start

    row = row_start
    scale = float(np.sqrt(reg))
    normalized = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-12)
    for i in range(neighbors.shape[0]):
        rows.append(row)
        cols.append(i)
        vals.append(scale)
        for local_idx, j in enumerate(neighbors[i]):
            rows.append(row)
            cols.append(int(j))
            vals.append(-scale * float(normalized[i, local_idx]))
        rhs.append(0.0)
        row += 1
    return row


def _estimate_gradient_from_potential(
    X: np.ndarray,
    potential: np.ndarray,
    neighbors: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> np.ndarray:
    gradients = np.zeros_like(X)
    eye = np.eye(X.shape[1])
    for i in range(X.shape[0]):
        neigh = neighbors[i]
        dX = X[neigh] - X[i]
        dphi = potential[neigh] - potential[i]
        w = weights[i]
        lhs = (dX.T * w) @ dX + float(ridge) * eye
        rhs = dX.T @ (w * dphi)
        gradients[i] = np.linalg.solve(lhs, rhs)
    return gradients


def potential_from_gradient(
    X,
    gradient,
    *,
    n_neighbors: int = 25,
    sign: float = 1.0,
    bandwidth: float | None = None,
    cell_consistency_reg: float = 0.0,
    laplacian_reg: float = 1e-3,
    gradient_ridge: float = 1e-5,
    gauge_weight: float = 1.0,
    lsqr_atol: float = 1e-8,
    lsqr_btol: float = 1e-8,
    lsqr_iter_lim: int | None = None,
    return_system: bool = False,
) -> dict[str, object]:
    """Estimate a scalar potential from local gradient vectors.

    The solver uses directed kNN graph edges and fits potential values
    ``phi`` so that local potential differences agree with the input gradient:

    ``phi[j] - phi[i] ~= sign * dot(gradient[i], X[j] - X[i])``.

    If the input vectors are velocities from gradient descent, use
    ``sign=-1`` to recover a potential whose negative gradient follows the
    velocity. If the input vectors are true gradients, keep ``sign=1``.

    Parameters
    ----------
    X:
        Fitted point positions with shape ``(n_cells, n_features)``.
    gradient:
        Gradient-like vectors with the same shape as ``X``.
    n_neighbors:
        Number of neighbors used for graph edges.
    sign:
        Sign convention between the input vector and potential gradient.
    bandwidth:
        Gaussian edge-weight bandwidth. Defaults to the median kNN radius.
    cell_consistency_reg:
        Weight for the weighted-neighbor consistency constraint
        ``phi_i ~= sum_j alpha_ij (phi_j + sign * <g_j, x_i - x_j>)``.
    laplacian_reg:
        Weight for smooth graph-Laplacian regularization.
    gradient_ridge:
        Ridge penalty for recovering gradients from the fitted potential.
    gauge_weight:
        Weight for the mean-zero gauge constraint.
    return_system:
        If ``True``, include the sparse least-squares matrix and RHS.

    Returns
    -------
    dict
        Contains ``potential``, ``gradient``, ``residual``, ``neighbors``,
        ``weights``, and residual diagnostics.
    """
    X, gradient = _validate_inputs(X, gradient)
    neighbors, weights, distances = _knn_graph(X, n_neighbors, bandwidth)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs: list[float] = []
    row = 0

    row = _add_edge_rows(rows, cols, vals, rhs, X, gradient, neighbors, weights, float(sign), row)
    row = _add_cell_consistency_rows(
        rows,
        cols,
        vals,
        rhs,
        X,
        gradient,
        neighbors,
        weights,
        float(sign),
        row,
        float(cell_consistency_reg),
    )
    row = _add_laplacian_rows(rows, cols, vals, rhs, neighbors, weights, row, float(laplacian_reg))

    gauge_scale = float(np.sqrt(gauge_weight))
    if gauge_scale > 0:
        for i in range(X.shape[0]):
            rows.append(row)
            cols.append(i)
            vals.append(gauge_scale / X.shape[0])
        rhs.append(0.0)
        row += 1

    system = sparse.csr_matrix((vals, (rows, cols)), shape=(row, X.shape[0]))
    target = np.asarray(rhs, dtype=float)
    solution = lsqr(system, target, atol=lsqr_atol, btol=lsqr_btol, iter_lim=lsqr_iter_lim)
    potential = solution[0]
    potential = potential - np.mean(potential)

    recovered_gradient = _estimate_gradient_from_potential(
        X,
        potential,
        neighbors,
        weights,
        ridge=gradient_ridge,
    )
    residual = gradient - float(sign) * recovered_gradient

    edge_residuals = []
    edge_weights = []
    for i in range(X.shape[0]):
        for local_idx, j in enumerate(neighbors[i]):
            observed = potential[j] - potential[i]
            predicted = float(sign) * float(np.dot(gradient[i], X[j] - X[i]))
            edge_residuals.append(observed - predicted)
            edge_weights.append(weights[i, local_idx])
    edge_residuals = np.asarray(edge_residuals, dtype=float)
    edge_weights = np.asarray(edge_weights, dtype=float)

    result = {
        "potential": potential,
        "gradient": recovered_gradient,
        "residual": residual,
        "neighbors": neighbors,
        "weights": weights,
        "distances": distances,
        "edge_residual_rmse": float(
            np.sqrt(np.average(edge_residuals**2, weights=edge_weights + 1e-12))
        ),
        "gradient_residual_rmse": float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
        "conservative_ratio": float(
            np.linalg.norm(residual) / (np.linalg.norm(gradient) + 1e-12)
        ),
        "lsqr": {
            "istop": int(solution[1]),
            "iterations": int(solution[2]),
            "residual_norm": float(solution[3]),
            "condition_estimate": float(solution[6]),
        },
    }
    if return_system:
        result["system_matrix"] = system
        result["system_rhs"] = target
    return result
