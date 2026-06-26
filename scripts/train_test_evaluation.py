"""Train/test prediction helpers for MANFIT evaluation.

This module keeps held-out evaluation workflows separate from the core
``VelocityManifoldFitter`` implementation. The helpers here fit the manifold
on training data, place held-out points on the fitted training manifold with
LLE-style local weights, and estimate held-out velocities from the fitted
training vector field.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from scripts.velocity_manifold_fitter import VelocityManifoldFitter


def compute_lle_weights(query, reference, n_neighbors=15, reg=1e-3, eps=1e-12):
    """Represent query points by local linear weights on reference points.

    For each query point, this solves the standard LLE constrained
    least-squares problem over nearest reference neighbors. The returned
    weights sum to one row-wise and can copy any aligned reference quantity to
    the query points.
    """
    query = np.asarray(query, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if query.ndim != 2 or reference.ndim != 2:
        raise ValueError("query and reference must be two-dimensional arrays")
    if query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference must have the same feature dimension")
    if reference.shape[0] < 1:
        raise ValueError("reference must contain at least one point")

    n_neighbors = min(int(n_neighbors), reference.shape[0])
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1")

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(reference)
    distances, indices = nbrs.kneighbors(query)
    weights = np.zeros((query.shape[0], n_neighbors), dtype=float)

    for i in range(query.shape[0]):
        local = reference[indices[i]] - query[i]
        gram = local @ local.T
        trace = float(np.trace(gram))
        ridge = float(reg) * trace if trace > eps else float(reg)
        gram = gram + ridge * np.eye(n_neighbors)
        ones = np.ones(n_neighbors, dtype=float)
        try:
            wi = np.linalg.solve(gram, ones)
        except np.linalg.LinAlgError:
            wi = np.linalg.lstsq(gram, ones, rcond=None)[0]

        total = float(np.sum(wi))
        if abs(total) <= eps:
            wi = np.full(n_neighbors, 1.0 / n_neighbors, dtype=float)
        else:
            wi = wi / total
        weights[i] = wi

    return indices, weights, distances


def kernel_average_at(query, reference, values, n_neighbors=15, bandwidth=None, eps=1e-12):
    """Estimate values at query points by local Gaussian kernel averaging."""
    query = np.asarray(query, dtype=float)
    reference = np.asarray(reference, dtype=float)
    values = np.asarray(values, dtype=float)
    if query.ndim != 2 or reference.ndim != 2 or values.ndim != 2:
        raise ValueError("query, reference, and values must be two-dimensional arrays")
    if query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference must have the same feature dimension")
    if reference.shape[0] != values.shape[0]:
        raise ValueError("reference and values must have the same number of rows")
    if reference.shape[0] < 1:
        raise ValueError("reference must contain at least one point")

    n_neighbors = min(int(n_neighbors), reference.shape[0])
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1")

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(reference)
    distances, indices = nbrs.kneighbors(query)
    if bandwidth is None:
        h = distances[:, -1:] + eps
    else:
        h = np.full((query.shape[0], 1), float(bandwidth) + eps)

    weights = np.exp(-0.5 * (distances / h) ** 2)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    averaged = np.einsum("nk,nkd->nd", weights, values[indices])
    return averaged, indices, weights, distances


def fit_train_test_manifold_predictions(
    Y,
    W,
    test_size=0.2,
    random_state=0,
    train_indices=None,
    test_indices=None,
    stratify=None,
    fitter_kwargs=None,
    fit_kwargs=None,
    lle_neighbors=15,
    lle_reg=1e-3,
    velocity_neighbors=15,
    velocity_bandwidth=None,
    return_original_space=True,
):
    """Fit on training data and predict held-out position and velocity.

    This is evaluation scaffolding only; it does not change the manifold
    velocity fitting algorithm. It returns predictions without computing
    evaluation metrics.
    """
    Y = np.asarray(Y, dtype=float)
    W = np.asarray(W, dtype=float)
    if Y.shape != W.shape:
        raise ValueError("Y and W must match shape")
    if Y.ndim != 2:
        raise ValueError("Y and W must be two-dimensional arrays")

    n = Y.shape[0]
    if train_indices is None or test_indices is None:
        indices = np.arange(n)
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    if train_indices.ndim != 1 or test_indices.ndim != 1:
        raise ValueError("train_indices and test_indices must be one-dimensional")
    if train_indices.size < 2:
        raise ValueError("At least two training points are required")
    if test_indices.size < 1:
        raise ValueError("At least one testing point is required")

    fitter_kwargs = {} if fitter_kwargs is None else dict(fitter_kwargs)
    fitter_kwargs.setdefault("random_state", random_state)
    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    fit_kwargs.setdefault("return_dict", True)

    fitter = VelocityManifoldFitter(Y[train_indices], W[train_indices], **fitter_kwargs)
    fit_result = fitter.fit(**fit_kwargs)
    if not isinstance(fit_result, dict):
        raise ValueError("fit_kwargs must not disable return_dict")

    if fitter.global_pca is None:
        Y_test_model = Y[test_indices]
    else:
        Y_test_model = fitter.global_pca.transform(Y[test_indices])

    lle_indices, lle_weights, lle_distances = compute_lle_weights(
        Y_test_model,
        fitter.Y,
        n_neighbors=lle_neighbors,
        reg=lle_reg,
        eps=fitter.eps,
    )
    X_train_fit = fit_result["X"]
    V_train_fit = fit_result["V"]
    X_test_hat = np.einsum("nk,nkd->nd", lle_weights, X_train_fit[lle_indices])

    V_test_hat, velocity_indices, velocity_weights, velocity_distances = kernel_average_at(
        X_test_hat,
        X_train_fit,
        V_train_fit,
        n_neighbors=velocity_neighbors,
        bandwidth=velocity_bandwidth,
        eps=fitter.eps,
    )

    result = {
        "fitter": fitter,
        "fit_result": fit_result,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "Y_test_model": Y_test_model,
        "X_test_hat": X_test_hat,
        "V_test_hat": V_test_hat,
        "lle_indices": lle_indices,
        "lle_weights": lle_weights,
        "lle_distances": lle_distances,
        "velocity_indices": velocity_indices,
        "velocity_weights": velocity_weights,
        "velocity_distances": velocity_distances,
    }

    if return_original_space and fitter.global_pca is not None:
        result["X_test_hat_original"] = fitter.global_pca.inverse_transform(X_test_hat)
        result["V_test_hat_original"] = V_test_hat @ fitter.global_pca.components_
    elif return_original_space:
        result["X_test_hat_original"] = X_test_hat
        result["V_test_hat_original"] = V_test_hat

    return result
