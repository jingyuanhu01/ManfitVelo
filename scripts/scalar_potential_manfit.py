"""Scalar-potential informed manifold fitting utilities.

These routines implement the conservative scalar-potential idea from the
notes: potential-aware local neighborhoods. They are intentionally small
experimental baselines for the S-curve notebook rather than a full production
estimator.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from scripts.velocity_manifold_fitter import VelocityManifoldFitter


def normalize_rows(X, eps=1e-12):
    X = np.asarray(X, dtype=float)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def estimate_gradient_from_neighbors(X, values, n_neighbors=42, ridge=5e-2):
    """Estimate an ambient gradient field from positions and scalar values."""
    X = np.asarray(X, dtype=float)
    values = np.asarray(values, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if values.shape[0] != X.shape[0]:
        raise ValueError("values must have one entry per row of X")

    n_neighbors = min(int(n_neighbors), X.shape[0] - 1)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    _, indices = nbrs.kneighbors(X)
    gradients = np.zeros_like(X)
    eye = np.eye(X.shape[1])

    for i, neigh in enumerate(indices[:, 1:]):
        dX = X[neigh] - X[i]
        df = values[neigh] - values[i]
        gradients[i] = np.linalg.solve(dX.T @ dX + ridge * eye, dX.T @ df)

    return gradients


def estimate_gradient_confidence_from_neighbors(
    X,
    values,
    n_neighbors=42,
    ridge=5e-2,
    condition_scale=30.0,
):
    """Estimate local scalar gradients and a regression-quality confidence.

    Confidence combines clipped local R^2 with a condition-number penalty for
    the local design matrix. It is used as a soft trust weight for the scalar
    gradient as an auxiliary tangent signal.
    """
    X = np.asarray(X, dtype=float)
    values = np.asarray(values, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if values.shape[0] != X.shape[0]:
        raise ValueError("values must have one entry per row of X")

    n_neighbors = min(int(n_neighbors), X.shape[0] - 1)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    _, indices = nbrs.kneighbors(X)
    gradients = np.zeros_like(X)
    confidence = np.zeros(X.shape[0], dtype=float)
    eye = np.eye(X.shape[1])

    for i, neigh in enumerate(indices[:, 1:]):
        dX = X[neigh] - X[i]
        df = values[neigh] - values[i]
        lhs = dX.T @ dX + ridge * eye
        gradient = np.linalg.solve(lhs, dX.T @ df)
        gradients[i] = gradient

        fitted = dX @ gradient
        ss_res = float(np.sum((df - fitted) ** 2))
        ss_tot = float(np.sum((df - np.mean(df)) ** 2)) + 1e-12
        r2 = np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0)

        singular_values = np.linalg.svd(dX, compute_uv=False)
        cond = singular_values[0] / (singular_values[-1] + 1e-12)
        condition_score = 1.0 / (1.0 + cond / float(condition_scale))
        confidence[i] = r2 * condition_score

    if np.max(confidence) > 0:
        confidence = confidence / np.max(confidence)
    return gradients, confidence


def _weighted_local_pca_basis(dX, weights, tangent_dim=2):
    weights = np.asarray(weights, dtype=float)
    weighted_cov = (dX.T * weights) @ dX / (np.sum(weights) + 1e-12)
    eigvals, eigvecs = np.linalg.eigh(weighted_cov)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order[:tangent_dim]]


def _normal_candidate_grid(n_candidates=96):
    """Deterministic approximately uniform normals on the 3D sphere."""
    idx = np.arange(int(n_candidates), dtype=float) + 0.5
    z = 1.0 - 2.0 * idx / float(n_candidates)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    theta = np.pi * (1.0 + np.sqrt(5.0)) * idx
    return np.c_[radius * np.cos(theta), radius * np.sin(theta), z]


def _plane_basis_from_normal(normal):
    normal = np.asarray(normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    _, _, vh = np.linalg.svd(normal.reshape(1, -1), full_matrices=True)
    return vh[1:].T


def _tangent_constrained_basis(
    dX,
    dY,
    weights,
    *,
    scalar_lambda=1.0,
    scalar_error_scale=1.0,
    ridge=1e-3,
    candidate_normals=None,
):
    """Select a 2D tangent plane by profiling out local scalar slopes."""
    if dX.shape[1] != 3:
        raise ValueError("The tangent-constrained baseline expects 3D inputs")

    pca_basis = _weighted_local_pca_basis(dX, weights, tangent_dim=2)
    pca_normal = np.cross(pca_basis[:, 0], pca_basis[:, 1])
    pca_normal /= np.linalg.norm(pca_normal) + 1e-12
    normals = np.vstack([pca_normal, -pca_normal, candidate_normals])

    best_loss = np.inf
    best_basis = pca_basis
    for normal in normals:
        U = _plane_basis_from_normal(normal)
        coords = dX @ U
        lhs = (coords.T * weights) @ coords + ridge * np.eye(U.shape[1])
        rhs = coords.T @ (weights * dY)
        slope = np.linalg.solve(lhs, rhs)
        residual = dY - coords @ slope
        normal_error = np.sum(weights * (dX @ normal) ** 2)
        scalar_error = np.sum(weights * residual**2) / (scalar_error_scale + 1e-12)
        loss = normal_error + scalar_lambda * scalar_error
        if loss < best_loss:
            best_loss = loss
            best_basis = U

    return best_basis


def _local_geometry_fit(
    X,
    scalar,
    *,
    k=32,
    T=5,
    eta=0.65,
    tangent_dim=2,
    use_potential_weights=False,
    use_tangent_constraint=False,
    scalar_lambda=1.0,
    scalar_bandwidth=None,
    tangent_candidate_normals=96,
    ridge=1e-3,
    random_state=0,
):
    """Iterative normal-only smoothing with optional scalar-potential terms."""
    del random_state
    X = np.asarray(X, dtype=float)
    scalar = np.asarray(scalar, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if scalar.shape[0] != X.shape[0]:
        raise ValueError("scalar must have one entry per row of X")
    if use_tangent_constraint and (X.shape[1] != 3 or tangent_dim != 2):
        raise ValueError("The tangent-constrained baseline supports 2D planes in 3D")

    Z = X.copy()
    k = min(int(k), X.shape[0] - 1)
    candidate_normals = _normal_candidate_grid(tangent_candidate_normals)
    if scalar_bandwidth is None:
        scalar_bandwidth = np.median(np.abs(scalar - np.median(scalar))) + 1e-12
    scalar_error_scale = scalar_bandwidth**2

    for _ in range(int(T)):
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(Z)
        distances, indices = nbrs.kneighbors(Z)
        spatial_bandwidth = np.median(distances[:, -1]) + 1e-12
        Z_next = Z.copy()

        for i in range(Z.shape[0]):
            neigh = indices[i, 1:]
            dists = distances[i, 1:]
            weights = np.exp(-0.5 * (dists / spatial_bandwidth) ** 2)
            if use_potential_weights:
                dscalar_abs = np.abs(scalar[neigh] - scalar[i])
                weights *= np.exp(-0.5 * (dscalar_abs / scalar_bandwidth) ** 2)
            weights = weights / (np.sum(weights) + 1e-12)

            center = weights @ Z[neigh]
            dX_centered = Z[neigh] - center
            if use_tangent_constraint:
                basis = _tangent_constrained_basis(
                    Z[neigh] - Z[i],
                    scalar[neigh] - scalar[i],
                    weights,
                    scalar_lambda=scalar_lambda,
                    scalar_error_scale=scalar_error_scale,
                    ridge=ridge,
                    candidate_normals=candidate_normals,
                )
            else:
                basis = _weighted_local_pca_basis(dX_centered, weights, tangent_dim=tangent_dim)

            normal_component = (np.eye(Z.shape[1]) - basis @ basis.T) @ (Z[i] - center)
            Z_next[i] = Z[i] - eta * normal_component

        Z = Z_next

    gradient = estimate_gradient_from_neighbors(Z, scalar, n_neighbors=42, ridge=5e-2)
    return {"position": Z, "gradient": gradient}


def fit_potential_aware_neighborhoods(X, scalar, **kwargs):
    """Fit using scalar-compatible local weights, matching note section 6."""
    return _local_geometry_fit(
        X,
        scalar,
        use_potential_weights=True,
        **kwargs,
    )


def fit_tangent_constrained_scalar(X, scalar, **kwargs):
    """Fit using scalar-regression constrained local tangent estimates."""
    return _local_geometry_fit(
        X,
        scalar,
        use_potential_weights=False,
        use_tangent_constraint=True,
        **kwargs,
    )


def fit_self_consistent_gradient_manfit(
    X,
    scalar,
    *,
    outer_iterations=4,
    gradient_n_neighbors=42,
    gradient_ridge=5e-2,
    confidence_power=1.0,
    k=15,
    inner_T=2,
    eta_g=0.35,
    theta=0.2,
    kappa=2.0,
    adaptive_variance_threshold=0.85,
    adaptive_d_min=2,
    random_state=0,
):
    """Alternating scalar-gradient estimation and confidence-aware MANFIT.

    The gradient is re-estimated on the current fitted geometry instead of
    being estimated once on the noisy observations. A local regression
    confidence controls how strongly each estimated gradient affects the
    velocity-aware neighbor graph and directional weights.
    """
    Z = np.asarray(X, dtype=float).copy()
    scalar = np.asarray(scalar, dtype=float)
    history = []
    gradient = np.zeros_like(Z)
    confidence = np.ones(Z.shape[0], dtype=float)

    for outer in range(int(outer_iterations)):
        gradient, confidence = estimate_gradient_confidence_from_neighbors(
            Z,
            scalar,
            n_neighbors=gradient_n_neighbors,
            ridge=gradient_ridge,
        )
        confidence_for_fit = np.clip(confidence, 0.0, 1.0) ** float(confidence_power)

        fitter = VelocityManifoldFitter(
            Z,
            gradient,
            d_mode="adaptive",
            adaptive_variance_threshold=adaptive_variance_threshold,
            adaptive_d_min=adaptive_d_min,
            k=k,
            T=inner_T,
            eta_g=eta_g,
            theta=theta,
            kappa=kappa,
            bandwidth_mode="variable",
            use_PCA=False,
            velocity_confidence=confidence_for_fit,
            random_state=random_state + outer,
        )
        result = fitter.fit(update_mode="normal_only", return_dict=True)
        Z = result["X"]
        gradient = result["V"]
        history.append(
            {
                "outer_iteration": outer,
                "mean_confidence": float(np.mean(confidence)),
                "median_confidence": float(np.median(confidence)),
                "mean_inner_step_norm": float(
                    np.mean([h["mean_step_norm"] for h in result["history"]])
                ),
            }
        )

    gradient, confidence = estimate_gradient_confidence_from_neighbors(
        Z,
        scalar,
        n_neighbors=gradient_n_neighbors,
        ridge=gradient_ridge,
    )
    final_projector = VelocityManifoldFitter(
        Z,
        gradient,
        d_mode="adaptive",
        adaptive_variance_threshold=adaptive_variance_threshold,
        adaptive_d_min=adaptive_d_min,
        k=k,
        T=1,
        eta_g=0.0,
        theta=theta,
        kappa=kappa,
        bandwidth_mode="variable",
        use_PCA=False,
        velocity_confidence=np.clip(confidence, 0.0, 1.0) ** float(confidence_power),
        random_state=random_state + int(outer_iterations),
    )
    projected = final_projector.fit(update_mode="normal_only", return_dict=True)
    return {
        "position": Z,
        "gradient": projected["V"],
        "raw_gradient": gradient,
        "confidence": confidence,
        "history": history,
    }


def s_curve_projection_distance(X, n_grid=4096):
    """Distance from points to the noiseless S-curve surface."""
    X = np.asarray(X, dtype=float)
    t_grid = np.linspace(-1.5 * np.pi, 1.5 * np.pi, int(n_grid))
    curve_x = np.sin(t_grid)
    curve_z = np.sign(t_grid) * (np.cos(t_grid) - 1.0)
    dx2 = (X[:, [0]] - curve_x[None, :]) ** 2
    dz2 = (X[:, [2]] - curve_z[None, :]) ** 2
    return np.sqrt(np.min(dx2 + dz2, axis=1))


def gradient_cosine_error(estimated, truth, eps=1e-12):
    estimated = normalize_rows(estimated, eps=eps)
    truth = normalize_rows(truth, eps=eps)
    cosine = np.sum(estimated * truth, axis=1)
    return float(np.mean(1.0 - np.clip(cosine, -1.0, 1.0)))


def local_tangent_projectors(X, n_neighbors=32, tangent_dim=2):
    """Estimate local PCA tangent projectors from point positions."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    n_neighbors = min(int(n_neighbors), X.shape[0] - 1)
    tangent_dim = min(int(tangent_dim), X.shape[1])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    _, indices = nbrs.kneighbors(X)
    projectors = np.zeros((X.shape[0], X.shape[1], X.shape[1]), dtype=float)

    for i, neigh in enumerate(indices[:, 1:]):
        local = X[neigh]
        centered = local - np.mean(local, axis=0)
        cov = centered.T @ centered / max(centered.shape[0], 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        basis = eigvecs[:, order[:tangent_dim]]
        projectors[i] = basis @ basis.T

    return projectors


def project_vectors_to_local_tangent(vectors, X, n_neighbors=32, tangent_dim=2):
    """Project vectors onto local PCA tangent spaces estimated from X."""
    vectors = np.asarray(vectors, dtype=float)
    projectors = local_tangent_projectors(
        X,
        n_neighbors=n_neighbors,
        tangent_dim=tangent_dim,
    )
    return np.einsum("nij,nj->ni", projectors, vectors)


def summarize_fit_errors(label, position, gradient, truth_position, truth_gradient):
    """Return position, manifold-distance, and gradient-direction errors."""
    position = np.asarray(position, dtype=float)
    truth_position = np.asarray(truth_position, dtype=float)
    projected_gradient = project_vectors_to_local_tangent(
        gradient,
        position,
        n_neighbors=32,
        tangent_dim=2,
    )
    return {
        "method": label,
        "indexed_position_rmse": float(
            np.sqrt(np.mean(np.sum((position - truth_position) ** 2, axis=1)))
        ),
        "s_curve_distance_rmse": float(
            np.sqrt(np.mean(s_curve_projection_distance(position) ** 2))
        ),
        "gradient_cosine_error": gradient_cosine_error(gradient, truth_gradient),
        "projected_gradient_cosine_error": gradient_cosine_error(
            projected_gradient,
            truth_gradient,
        ),
    }


def format_metric_table(metrics):
    columns = [
        "method",
        "indexed_position_rmse",
        "s_curve_distance_rmse",
        "gradient_cosine_error",
        "projected_gradient_cosine_error",
    ]
    widths = {
        col: max(len(col), *(len(f"{row[col]:.4f}") if col != "method" else len(row[col]) for row in metrics))
        for col in columns
    }
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    lines = [header, "  ".join("-" * widths[col] for col in columns)]
    for row in metrics:
        lines.append(
            "  ".join(
                row[col].ljust(widths[col])
                if col == "method"
                else f"{row[col]:.4f}".rjust(widths[col])
                for col in columns
            )
        )
    return "\n".join(lines)
