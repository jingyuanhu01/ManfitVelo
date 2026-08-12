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

    Also returns `relative_error` (added 2026-08-12): the local regression's
    own unclipped `ss_res / ss_tot` ratio, i.e. `1 - r2` before it gets
    clipped to [0, 1] and folded into `confidence`. This is the raw,
    dimensionless fitting-error signal `VelocityManifoldFitter`'s
    `lambda_v_confidence_scaling="inverse_error"`/`"rank"` modes use
    directly (`lambda_v_effective = lambda_v / (1 + relative_error)`, or
    `lambda_v * (1 - percentile_rank(relative_error))`) -- both genuine
    decreasing functions of the estimation error itself, with no extra
    shape hyperparameter to separately tune, unlike the `confidence**power`
    family.
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
    relative_error = np.zeros(X.shape[0], dtype=float)
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
        relative_error[i] = ss_res / ss_tot
        r2 = np.clip(1.0 - relative_error[i], 0.0, 1.0)

        singular_values = np.linalg.svd(dX, compute_uv=False)
        cond = singular_values[0] / (singular_values[-1] + 1e-12)
        condition_score = 1.0 / (1.0 + cond / float(condition_scale))
        confidence[i] = r2 * condition_score

    if np.max(confidence) > 0:
        confidence = confidence / np.max(confidence)
    return gradients, confidence, relative_error


def fit_scalar_gradient_manfit(
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
    lambda_v=0.0,
    lambda_v_confidence_scaling="none",
    lambda_v_confidence_power=1.0,
    velocity_covariance_mode="centered",
    velocity_trace_normalization="match_position_trace",
    adaptive_variance_threshold=0.85,
    adaptive_d_min=2,
    random_state=0,
    oracle_gradient=None,
):
    """Alternating scalar-gradient estimation and confidence-aware MANFIT.

    The gradient is re-estimated on the current fitted geometry instead of
    being estimated once on the noisy observations. A local regression
    confidence controls how strongly each estimated gradient affects the
    velocity-aware neighbor graph and directional weights.

    This is the scalar-field analog of M6 (VelocityManifoldFitter): the
    estimated gradient is fed in as if it were a velocity, so it drives both
    the velocity-aware neighbor reranking and (via `lambda_v`) the tangent
    covariance blend, exactly as for real velocity data -- unlike
    `fit_potential_aware_neighborhoods` (removed 2026-08-12, see
    simulation/current_plan.md P4.0), which only used the scalar field as a
    multiplicative neighbor reweighting and never touched either mechanism.

    `lambda_v` defaults to `0.0` (matching `VelocityManifoldFitter`'s own
    class default, i.e. no covariance blend) rather than being silently
    fixed -- callers who want the frozen-protocol behavior must pass it
    explicitly (P0.1 froze `lambda_v=1.0`, `velocity_covariance_mode=
    "uncentered"` for the vector-field M6 pipeline; whether the same values
    are appropriate for scalar gradients is not yet validated -- P4.1 is the
    ablation designed to check this, see current_plan.md).

    `lambda_v_confidence_scaling`/`lambda_v_confidence_power` (added
    2026-08-12, same day as the oracle_gradient ablation above) let
    `lambda_v` itself be discounted per-point instead of applying a single
    global compromise value to every point regardless of how trustworthy its
    own gradient estimate is -- see `VelocityManifoldFitter`'s own docstring
    for the mechanism and motivation (P4.1 found a fixed `lambda_v=1.0`
    helps when the gradient is exact but actively hurts the realistic
    estimated-gradient case; this is the fix rather than picking one flat
    compromise value). Three families, kept side by side rather than one
    superseding another -- each has a genuine trade-off:
    - `"linear"`/`"power"` reuse this function's own `confidence` (already
      computed for `velocity_confidence`) raised to `lambda_v_confidence_
      power` -- this power is a *separate* free shape parameter, not itself
      derived from anything. Empirically the strongest of the three at
      power=16 on `scalar_saddle` (see current_plan.md P4.1 follow-up), but that
      exponent was only ever the smallest value in an exploratory grid
      evaluated *on final seeds* -- it has never gone through a real
      tuning-seed selection and must not be read as recommended or frozen.
    - `"inverse_error"` (added 2026-08-12) uses `lambda_v_effective =
      lambda_v / (1 + relative_error)`, where `relative_error` is this
      function's own `estimate_gradient_confidence_from_neighbors` local
      regression's raw, unclipped `ss_res/ss_tot` -- a genuine decreasing
      function of the estimation error itself, with no extra shape
      hyperparameter to separately select. Empirically much weaker than
      `"power"` on `scalar_saddle`, because its steepness is pinned to the
      raw numeric scale of `relative_error`, which happened to be small.
    - `"rank"` (added 2026-08-12, same day, a direct response to the
      `"power"`-exponent tuning-seed gap above) uses `lambda_v_effective =
      lambda_v * (1 - percentile_rank)`, `percentile_rank` being this
      point's ordinal rank of `relative_error` within the current batch.
      Also zero free parameters, but -- unlike `"inverse_error"` --
      invariant to the absolute numeric scale of `relative_error`, so it
      keeps discounting meaningfully even when errors are small in
      absolute terms.
    Note `lambda_v_confidence_power` is independent of `confidence_power`
    above -- that one shapes `velocity_confidence` itself (which also drives
    neighbor reranking/directional weighting via `estimate_gradient_
    confidence_from_neighbors`'s `confidence` output), this one (when
    `lambda_v_confidence_scaling="power"`) additionally shapes only how much
    of `lambda_v` a given point's confidence earns; `"inverse_error"`/
    `"rank"` don't use either confidence or this power at all.

    `k` is NOT computed internally (matching `fit_vmf_variant`'s own
    design -- the orchestration layer is responsible for that, not the
    fitting routine itself). Frozen-protocol callers should pass
    `simulation.benchmark_core.neighbor_count(n, d)` (the same k(n,d) rule
    used everywhere else in this pipeline) rather than relying on the
    k=15 default here, which is only a standalone-usability fallback.

    `oracle_gradient` (current_plan.md P4.1's ablation): if given (an array
    shaped like `X`), it replaces the estimated gradient at every outer
    iteration and at the final projection step, with confidence fixed at 1.0
    everywhere -- `estimate_gradient_confidence_from_neighbors` is never
    called. Every other mechanic (outer/inner iteration counts, k, lambda_v,
    ...) is unchanged, so comparing an `oracle_gradient` run against a
    normal (estimated) run on the same data isolates joint geometric-fitting
    error from local-regression (gradient estimation) error: whatever error
    remains under `oracle_gradient` is attributable to the fitting stage
    alone, since the gradient input is exact by construction.
    """
    Z = np.asarray(X, dtype=float).copy()
    scalar = np.asarray(scalar, dtype=float)
    use_oracle = oracle_gradient is not None
    if use_oracle:
        oracle_gradient = np.asarray(oracle_gradient, dtype=float)
        if oracle_gradient.shape != Z.shape:
            raise ValueError("oracle_gradient must have the same shape as X")
    history = []
    gradient = np.zeros_like(Z)
    confidence = np.ones(Z.shape[0], dtype=float)

    def make_fitter(position, gradient_field, confidence_field, relative_error_field, T, eta, seed):
        return VelocityManifoldFitter(
            position,
            gradient_field,
            d_mode="adaptive",
            adaptive_variance_threshold=adaptive_variance_threshold,
            adaptive_d_min=adaptive_d_min,
            k=k,
            T=T,
            eta_g=eta,
            theta=theta,
            kappa=kappa,
            bandwidth_mode="variable",
            use_PCA=False,
            velocity_confidence=confidence_field,
            lambda_v=lambda_v,
            lambda_v_confidence_scaling=lambda_v_confidence_scaling,
            lambda_v_confidence_power=lambda_v_confidence_power,
            lambda_v_relative_error=relative_error_field,
            velocity_covariance_mode=velocity_covariance_mode,
            velocity_trace_normalization=velocity_trace_normalization,
            random_state=seed,
        )

    def current_gradient_confidence(position):
        if use_oracle:
            n = position.shape[0]
            return oracle_gradient, np.ones(n, dtype=float), np.zeros(n, dtype=float)
        return estimate_gradient_confidence_from_neighbors(
            position, scalar, n_neighbors=gradient_n_neighbors, ridge=gradient_ridge,
        )

    for outer in range(int(outer_iterations)):
        gradient, confidence, relative_error = current_gradient_confidence(Z)
        confidence_for_fit = np.clip(confidence, 0.0, 1.0) ** float(confidence_power)

        fitter = make_fitter(Z, gradient, confidence_for_fit, relative_error, inner_T, eta_g, random_state + outer)
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

    gradient, confidence, relative_error = current_gradient_confidence(Z)
    final_projector = make_fitter(
        Z, gradient, np.clip(confidence, 0.0, 1.0) ** float(confidence_power), relative_error,
        1, 0.0, random_state + int(outer_iterations),
    )
    projected = final_projector.fit(update_mode="normal_only", return_dict=True)
    return {
        "position": Z,
        "gradient": projected["V"],
        "raw_gradient": gradient,
        "confidence": confidence,
        "history": history,
    }
