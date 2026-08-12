"""Velocity-aware manifold fitting.

This module contains the merged manifold fitting implementation used by the
simulation and RNA-velocity notebooks. The public API is intentionally small:

* :func:`select_adaptive_local_pca_dimension` chooses a local PCA dimension
  from a covariance spectrum.
* :func:`reduce_global_dimension` optionally reduces high-dimensional state and
  velocity matrices before fitting.
* :class:`VelocityManifoldFitter` fits denoised cell states and projected
  velocities with either adaptive or globally fixed local tangent dimension.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def select_adaptive_local_pca_dimension(
    eigvals,
    variance_threshold=0.8,
    d_min=2,
    d_max=None,
    eps=1e-12,
):
    """Choose a local PCA dimension from eigenvalues.

    The selected dimension is the smallest value whose cumulative explained
    variance reaches ``variance_threshold``, clipped to ``[d_min, d_max]``.
    If the local covariance is numerically zero, the function returns
    ``d_min``.

    :param eigvals: Eigenvalues from a local covariance matrix.
    :type eigvals: array-like
    :param float variance_threshold: Cumulative variance target in ``(0, 1]``.
    :param int d_min: Minimum allowed local dimension. Defaults to ``2``.
    :param d_max: Maximum allowed local dimension. If ``None``, use all
        available eigenvalues.
    :type d_max: int or None
    :param float eps: Numerical tolerance for zero total variance.
    :returns: Selected local PCA dimension.
    :rtype: int
    """
    eigvals = np.asarray(eigvals, dtype=float)
    eigvals = np.maximum(eigvals, 0.0)
    eigvals = np.sort(eigvals)[::-1]

    if d_max is None:
        d_max = eigvals.size
    d_max = min(int(d_max), eigvals.size)
    d_min = min(max(int(d_min), 1), d_max)

    total = float(np.sum(eigvals))
    if total <= eps:
        return d_min

    cumulative = np.cumsum(eigvals) / total
    d = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    return min(max(d, d_min), d_max)


def reduce_global_dimension(X, V, n_components=30, random_state=0):
    """Reduce state and velocity matrices with a shared PCA basis.

    PCA is fit on ``X``. The state matrix is transformed with
    ``pca.fit_transform(X)``, and the velocity matrix is projected into the
    same basis by ``V @ pca.components_.T``. If the input dimension is already
    less than or equal to ``n_components``, the original arrays are returned
    with ``pca=None``.

    :param X: State matrix with shape ``(n_cells, n_features)``.
    :type X: array-like
    :param V: Velocity matrix with the same shape as ``X``.
    :type V: array-like
    :param int n_components: Target global PCA dimension. Defaults to ``30``.
    :param random_state: Random seed forwarded to scikit-learn PCA.
    :returns: ``(X_reduced, V_reduced, pca)``.
    :rtype: tuple
    :raises ValueError: If ``X`` and ``V`` have different shapes.
    """
    X = np.asarray(X, dtype=float)
    V = np.asarray(V, dtype=float)
    if X.shape != V.shape:
        raise ValueError("X and V must match shape")
    n_components = min(int(n_components), X.shape[0], X.shape[1])
    if X.shape[1] <= n_components:
        return X, V, None

    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X)
    V_reduced = V @ pca.components_.T
    return X_reduced, V_reduced, pca


class VelocityManifoldFitter:
    """Velocity-aware manifold fitter.

    The fitter takes a state matrix ``Y`` and matching velocity matrix ``W``.
    It builds a velocity-aware neighbor graph, estimates local tangent spaces
    with weighted PCA, projects velocities onto those tangent spaces, and
    updates positions by a normal-only default rule.

    Parameters are grouped by how often they should be touched:

    **High-priority tuning parameters**
        ``d_mode``, ``adaptive_variance_threshold``, ``adaptive_d_min``,
        ``k``, ``T``, ``eta_g``, and ``theta``. These are the main knobs to
        tune across datasets.

    **Default modes**
        ``update_mode`` is passed to :meth:`fit` and defaults to
        ``"normal_only"``. ``bandwidth_mode`` defaults to ``"variable"``.
        These should usually stay fixed unless a diagnostic sweep suggests
        otherwise.

    **Low-priority parameters**
        ``global_d``, ``use_PCA``, ``PCA_dim``, ``gamma``, ``beta``,
        ``kappa``, ``cv``, ``max_step_frac``, ``h``, cosine conventions, and
        neighbor recomputation controls. These are useful for diagnostics but
        are not expected to need frequent tuning.

    :param Y: State matrix with shape ``(n_cells, n_features)``.
    :type Y: array-like
    :param W: Velocity matrix with the same shape as ``Y``.
    :type W: array-like
    :param str d_mode: Local PCA dimension mode. ``"adaptive"`` selects a
        local dimension per point; ``"global"`` uses ``global_d`` everywhere.
        Defaults to ``"adaptive"``.
    :param float adaptive_variance_threshold: Explained-variance threshold for
        adaptive local dimensions. Defaults to ``0.8``.
    :param int adaptive_d_min: Minimum adaptive local dimension. Defaults to
        ``2``.
    :param adaptive_d_max: Optional maximum adaptive local dimension.
    :type adaptive_d_max: int or None
    :param int k: Number of neighbors used for local fitting. Defaults to
        ``25``.
    :param int T: Number of fitting iterations. Defaults to ``5``.
    :param float eta_g: Normal correction step size. Smaller values are often
        more stable. Defaults to ``0.45``.
    :param float theta: Strength of velocity-aware neighbor scoring. Smaller
        values are usually more stable. Defaults to ``0.1``.
    :param float lambda_v: Strength of the trace-normalized local velocity
        covariance in tangent estimation. ``0`` recovers the position-only
        covariance exactly. Defaults to ``0``.
    :param str lambda_v_confidence_scaling: How ``lambda_v`` is discounted per
        point before entering the covariance blend, instead of applying the
        same global ``lambda_v`` to every point regardless of how reliable
        its velocity/gradient is. ``"none"`` (default) is bit-identical to
        the pre-2026-08-12 behavior, since this option did not exist before.
        Motivation (see ``simulation/current_plan.md`` P4.1): a fixed
        vector-field-tuned ``lambda_v=1.0`` was found to help the
        oracle-gradient scalar pipeline but hurt the realistic
        estimated-gradient one -- one flat value forces a compromise between
        "helps when trustworthy" and "hurts when not." Three families,
        listed roughly in the order they were tried, each kept because each
        has a genuine trade-off rather than one strictly superseding another:

        - ``"linear"``/``"power"`` use ``velocity_confidence`` (already
          computed for neighbor reranking/directional weighting) --
          ``lambda_v * velocity_confidence[i]``, or ``lambda_v *
          velocity_confidence[i] ** lambda_v_confidence_power``.
          ``lambda_v_confidence_power`` is a *separate* free shape parameter
          here, not itself derived from anything. Empirically the strongest
          of the three families at power=16 on a scalar-gradient scenario
          (see ``simulation/current_plan.md`` P4.1 follow-up), but that specific
          exponent was only ever picked as the smallest value in an
          exploratory grid evaluated *on final seeds* -- it has never gone
          through the proper tuning-seed selection procedure and must not
          be read as a recommended or frozen setting.
        - ``"inverse_error"`` (added 2026-08-12) uses ``lambda_v_effective =
          lambda_v / (1 + lambda_v_relative_error[i])`` -- a genuine
          decreasing function of an actual per-point estimation error, with
          no extra shape hyperparameter to separately select. Empirically
          much weaker than ``"power"`` on the one scenario tested, because
          its steepness is pinned to the raw numeric scale of
          ``lambda_v_relative_error`` (which happened to be small, so ``1 /
          (1 + x)`` stayed close to 1 for most points) -- a real cost of
          having no free parameter to compensate for that.
        - ``"rank"`` (added 2026-08-12, same day, direct response to the
          ``"power"`` tuning-seed gap above) uses ``lambda_v_effective =
          lambda_v * (1 - percentile_rank[i])``, where
          ``percentile_rank[i]`` is point ``i``'s ordinal rank of
          ``lambda_v_relative_error`` within the current batch, in
          ``[0, 1]``. Also zero free parameters, but unlike
          ``"inverse_error"`` it is invariant to the absolute numeric scale
          of the error (only relative ordering within the batch matters),
          so it discounts more aggressively when errors happen to be small
          in absolute terms.

        Both ``"inverse_error"`` and ``"rank"`` require
        ``lambda_v_relative_error`` to be supplied by the caller (e.g.
        ``scripts.scalar_potential_manfit.fit_scalar_gradient_manfit``
        passes its own local-regression ``ss_res/ss_tot``); left at its
        default (all zeros) otherwise, which means no discount at all for
        ``"inverse_error"`` (``1/(1+0) = 1``), but a uniform half discount
        for ``"rank"`` (there is nothing to rank when every point looks
        identically (un)confident, so an arbitrary tie-breaking order is
        avoided in favor of applying the same 0.5 factor everywhere).
    :param float lambda_v_confidence_power: Exponent used when
        ``lambda_v_confidence_scaling="power"``. Values above 1 discount
        low-confidence points more aggressively; values below 1 discount
        them more gently. Unused otherwise. Defaults to ``1.0``.
    :param lambda_v_relative_error: Optional per-cell dimensionless fitting
        error (e.g. a local regression's ``ss_res/ss_tot``), used when
        ``lambda_v_confidence_scaling`` is ``"inverse_error"`` or ``"rank"``.
        Unbounded above (no clipping to [0, 1], unlike ``velocity_confidence``)
        -- both consuming modes are well-defined for any nonnegative value.
        Defaults to all zeros (no discount) when not given.
    :param str velocity_covariance_mode: Local velocity construction used for
        tangent estimation: ``"centered"``, ``"uncentered"``, or
        ``"covariance_plus_mean"``. The latter two are algebraically
        equivalent under common weights and are retained as explicit audit
        labels. Defaults to ``"centered"``.
    :param str velocity_trace_normalization: Normalization applied before
        combining covariances. The supported rule, ``"match_position_trace"``,
        scales the velocity covariance to the trace of the position covariance.
    :param float velocity_tangent_weight: An independent, additive covariance
        term contributed upstream (Jingyuan Hu, commit "Add
        velocity-augmented tangent fitting") and reconciled here alongside
        ``lambda_v`` rather than merged into it, since the two are
        numerically different constructions built on the same idea -- see
        :meth:`_velocity_tangent_term` for the exact formula and how it
        differs from ``lambda_v``'s own mechanism (unit-normalized velocity
        *directions* rather than raw vectors, an extra per-neighbor
        ``velocity_confidence`` discount, and trace(C_position)-scaled
        rather than trace-matched). Purely additive on top of whatever
        ``lambda_v`` already contributes; keyword-only. Defaults to ``0.0``
        (no effect, matching its upstream default), so leaving it unset
        reproduces pre-merge behavior exactly.
    :param bool record_tangent_diagnostics: Save pointwise covariance spectra
        and matrices at every tangent update. Intended for synthetic mechanism
        diagnostics and disabled by default.
    :param bool return_tangent_diagnostics: Retain the final pointwise
        covariance diagnostic snapshot without retaining the full history.
        Disabled by default to avoid quadratic-in-ambient-dimension storage in
        ordinary application runs.
    :param str bandwidth_mode: Kernel bandwidth mode, either ``"variable"`` or
        ``"fixed"``. Defaults to ``"variable"``.
    :param bool recompute_neighbors: Whether to rebuild neighbors during
        fitting. Defaults to ``False``.
    :param float gamma: Sharpness of velocity-aware neighbor scoring.
    :param float beta: Radial kernel exponent.
    :param float kappa: Directional kernel strength.
    :param float cv: Tangential velocity transport strength for
        ``update_mode="original"``. Defaults to ``0.0``.
    :param float max_step_frac: Per-iteration step cap as a fraction of local
        bandwidth.
    :param int global_d: Fixed local PCA dimension used when
        ``d_mode="global"``. This is a low-priority diagnostic parameter.
    :param bool use_PCA: Whether to globally reduce ``Y`` and ``W`` before
        fitting. Defaults to ``True``.
    :param int PCA_dim: Global PCA dimension used when ``use_PCA=True``.
        Defaults to ``30``.
    :param float h: Fixed bandwidth used when ``bandwidth_mode="fixed"``.
    :param bool use_abs_cos: Use absolute cosine in velocity-aware distance.
    :param bool weight_use_abs_cos: Use absolute cosine in directional kernel
        weights.
    :param velocity_confidence: Optional per-cell confidence in ``W``. Values
        near zero reduce velocity-aware scoring and directional weighting.
    :param random_state: Random seed for NumPy and PCA.
    :param int candidate_mult: Multiplier for Euclidean candidate neighbors
        before velocity-aware reranking.
    :param int neighbor_update_freq: Neighbor recomputation frequency when
        ``recompute_neighbors=True``.
    :param float eps: Numerical tolerance.
    :raises ValueError: If inputs or mode parameters are invalid.
    """

    def __init__(
        self,
        Y,
        W,
        # Important tuning parameters
        d_mode="adaptive",
        adaptive_variance_threshold=0.8,
        adaptive_d_min=2,
        adaptive_d_max=None,
        k=25,
        T=5,
        eta_g=0.45,
        theta=0.1,
        lambda_v=0.0,
        lambda_v_confidence_scaling="none",
        lambda_v_confidence_power=1.0,
        lambda_v_relative_error=None,
        velocity_covariance_mode="centered",
        velocity_trace_normalization="match_position_trace",
        record_tangent_diagnostics=False,
        return_tangent_diagnostics=False,
        # Defaults that should usually stay fixed
        bandwidth_mode="variable",
        recompute_neighbors=False,
        # Lower-priority parameters
        gamma=2.0,
        beta=1.0,
        kappa=1.0,
        cv=0.0,
        max_step_frac=0.2,
        global_d=2,
        use_PCA=True,
        PCA_dim=30,
        h=0.8,
        use_abs_cos=False,
        weight_use_abs_cos=True,
        velocity_confidence=None,
        random_state=0,
        candidate_mult=4,
        neighbor_update_freq=1,
        eps=1e-12,
        *,
        velocity_tangent_weight=0.0,
    ):
        if random_state is not None:
            np.random.seed(random_state)

        self.Y_original = np.asarray(Y, dtype=float)
        self.W_original = np.asarray(W, dtype=float)

        if self.Y_original.shape != self.W_original.shape:
            raise ValueError("Y and W must match shape")

        self.use_PCA = bool(use_PCA)
        self.PCA_dim = int(PCA_dim)
        if self.PCA_dim < 1:
            raise ValueError("PCA_dim must be at least 1")
        if self.use_PCA:
            self.Y, self.W, self.global_pca = reduce_global_dimension(
                self.Y_original,
                self.W_original,
                n_components=self.PCA_dim,
                random_state=random_state,
            )
        else:
            self.Y = self.Y_original.copy()
            self.W = self.W_original.copy()
            self.global_pca = None

        self.n, self.D = self.Y.shape
        if self.n <= 1:
            raise ValueError("At least two points are required")

        self.k = min(int(k), self.n - 1)
        self.d_mode = d_mode
        self.adaptive_variance_threshold = float(adaptive_variance_threshold)
        self.adaptive_d_min = int(adaptive_d_min)
        self.adaptive_d_max = None if adaptive_d_max is None else int(adaptive_d_max)
        self.global_d = int(global_d)
        self.theta = float(theta)
        self.lambda_v = float(lambda_v)
        self.lambda_v_confidence_scaling = str(lambda_v_confidence_scaling)
        self.lambda_v_confidence_power = float(lambda_v_confidence_power)
        self.velocity_covariance_mode = str(velocity_covariance_mode)
        self.velocity_trace_normalization = str(velocity_trace_normalization)
        self.velocity_tangent_weight = float(velocity_tangent_weight)
        self.record_tangent_diagnostics = bool(record_tangent_diagnostics)
        self.return_tangent_diagnostics = bool(return_tangent_diagnostics)
        self.gamma = float(gamma)
        self.use_abs_cos = bool(use_abs_cos)
        self.weight_use_abs_cos = bool(weight_use_abs_cos)
        if velocity_confidence is None:
            self.velocity_confidence = np.ones(self.n, dtype=float)
        else:
            self.velocity_confidence = np.asarray(velocity_confidence, dtype=float)
            if self.velocity_confidence.shape != (self.n,):
                raise ValueError("velocity_confidence must have shape (n_cells,)")
            self.velocity_confidence = np.clip(self.velocity_confidence, 0.0, 1.0)
        if lambda_v_relative_error is None:
            self.lambda_v_relative_error = np.zeros(self.n, dtype=float)
        else:
            self.lambda_v_relative_error = np.asarray(lambda_v_relative_error, dtype=float)
            if self.lambda_v_relative_error.shape != (self.n,):
                raise ValueError("lambda_v_relative_error must have shape (n_cells,)")
        self.kappa = float(kappa)
        self.h = float(h)
        self.bandwidth_mode = bandwidth_mode
        self.beta = float(beta)
        self.eta_g = float(eta_g)
        self.cv = float(cv)
        self.max_step_frac = float(max_step_frac)
        self.T = int(T)
        self.recompute_neighbors = bool(recompute_neighbors)
        self.candidate_mult = int(candidate_mult)
        self.neighbor_update_freq = int(neighbor_update_freq)
        self.eps = float(eps)

        if self.d_mode == "adaptive":
            if not 0.0 < self.adaptive_variance_threshold <= 1.0:
                raise ValueError("adaptive_variance_threshold must be in (0, 1]")
            if self.adaptive_d_min < 1 or self.adaptive_d_min > self.D:
                raise ValueError("adaptive_d_min must be between 1 and the ambient dimension")
            if self.adaptive_d_max is not None and (
                self.adaptive_d_max < self.adaptive_d_min or self.adaptive_d_max > self.D
            ):
                raise ValueError("adaptive_d_max must be between adaptive_d_min and the ambient dimension")
        elif self.d_mode == "global":
            if self.global_d < 1 or self.global_d > self.D:
                raise ValueError("global_d must be between 1 and the ambient dimension")
        else:
            raise ValueError("d_mode must be 'adaptive' or 'global'")
        if self.bandwidth_mode not in {"variable", "fixed"}:
            raise ValueError("bandwidth_mode must be 'variable' or 'fixed'")
        if self.lambda_v < 0.0:
            raise ValueError("lambda_v must be nonnegative")
        if self.lambda_v_confidence_scaling not in {"none", "linear", "power", "inverse_error", "rank"}:
            raise ValueError(
                "lambda_v_confidence_scaling must be 'none', 'linear', 'power', "
                "'inverse_error', or 'rank'"
            )
        if self.velocity_covariance_mode not in {
            "centered", "uncentered", "covariance_plus_mean"
        }:
            raise ValueError(
                "velocity_covariance_mode must be 'centered', 'uncentered', "
                "or 'covariance_plus_mean'"
            )
        if self.velocity_trace_normalization != "match_position_trace":
            raise ValueError("velocity_trace_normalization must be 'match_position_trace'")
        if not np.isfinite(self.velocity_tangent_weight) or self.velocity_tangent_weight < 0.0:
            raise ValueError("velocity_tangent_weight must be a finite nonnegative number")

        self.X = self.Y.copy()
        self.U = None
        self.P = None
        self.local_dims = None
        self.v = None
        self.neighbors = None
        self.weights = None
        self.bandwidths = None
        self.history = []
        self.last_tangent_diagnostics = None
        self.tangent_diagnostics_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _cosine_rows(self, a, b, use_abs=False):
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + self.eps
        c = np.sum(a * b, axis=1) / denom
        return np.abs(c) if use_abs else c

    def _velocity_aware_distance(self, diff, velocity, use_abs_cos=None, confidence=1.0):
        d0 = np.linalg.norm(diff, axis=1)
        if np.linalg.norm(velocity) < self.eps:
            return d0, np.zeros_like(d0)

        if use_abs_cos is None:
            use_abs_cos = self.use_abs_cos

        Wi = np.repeat(velocity[None, :], diff.shape[0], axis=0)
        cos_val = self._cosine_rows(Wi, diff, use_abs=use_abs_cos)
        vel_term = 1.0 - self._sigmoid(self.gamma * float(confidence) * cos_val)
        dist = (1.0 - self.theta) * d0 + self.theta * vel_term
        return dist, cos_val

    def _build_neighbors(self, velocity=None):
        if velocity is None:
            velocity = self.W

        m = min(self.n, max(self.k * self.candidate_mult, self.k + 5) + 1)
        nbrs = NearestNeighbors(n_neighbors=m, metric="euclidean").fit(self.X)
        candidates = nbrs.kneighbors(self.X, return_distance=False)

        neighbors = np.zeros((self.n, self.k), dtype=int)
        for i in range(self.n):
            cand = candidates[i]
            cand = cand[cand != i]

            diff = self.X[cand] - self.X[i]
            score, _ = self._velocity_aware_distance(
                diff,
                velocity[i],
                confidence=self.velocity_confidence[i],
            )
            idx = np.argsort(score)[: self.k]
            neighbors[i] = cand[idx]

        self.neighbors = neighbors

    def _select_velocity(self, velocity_mode, blend_lambda):
        if velocity_mode == "projected":
            return self.v if self.v is not None else self.W
        if velocity_mode == "raw":
            return self.W
        if velocity_mode == "blend":
            vi = self.v if self.v is not None else self.W
            return (1.0 - blend_lambda) * self.W + blend_lambda * vi
        raise ValueError("velocity_mode must be 'projected', 'raw', or 'blend'")

    def _update_weights(self, velocity_mode="projected", blend_lambda=0.0):
        Xj = self.X[self.neighbors]
        xi = self.X[:, None, :]
        diff = Xj - xi
        diff_norm = np.linalg.norm(diff, axis=2)

        Wi = self._select_velocity(velocity_mode, blend_lambda)
        Wi_expanded = Wi[:, None, :]
        Wi_norm = np.linalg.norm(Wi_expanded, axis=2) + self.eps

        cos_val = np.sum(Wi_expanded * diff, axis=2) / (Wi_norm * (diff_norm + self.eps))
        if self.weight_use_abs_cos:
            cos_for_direction = np.abs(cos_val)
        else:
            cos_for_direction = cos_val

        confidence = self.velocity_confidence[:, None]
        cos_for_direction = confidence * cos_for_direction
        cos_for_distance = np.abs(cos_val) if self.use_abs_cos else cos_val
        cos_for_distance = confidence * cos_for_distance
        vel_term = 1.0 - self._sigmoid(self.gamma * cos_for_distance)
        dist = (1.0 - self.theta) * diff_norm + self.theta * vel_term

        if self.bandwidth_mode == "variable":
            h = np.max(diff_norm, axis=1) + self.eps
        elif self.bandwidth_mode == "fixed":
            h = np.full(self.n, self.h + self.eps)
        else:
            raise ValueError("bandwidth_mode must be 'variable' or 'fixed'")

        scaled = dist / h[:, None]
        spatial = np.maximum(0.0, 1.0 - scaled**2) ** self.beta
        directional = np.exp(self.kappa * cos_for_direction)
        w_tilde = spatial * directional + self.eps

        self.weights = w_tilde / w_tilde.sum(axis=1, keepdims=True)
        self.bandwidths = h

    def _velocity_covariance(self, neigh, weights):
        local_velocity = self.W[neigh]
        mean_velocity = np.sum(weights[:, None] * local_velocity, axis=0)
        centered = local_velocity - mean_velocity
        covariance = (weights[:, None] * centered).T @ centered
        mean_flow = np.outer(mean_velocity, mean_velocity)
        if self.velocity_covariance_mode == "centered":
            return covariance, mean_velocity
        if self.velocity_covariance_mode == "uncentered":
            return (weights[:, None] * local_velocity).T @ local_velocity, mean_velocity
        return covariance + mean_flow, mean_velocity

    def _velocity_tangent_term(self, neigh, weights, C_position):
        """Independent, additive covariance term contributed upstream by
        Jingyuan Hu (commit "Add velocity-augmented tangent fitting",
        2026-07-14) as ``velocity_tangent_weight`` -- reconciled here
        alongside ``lambda_v`` (added/extended over the following weeks)
        rather than replacing it, since the two were built independently on
        the same underlying idea (blend a velocity-derived covariance into
        the tangent-covariance estimate) but are NOT numerically the same
        construction:

        - Uses UNIT-NORMALIZED neighbor velocity *directions*, not the raw
          (weighted) velocity vectors ``_velocity_covariance``'s
          ``"uncentered"`` mode uses -- so this term is insensitive to
          velocity-magnitude variation among neighbors, only direction
          spread matters.
        - Additionally discounts each neighbor's own contribution by that
          neighbor's ``velocity_confidence`` (``_velocity_covariance`` has
          no analogous per-neighbor discount).
        - Scales by ``trace(C_position)`` directly rather than trace-matching
          exactly: this ``C_velocity`` already has trace <= 1 by
          construction (confidence-weighted, kernel-weighted unit vectors,
          kernel weights summing to 1), so the resulting term's own trace is
          ``velocity_tangent_weight * trace(C_position) * trace(C_velocity)``,
          not necessarily equal to ``trace(C_position)`` the way ``lambda_v``'s
          own trace-matched term is.

        Purely additive on top of whatever ``lambda_v`` already produced --
        ``velocity_tangent_weight=0.0`` (the default) makes this exactly
        zero, so it never changes lambda_v-only behavior.
        """
        local_velocity = self.W[neigh]
        velocity_norm = np.linalg.norm(local_velocity, axis=1)
        valid_velocity = velocity_norm > self.eps
        velocity_direction = np.zeros_like(local_velocity)
        velocity_direction[valid_velocity] = (
            local_velocity[valid_velocity] / velocity_norm[valid_velocity, None]
        )
        velocity_weight = weights * self.velocity_confidence[neigh] * valid_velocity.astype(float)
        C_velocity = (velocity_weight[:, None] * velocity_direction).T @ velocity_direction
        position_scale = float(np.trace(C_position))
        return self.velocity_tangent_weight * position_scale * C_velocity

    def _effective_lambda_v(self) -> np.ndarray:
        """Per-point lambda_v after confidence scaling. "none" (the default,
        and the only mode that existed before 2026-08-12) returns the same
        global self.lambda_v for every point -- bit-identical to the old
        unconditional scalar use. See lambda_v_confidence_scaling's
        docstring entry for the motivation."""
        if self.lambda_v_confidence_scaling == "none":
            return np.full(self.n, self.lambda_v, dtype=float)
        if self.lambda_v_confidence_scaling == "linear":
            return self.lambda_v * self.velocity_confidence
        if self.lambda_v_confidence_scaling == "power":
            return self.lambda_v * self.velocity_confidence ** self.lambda_v_confidence_power
        if self.lambda_v_confidence_scaling == "inverse_error":
            # A genuine decreasing function of an actual estimation error, no
            # extra shape hyperparameter (see docstring) -- but its steepness
            # is fixed by the raw magnitude of lambda_v_relative_error, which
            # is not itself calibrated to any particular scale.
            return self.lambda_v / (1.0 + self.lambda_v_relative_error)
        # "rank": purely ordinal, zero free parameters -- unlike
        # "inverse_error" it is invariant to the absolute magnitude/units of
        # lambda_v_relative_error (which is what made "inverse_error" weak
        # when errors happen to be small in absolute terms), because only
        # each point's error rank *within this batch* matters. See
        # docstring for the motivation (added same day as a direct response
        # to power=16 having been picked from an exploratory grid evaluated
        # on final seeds rather than a real tuning-seed selection).
        n = self.n
        if n <= 1:
            return np.full(n, self.lambda_v, dtype=float)
        rel_err = self.lambda_v_relative_error
        if np.ptp(rel_err) <= 0.0:
            # No differentiating information to rank on (e.g. the "not
            # supplied" default of all zeros, or -- vanishingly unlikely
            # with real floating-point regression errors -- an exact tie
            # across every point). An arbitrary argsort order would silently
            # spread lambda_v from 1x down to 0x with no real justification;
            # fall back to a uniform half discount instead.
            return np.full(n, self.lambda_v * 0.5, dtype=float)
        order = np.argsort(rel_err)
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n, dtype=float)
        percentile = ranks / (n - 1)
        return self.lambda_v * (1.0 - percentile)

    def _compute_local_tangent(self, diagnostic_iteration=None, diagnostic_phase=None):
        fixed_d = None if self.d_mode == "adaptive" else self.global_d
        collect_diagnostics = self.record_tangent_diagnostics or self.return_tangent_diagnostics
        U_all = [] if fixed_d is None else np.zeros((self.n, self.D, fixed_d), dtype=float)
        P_all = np.zeros((self.n, self.D, self.D), dtype=float)
        local_dims = np.zeros(self.n, dtype=int)
        effective_lambda_v = self._effective_lambda_v()
        skip_velocity_covariance = self.lambda_v_confidence_scaling == "none" and self.lambda_v == 0.0 and not collect_diagnostics
        if collect_diagnostics:
            joint_eigvals_all = np.zeros((self.n, self.D), dtype=float)
            position_eigvals_all = np.zeros((self.n, self.D), dtype=float)
            velocity_raw_all = np.zeros((self.n, self.D, self.D), dtype=float)
            velocity_scaled_all = np.zeros((self.n, self.D, self.D), dtype=float)
            position_covariance_all = np.zeros((self.n, self.D, self.D), dtype=float)
            mean_velocity_all = np.zeros((self.n, self.D), dtype=float)

        for i in range(self.n):
            neigh = self.neighbors[i]
            w = self.weights[i]
            lv = effective_lambda_v[i]

            x_bar = np.sum(w[:, None] * self.X[neigh], axis=0)
            diff = self.X[neigh] - x_bar
            C_position = (w[:, None] * diff).T @ diff
            C_position = 0.5 * (C_position + C_position.T)
            # This branch preserves the historical covariance bit-for-bit for
            # lambda_v=0 under the default "none" scaling (the only mode
            # that existed before 2026-08-12), including its eigendecomposition input.
            if skip_velocity_covariance:
                C = C_position
            else:
                C_velocity_raw, mean_velocity = self._velocity_covariance(neigh, w)
                C_velocity_raw = 0.5 * (C_velocity_raw + C_velocity_raw.T)
                position_trace = float(np.trace(C_position))
                velocity_trace = float(np.trace(C_velocity_raw))
                if position_trace > self.eps and velocity_trace > self.eps:
                    C_velocity = C_velocity_raw * (position_trace / velocity_trace)
                else:
                    C_velocity = np.zeros_like(C_velocity_raw)
                C = C_position if lv == 0.0 else C_position + lv * C_velocity
            if self.velocity_tangent_weight > 0.0:
                C = C + self._velocity_tangent_term(neigh, w, C_position)
            C = 0.5 * (C + C.T)

            eigvals, eigvecs = np.linalg.eigh(C)
            order = np.argsort(eigvals)[::-1]
            if fixed_d is None:
                local_d = select_adaptive_local_pca_dimension(
                    eigvals,
                    variance_threshold=self.adaptive_variance_threshold,
                    d_min=self.adaptive_d_min,
                    d_max=self.adaptive_d_max,
                    eps=self.eps,
                )
            else:
                local_d = fixed_d

            idx = order[:local_d]
            U = eigvecs[:, idx]
            if fixed_d is None:
                U_all.append(U)
            else:
                U_all[i] = U
            P_all[i] = U @ U.T
            local_dims[i] = local_d
            if collect_diagnostics:
                joint_eigvals_all[i] = eigvals[order]
                position_eigvals_all[i] = np.linalg.eigvalsh(C_position)[::-1]
                velocity_raw_all[i] = C_velocity_raw
                velocity_scaled_all[i] = C_velocity
                position_covariance_all[i] = C_position
                mean_velocity_all[i] = mean_velocity

        self.U = U_all
        self.P = P_all
        self.local_dims = local_dims
        self.last_tangent_diagnostics = None if not collect_diagnostics else {
            "iteration": diagnostic_iteration,
            "phase": diagnostic_phase,
            "joint_eigenvalues": joint_eigvals_all,
            "position_eigenvalues": position_eigvals_all,
            "position_covariance": position_covariance_all,
            "velocity_covariance_raw": velocity_raw_all,
            "velocity_covariance_scaled": velocity_scaled_all,
            "mean_velocity": mean_velocity_all,
            "effective_lambda_v": effective_lambda_v.copy(),
            "projectors": P_all.copy(),
            "neighbors": self.neighbors.copy(),
            "weights": self.weights.copy(),
        }
        if self.record_tangent_diagnostics:
            self.tangent_diagnostics_history.append(self.last_tangent_diagnostics)

    def _project_velocity(self, velocity=None):
        if velocity is None:
            velocity = self.W
        self.v = np.einsum("nij,nj->ni", self.P, velocity)

    def _local_mean_shift(self):
        Xj = self.X[self.neighbors]
        x_bar = np.sum(self.weights[:, :, None] * Xj, axis=1)
        return x_bar, x_bar - self.X

    def _cap_steps(self, steps):
        cap = self.max_step_frac * self.bandwidths
        step_norm = np.linalg.norm(steps, axis=1)
        scale = np.minimum(1.0, cap / (step_norm + self.eps))
        return steps * scale[:, None]

    def fit(
        self,
        update_mode="normal_only",
        velocity_mode="projected",
        blend_lambda=0.0,
        return_dict=False,
    ):
        """Run manifold fitting.

        ``update_mode="normal_only"`` is the recommended default. It moves each
        point only in the estimated normal direction and uses the tangent space
        to project velocities. ``update_mode="original"`` retains the older
        mean-shift plus tangential velocity transport update and is mainly
        useful as a comparison mode.

        :param str update_mode: ``"normal_only"`` or ``"original"``. Defaults
            to ``"normal_only"``.
        :param str velocity_mode: Velocity source for weights. One of
            ``"projected"``, ``"raw"``, or ``"blend"``.
        :param float blend_lambda: Blend amount used only when
            ``velocity_mode="blend"``.
        :param bool return_dict: If ``True``, return positions, velocities,
            neighbors, weights, tangent projectors, local dimensions,
            bandwidths, global PCA object, and iteration history.
        :returns: Fitted positions, or a result dictionary when
            ``return_dict=True``.
        :rtype: numpy.ndarray or dict
        :raises ValueError: If ``update_mode`` is invalid.
        """
        if update_mode not in {"original", "normal_only"}:
            raise ValueError("update_mode must be 'original' or 'normal_only'")

        working_velocity = self.W.copy()
        self._build_neighbors(working_velocity)

        for t in range(self.T):
            if self.recompute_neighbors and (t > 0) and (t % self.neighbor_update_freq == 0):
                self._build_neighbors(working_velocity)

            self._update_weights(velocity_mode=velocity_mode, blend_lambda=blend_lambda)
            self._compute_local_tangent(diagnostic_iteration=t, diagnostic_phase="pre_update")
            self._project_velocity(working_velocity)

            _, mean_shift = self._local_mean_shift()
            if update_mode == "original":
                v_norm = np.linalg.norm(self.v, axis=1, keepdims=True) + self.eps
                v_dir = self.v / v_norm
                steps = self.eta_g * mean_shift + self.cv * self.bandwidths[:, None] * v_dir
            else:
                tangent_shift = np.einsum("nij,nj->ni", self.P, mean_shift)
                normal_shift = mean_shift - tangent_shift
                steps = self.eta_g * normal_shift

            steps = self._cap_steps(steps)
            self.X = self.X + steps
            working_velocity = self.v.copy()
            self.history.append(
                {
                    "iteration": t,
                    "mean_step_norm": float(np.mean(np.linalg.norm(steps, axis=1))),
                    "max_step_norm": float(np.max(np.linalg.norm(steps, axis=1))),
                }
            )

        self._update_weights(velocity_mode=velocity_mode, blend_lambda=blend_lambda)
        self._compute_local_tangent(diagnostic_iteration=self.T, diagnostic_phase="final")
        self._project_velocity(self.W)

        if return_dict:
            return {
                "X": self.X,
                "V": self.v,
                "neighbors": self.neighbors,
                "weights": self.weights,
                "U": self.U,
                "P": self.P,
                "local_dims": self.local_dims,
                "bandwidths": self.bandwidths,
                "global_pca": self.global_pca,
                "history": self.history,
                "tangent_diagnostics": self.last_tangent_diagnostics,
                "tangent_diagnostics_history": self.tangent_diagnostics_history,
                "algorithm_settings": {
                    "lambda_v": self.lambda_v,
                    "lambda_v_confidence_scaling": self.lambda_v_confidence_scaling,
                    "lambda_v_confidence_power": self.lambda_v_confidence_power,
                    "lambda_v_relative_error_mean": float(np.mean(self.lambda_v_relative_error)),
                    "velocity_covariance_mode": self.velocity_covariance_mode,
                    "velocity_trace_normalization": self.velocity_trace_normalization,
                    "velocity_tangent_weight": self.velocity_tangent_weight,
                    "d_mode": self.d_mode,
                    "global_d": self.global_d if self.d_mode == "global" else None,
                    "adaptive_variance_threshold": (
                        self.adaptive_variance_threshold if self.d_mode == "adaptive" else None
                    ),
                    "k": self.k,
                    "T": self.T,
                },
            }
        return self.X
