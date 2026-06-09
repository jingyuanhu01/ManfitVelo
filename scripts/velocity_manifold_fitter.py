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

        self.X = self.Y.copy()
        self.U = None
        self.P = None
        self.local_dims = None
        self.v = None
        self.neighbors = None
        self.weights = None
        self.bandwidths = None
        self.history = []

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

    def _compute_local_tangent(self):
        fixed_d = None if self.d_mode == "adaptive" else self.global_d
        U_all = [] if fixed_d is None else np.zeros((self.n, self.D, fixed_d), dtype=float)
        P_all = np.zeros((self.n, self.D, self.D), dtype=float)
        local_dims = np.zeros(self.n, dtype=int)

        for i in range(self.n):
            neigh = self.neighbors[i]
            w = self.weights[i]

            x_bar = np.sum(w[:, None] * self.X[neigh], axis=0)
            diff = self.X[neigh] - x_bar
            C = (w[:, None] * diff).T @ diff
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

        self.U = U_all
        self.P = P_all
        self.local_dims = local_dims

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
            self._compute_local_tangent()
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
        self._compute_local_tangent()
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
            }
        return self.X
