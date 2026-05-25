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
    def __init__(
        self,
        Y,
        W,
        # Important tuning parameters
        d=2,
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
        h=0.8,
        use_abs_cos=False,
        weight_use_abs_cos=True,
        random_state=0,
        candidate_mult=4,
        neighbor_update_freq=1,
        eps=1e-12,
    ):
        if random_state is not None:
            np.random.seed(random_state)

        self.Y = np.asarray(Y, dtype=float)
        self.W = np.asarray(W, dtype=float)

        if self.Y.shape != self.W.shape:
            raise ValueError("Y and W must match shape")

        self.n, self.D = self.Y.shape
        if self.n <= 1:
            raise ValueError("At least two points are required")

        self.k = min(int(k), self.n - 1)
        self.d = d
        self.adaptive_variance_threshold = float(adaptive_variance_threshold)
        self.adaptive_d_min = int(adaptive_d_min)
        self.adaptive_d_max = None if adaptive_d_max is None else int(adaptive_d_max)
        self.theta = float(theta)
        self.gamma = float(gamma)
        self.use_abs_cos = bool(use_abs_cos)
        self.weight_use_abs_cos = bool(weight_use_abs_cos)
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

        if self.d == "adaptive":
            if not 0.0 < self.adaptive_variance_threshold <= 1.0:
                raise ValueError("adaptive_variance_threshold must be in (0, 1]")
            if self.adaptive_d_min < 1 or self.adaptive_d_min > self.D:
                raise ValueError("adaptive_d_min must be between 1 and the ambient dimension")
            if self.adaptive_d_max is not None and (
                self.adaptive_d_max < self.adaptive_d_min or self.adaptive_d_max > self.D
            ):
                raise ValueError("adaptive_d_max must be between adaptive_d_min and the ambient dimension")
        else:
            self.d = int(self.d)
            if self.d < 1 or self.d > self.D:
                raise ValueError("d must be between 1 and the ambient dimension, or 'adaptive'")
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

    def _velocity_aware_distance(self, diff, velocity, use_abs_cos=None):
        d0 = np.linalg.norm(diff, axis=1)
        if np.linalg.norm(velocity) < self.eps:
            return d0, np.zeros_like(d0)

        if use_abs_cos is None:
            use_abs_cos = self.use_abs_cos

        Wi = np.repeat(velocity[None, :], diff.shape[0], axis=0)
        cos_val = self._cosine_rows(Wi, diff, use_abs=use_abs_cos)
        vel_term = 1.0 - self._sigmoid(self.gamma * cos_val)
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
            score, _ = self._velocity_aware_distance(diff, velocity[i])
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

        cos_for_distance = np.abs(cos_val) if self.use_abs_cos else cos_val
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
        fixed_d = None if self.d == "adaptive" else self.d
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
                "history": self.history,
            }
        return self.X
