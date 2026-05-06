import numpy as np
from sklearn.neighbors import NearestNeighbors

class VelocityManifoldFitter:
    def __init__(
        self,
        Y,
        W,
        k=10,
        d=1,
        theta=0.0,
        gamma=1.0,
        use_abs_cos=False,
        kappa=1.0,
        h=0.8,
        beta=2.0,
        alpha=0.3,
        step_set=None,
        T=10,
        recompute_neighbors=True,
        random_state=None,
        candidate_mult=3,
        neighbor_update_freq = 5,
        eps=1e-12,
    ):
        if random_state is not None:
            np.random.seed(random_state)

        self.Y = np.asarray(Y, dtype=float)
        self.W = np.asarray(W, dtype=float)

        assert self.Y.shape == self.W.shape, "Y and W must match shape"
        self.n, self.D = self.Y.shape

        self.k = int(k)
        self.d = int(d)
        self.theta = theta
        self.gamma = gamma
        self.use_abs_cos = bool(use_abs_cos)
        self.kappa = kappa
        self.h = h
        self.beta = beta
        self.alpha = alpha
        self.T = int(T)
        self.recompute_neighbors = bool(recompute_neighbors)
        self.candidate_mult = int(candidate_mult)
        self.neighbor_update_freq = int(neighbor_update_freq)
        self.eps = eps

        if step_set is None:
            self.step_set = np.linspace(0, 1.0, 10)
        else:
            self.step_set = np.asarray(step_set, dtype=float)

        self.X = self.Y.copy()
        self.U = None
        self.v = None
        self.neighbors = None   # (n, k)
        self.weights = None     # (n, k)
        self.history = []

    def _sigmoid(self, z):
        z = np.clip(z, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))
    
    def _cosine(self, a, b, use_abs=False):
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + self.eps)
        c = float(np.dot(a, b) / denom)
        return abs(c) if use_abs else c

    def _distance(self, xi, xj, Wi):
        d0 = float(np.linalg.norm(xi - xj))
    
        Wi_norm = np.linalg.norm(Wi)
        if Wi_norm < 1e-12:
            return d0
    
        direction = xj - xi
        cos_val = self._cosine(Wi, direction, use_abs=True)
    
        vel_term = 1.0 - self._sigmoid(self.gamma * cos_val)
    
        return (1.0 - self.theta) * d0 + self.theta * vel_term


    def _kernel_weight(self, xi, xj, Wi):
        d = self._distance(xi, xj, Wi)
        direction = xj - xi
        cos_val = self._cosine(Wi, direction, use_abs=True)
    
        spatial = max(0.0, 1.0 - d / (self.h + self.eps)) ** self.beta
        directional = np.exp(self.kappa * cos_val)
        return spatial * directional


    def _build_neighbors(self):
        m = min(self.n, self.k * self.candidate_mult + 1)
    
        nbrs = NearestNeighbors(n_neighbors=m, metric='euclidean').fit(self.X)
        candidates = nbrs.kneighbors(return_distance=False)  # (n, m)
    
        neighbors = np.zeros((self.n, self.k), dtype=int)
    
        for i in range(self.n):
            xi = self.X[i]
            Wi = self.W[i]
    
            # remove self
            cand = candidates[i]
            cand = cand[cand != i]
    
            Xc = self.X[cand]
    
            diff = Xc - xi
            d0 = np.linalg.norm(diff, axis=1)
    
            Wi_norm = np.linalg.norm(Wi)
    
            if Wi_norm < self.eps:
                score = d0
            else:
                cos_val = diff @ Wi / (Wi_norm * (d0 + self.eps))
                if self.use_abs_cos:
                    cos_val = np.abs(cos_val)
    
                vel_term = 1.0 - self._sigmoid(self.gamma * cos_val)
                score = (1.0 - self.theta) * d0 + self.theta * vel_term
    
            idx = np.argsort(score)[:self.k]
            neighbors[i] = cand[idx]
    
        self.neighbors = neighbors
    
    
    def _update_weights(self, velocity_mode="projected", blend_lambda=0.0):
        n, k = self.n, self.k
        X = self.X
        neigh = self.neighbors  # (n, k)
    
        Xj = X[neigh]              # (n, k, D)
        xi = X[:, None, :]         # (n, 1, D)
        diff = Xj - xi             # (n, k, D)
    
        # choose velocity
        if velocity_mode == "projected":
            Wi = self.v if self.v is not None else self.W
        elif velocity_mode == "raw":
            Wi = self.W
        elif velocity_mode == "blend":
            vi = self.v if self.v is not None else self.W
            Wi = (1 - blend_lambda) * self.W + blend_lambda * vi
        else:
            raise ValueError
    
        Wi = Wi[:, None, :]        # (n, 1, D)
    
        # norms
        diff_norm = np.linalg.norm(diff, axis=2) + self.eps
        Wi_norm = np.linalg.norm(Wi, axis=2) + self.eps
    
        # cosine
        cos_val = np.sum(Wi * diff, axis=2) / (Wi_norm * diff_norm)
        if self.use_abs_cos:
            cos_val = np.abs(cos_val)
    
        # distance
        vel_term = 1.0 - self._sigmoid(self.gamma * cos_val)
        d = (1 - self.theta) * diff_norm + self.theta * vel_term
    
        # kernel
        spatial = np.maximum(0.0, 1.0 - d / (self.h + self.eps)) ** self.beta
        directional = np.exp(self.kappa * cos_val)
        w_tilde = spatial * directional
    
        # normalize
        w_tilde += self.eps
        self.weights = w_tilde / w_tilde.sum(axis=1, keepdims=True)

    def _compute_local_tangent(self):
        n, D, d = self.n, self.D, self.d
        U_all = np.zeros((n, D, d), dtype=float)

        for i in range(n):
            neigh = self.neighbors[i]
            w = self.weights[i]

            x_bar = np.sum(w[:, None] * self.X[neigh], axis=0)
            diff = self.X[neigh] - x_bar  # (k, D)

            C = (w[:, None] * diff).T @ diff
            C = 0.5 * (C + C.T)

            eigvals, eigvecs = np.linalg.eigh(C)
            idx = np.argsort(eigvals)[::-1][:d]
            U_all[i] = eigvecs[:, idx]

        self.U = U_all

    def _project_velocity(self):
        n = self.n
        v_all = np.zeros_like(self.W)

        for i in range(n):
            Ui = self.U[i]   # (D, d)
            Wi = self.W[i]
            v_all[i] = Ui @ (Ui.T @ Wi)

        self.v = v_all

    # -----------------------
    # Fit
    # -----------------------
    def fit(self, eta=1.0, mode="unweighted"):
        self._build_neighbors()
        self._update_weights()
    
        for t in range(self.T):
            if self.recompute_neighbors and (t % self.neighbor_update_freq == 0):
                self._build_neighbors()
    
            self._update_weights()
            self._compute_local_tangent()
            self._project_velocity()
    
            neigh = self.neighbors
            w = self.weights
            
            Xj = self.X[neigh]
            x_bar = np.sum(w[:, :, None] * Xj, axis=1)
            
            xi = self.X
            delta = x_bar - xi   # (n, D)
    
        # ----------------------------
        # Final direction
        # ----------------------------
        d = (1 - self.alpha) * delta + self.alpha * self.v
        
        # ----------------------------
        # Line search
        # ----------------------------
        num = -np.sum((xi - x_bar) * d, axis=1)
        den = np.sum(d * d, axis=1) + self.eps
        eta_star = num / den
        
        # ----------------------------
        # Quantile-based movement weight
        # ----------------------------
        if mode == "weighted":
            dist = np.linalg.norm(xi - x_bar, axis=1)  # (n,)
        
            # global rank-based quantile
            ranks = np.argsort(np.argsort(dist))
            ranks = ranks.astype(float) / (self.n - 1 + self.eps)
        
            # closest to centroid -> 1
            # furthest from centroid -> 0
            move_weight = 1.0 - ranks
        
            eta_star = eta_star * move_weight
        
        elif mode == "unweighted":
            pass
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.X = xi + eta_star[:, None] * d
    
        return self.X

        