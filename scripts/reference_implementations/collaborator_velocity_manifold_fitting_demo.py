
"""
Velocity-aware manifold fitting prototypes.

This file implements:
1) Original velocity-aware MF update:
   full mean-shift + projected-velocity transport.
2) Normal-only version:
   mean-shift is projected to the estimated normal space; velocity is projected
   but not used to move cell states along the manifold.

It also includes synthetic experiments on:
- noisy circle
- noisy Y-shaped branching trajectory

Run:
    python velocity_manifold_fitting_demo.py

Dependencies:
    numpy scipy scikit-learn matplotlib pandas
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


Array = np.ndarray


@dataclass
class VMFParams:
    k: int = 25
    d: int = 1
    n_iter: int = 8
    theta: float = 0.25
    gamma: float = 8.0
    beta: float = 2.0
    kappa: float = 1.0
    eta_g: float = 0.45
    cv: float = 0.15
    max_step_frac: float = 0.35
    random_state: int = 0


def _row_norm(X: Array, eps: float = 1e-12) -> Array:
    return np.sqrt(np.sum(X * X, axis=1, keepdims=True)) + eps


def _cosine_rows(a: Array, b: Array, eps: float = 1e-12) -> Array:
    return np.sum(a * b, axis=1) / ((_row_norm(a, eps)[:, 0]) * (_row_norm(b, eps)[:, 0]) + eps)


def _sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def build_velocity_aware_knn(X: Array, W: Array, params: VMFParams) -> Array:
    """
    Approximate velocity-aware KNN:
    first get Euclidean candidates, then rerank by
        (1-theta)*euclidean_distance + theta*(1-sigmoid(gamma*cos(W_i, X_j-X_i))).
    """
    n = X.shape[0]
    cand_k = min(n - 1, max(params.k * 4, params.k + 5))
    nn = NearestNeighbors(n_neighbors=cand_k + 1).fit(X)
    _, cand = nn.kneighbors(X)
    cand = cand[:, 1:]

    N = np.zeros((n, params.k), dtype=int)
    for i in range(n):
        J = cand[i]
        disp = X[J] - X[i]
        dist = norm(disp, axis=1)
        cos = _cosine_rows(np.repeat(W[i][None, :], len(J), axis=0), disp)
        score = (1 - params.theta) * dist + params.theta * (1 - _sigmoid(params.gamma * cos))
        N[i] = J[np.argsort(score)[: params.k]]
    return N


def local_pca_projector(Z: Array, N: Array, weights: Array, d: int) -> Tuple[Array, Array]:
    """
    Returns tangent bases U_i and projectors P_i = U_i U_i^T.
    U has shape (n, D, d), P has shape (n, D, D).
    """
    n, D = Z.shape
    U_all = np.zeros((n, D, d))
    P_all = np.zeros((n, D, D))
    for i in range(n):
        J = N[i]
        C = Z[J] - Z[i]
        w = weights[i][:, None]
        # weighted covariance
        M = (C * w).T @ C
        vals, vecs = np.linalg.eigh(M)
        U = vecs[:, np.argsort(vals)[::-1][:d]]
        U_all[i] = U
        P_all[i] = U @ U.T
    return U_all, P_all


def compute_weights(Z: Array, W: Array, N: Array, params: VMFParams) -> Tuple[Array, Array]:
    """
    Kernel weights with optional directional factor.
    """
    n = Z.shape[0]
    K = N.shape[1]
    weights = np.zeros((n, K))
    h = np.zeros(n)
    for i in range(n):
        J = N[i]
        disp = Z[J] - Z[i]
        dist = norm(disp, axis=1)
        h_i = np.max(dist) + 1e-12
        h[i] = h_i
        radial = np.maximum(1 - (dist / h_i) ** 2, 0.0) ** params.beta
        cos_abs = np.abs(_cosine_rows(np.repeat(W[i][None, :], K, axis=0), disp))
        directional = np.exp(params.kappa * cos_abs)
        w = radial * directional + 1e-12
        weights[i] = w / np.sum(w)
    return weights, h


def project_velocity(W: Array, P: Array) -> Array:
    return np.einsum("nij,nj->ni", P, W)


def fit_velocity_mf(
    X: Array,
    W: Array,
    params: VMFParams,
    mode: Literal["original", "normal_only"] = "original",
    freeze_graph: bool = True,
) -> Dict[str, Array]:
    """
    Original:
        Z <- Z + eta_g*(local_mean-Z) + cv*h*projected_velocity_direction
    Normal-only:
        Z <- Z + eta_g*(I-P)*(local_mean-Z)
        velocity is projected but does not transport Z tangentially.
    """
    Z = X.copy()
    Wcur = W.copy()

    N = build_velocity_aware_knn(Z, Wcur, params)

    for t in range(params.n_iter):
        if (not freeze_graph) and t > 0:
            N = build_velocity_aware_knn(Z, Wcur, params)

        weights, h = compute_weights(Z, Wcur, N, params)
        U, P = local_pca_projector(Z, N, weights, params.d)
        Vproj = project_velocity(Wcur, P)

        Z_new = Z.copy()
        for i in range(Z.shape[0]):
            J = N[i]
            m_i = np.sum(weights[i][:, None] * Z[J], axis=0)
            g_i = m_i - Z[i]

            step_cap = params.max_step_frac * h[i]
            if mode == "original":
                vdir = Vproj[i] / (norm(Vproj[i]) + 1e-12)
                step = params.eta_g * g_i + params.cv * h[i] * vdir
            elif mode == "normal_only":
                normal_g = g_i - P[i] @ g_i
                step = params.eta_g * normal_g
            else:
                raise ValueError("mode must be 'original' or 'normal_only'")

            s_norm = norm(step)
            if s_norm > step_cap:
                step = step * (step_cap / (s_norm + 1e-12))
            Z_new[i] = Z[i] + step

        Z = Z_new
        Wcur = Vproj

    weights, h = compute_weights(Z, Wcur, N, params)
    U, P = local_pca_projector(Z, N, weights, params.d)
    Vproj = project_velocity(W, P)
    return {"Z": Z, "V": Vproj, "N": N, "U": U, "P": P}


# ---------- Synthetic data ----------

def make_circle(n: int = 600, sigma_x: float = 0.08, sigma_v: float = 0.25, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 2 * np.pi, size=n))
    X0 = np.c_[np.cos(t), np.sin(t), np.zeros(n)]
    V0 = np.c_[-np.sin(t), np.cos(t), np.zeros(n)]
    X = X0 + rng.normal(scale=sigma_x, size=X0.shape)
    W = V0 + rng.normal(scale=sigma_v, size=V0.shape)
    return X, W, X0, V0, t


def make_yshape(n_per_branch: int = 220, sigma_x: float = 0.06, sigma_v: float = 0.22, seed: int = 1):
    rng = np.random.default_rng(seed)
    angles = np.array([np.pi / 2, 7 * np.pi / 6, 11 * np.pi / 6])
    X0_list, V0_list, branch = [], [], []
    for b, a in enumerate(angles):
        r = np.sort(rng.uniform(0.05, 1.0, size=n_per_branch))
        direction = np.array([np.cos(a), np.sin(a), 0.0])
        X0_list.append(r[:, None] * direction[None, :])
        V0_list.append(np.repeat(direction[None, :], n_per_branch, axis=0))
        branch += [b] * n_per_branch
    X0 = np.vstack(X0_list)
    V0 = np.vstack(V0_list)
    branch = np.array(branch)
    X = X0 + rng.normal(scale=sigma_x, size=X0.shape)
    W = V0 + rng.normal(scale=sigma_v, size=V0.shape)
    return X, W, X0, V0, branch


# ---------- Metrics ----------

def velocity_cosine(Vhat: Array, Vtrue: Array) -> float:
    c = _cosine_rows(Vhat, Vtrue)
    return float(np.nanmean(c))


def rmse_to_truth(Z: Array, X0: Array) -> float:
    return float(np.sqrt(np.mean(np.sum((Z - X0) ** 2, axis=1))))


def knn_preservation(X: Array, Z: Array, k: int = 20) -> float:
    I1 = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(return_distance=False)[:, 1:]
    I2 = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(return_distance=False)[:, 1:]
    return float(np.mean([len(set(I1[i]).intersection(I2[i])) / k for i in range(X.shape[0])]))


def local_metric_distortion(X: Array, Z: Array, k: int = 20) -> float:
    I = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(return_distance=False)[:, 1:]
    vals = []
    for i in range(X.shape[0]):
        for j in I[i]:
            dx = norm(X[i] - X[j]) + 1e-12
            dz = norm(Z[i] - Z[j]) + 1e-12
            vals.append(abs(np.log(dz / dx)))
    return float(np.mean(vals))


def branch_mixing_score(Z: Array, labels: Array, k: int = 20) -> float:
    """
    Fraction of kNN edges connecting different branches.
    For Y-shape, too much cross-branch mixing near the center is bad,
    but collapse can also inflate this. Use together with KNN preservation/LMD.
    """
    I = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(return_distance=False)[:, 1:]
    mix = np.mean([np.mean(labels[I[i]] != labels[i]) for i in range(Z.shape[0])])
    return float(mix)


def evaluate(name: str, X: Array, W: Array, X0: Array, V0: Array, out: Dict[str, Array], labels=None) -> Dict[str, float]:
    Z, V = out["Z"], out["V"]
    row = {
        "method": name,
        "position_rmse": rmse_to_truth(Z, X0),
        "velocity_cosine": velocity_cosine(V, V0),
        "knn_preservation": knn_preservation(X, Z, k=20),
        "local_metric_distortion": local_metric_distortion(X, Z, k=20),
        "mean_displacement": float(np.mean(norm(Z - X, axis=1))),
    }
    if labels is not None:
        row["branch_mixing"] = branch_mixing_score(Z, labels, k=20)
    return row


def plot_results(dataset_name: str, X: Array, X0: Array, V0: Array, results: Dict[str, Dict[str, Array]], labels=None):
    ncols = 1 + len(results)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    color = labels if labels is not None else np.arange(X.shape[0])
    axes[0].scatter(X[:, 0], X[:, 1], c=color, s=8)
    axes[0].set_title(f"{dataset_name}: noisy input")
    axes[0].set_aspect("equal")

    for ax, (name, out) in zip(axes[1:], results.items()):
        Z = out["Z"]
        V = out["V"]
        ax.scatter(Z[:, 0], Z[:, 1], c=color, s=8)
        step = max(1, Z.shape[0] // 80)
        ax.quiver(
            Z[::step, 0], Z[::step, 1],
            V[::step, 0], V[::step, 1],
            angles="xy", scale_units="xy", scale=18, width=0.003,
        )
        ax.set_title(name)
        ax.set_aspect("equal")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    return fig


def run_one_experiment(dataset: Literal["circle", "yshape"], seed: int = 0, outdir: str = "velocity_mf_outputs"):
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if dataset == "circle":
        X, W, X0, V0, aux = make_circle(seed=seed)
        labels = None
    elif dataset == "yshape":
        X, W, X0, V0, labels = make_yshape(seed=seed)
    else:
        raise ValueError(dataset)

    params = VMFParams(k=25, d=1, n_iter=10, eta_g=0.45, cv=0.15, theta=0.25)
    # dataclass workaround for old Python parsers not needed; keep explicit:
    params.random_state = seed

    original = fit_velocity_mf(X, W, params, mode="original")
    normal = fit_velocity_mf(X, W, params, mode="normal_only")

    rows = [
        {
            "method": "raw",
            "position_rmse": rmse_to_truth(X, X0),
            "velocity_cosine": velocity_cosine(W, V0),
            "knn_preservation": 1.0,
            "local_metric_distortion": 0.0,
            "mean_displacement": 0.0,
            **({"branch_mixing": branch_mixing_score(X, labels, k=20)} if labels is not None else {}),
        },
        evaluate("original_full_meanshift", X, W, X0, V0, original, labels),
        evaluate("normal_only", X, W, X0, V0, normal, labels),
    ]
    df = pd.DataFrame(rows)

    fig = plot_results(dataset, X, X0, V0, {
        "original full update": original,
        "normal-only update": normal,
    }, labels=labels)
    fig.savefig(Path(outdir) / f"{dataset}_comparison.png", dpi=180)
    plt.close(fig)

    df.to_csv(Path(outdir) / f"{dataset}_metrics.csv", index=False)
    return df


def main():
    all_rows = []
    for dataset in ["circle", "yshape"]:
        for seed in range(5):
            df = run_one_experiment(dataset, seed=seed)
            df["dataset"] = dataset
            df["seed"] = seed
            all_rows.append(df)

    res = pd.concat(all_rows, ignore_index=True)
    summary = res.groupby(["dataset", "method"]).agg(["mean", "std"])
    Path("velocity_mf_outputs").mkdir(exist_ok=True)
    res.to_csv("velocity_mf_outputs/all_metrics.csv", index=False)
    summary.to_csv("velocity_mf_outputs/summary_metrics.csv")

    print("\nPer-run metrics saved to velocity_mf_outputs/all_metrics.csv")
    print("Summary saved to velocity_mf_outputs/summary_metrics.csv")
    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()
