"""Generate a three-panel PCA report view for the P450 fitness landscape.

The figure mirrors the RNA velocity report style:

* observed PCA positions colored by measured T50
* position + gradient MANFIT positions colored by measured T50
* before/after movement paths in the same PCA display basis
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.velocity_manifold_fitter import VelocityManifoldFitter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "protein_latent_paper",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "manual-potential-gradient",
    )
    parser.add_argument("--gradient-neighbors", type=int, default=24)
    parser.add_argument("--gradient-ridge", type=float, default=1e-2)
    parser.add_argument("--fit-neighbors", type=int, default=20)
    parser.add_argument("--fit-iterations", type=int, default=8)
    parser.add_argument("--eta-g", type=float, default=0.35)
    parser.add_argument("--theta", type=float, default=0.2)
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--max-paths", type=int, default=160)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def read_p450_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read standardized P450 genotype features and measured T50."""

    onehot = np.load(data_dir / "p450_numeric_onehot.npy").astype(float)
    t50 = np.load(data_dir / "p450_numeric_t50.npy").astype(float)
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(onehot)
    return X, t50


def zscore(values: np.ndarray) -> np.ndarray:
    return (values - np.mean(values)) / (np.std(values) + 1e-12)


def estimate_gradient_from_neighbors(
    X: np.ndarray,
    values: np.ndarray,
    n_neighbors: int = 24,
    ridge: float = 1e-2,
) -> np.ndarray:
    """Estimate an ambient scalar gradient field from local linear fits."""

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


def fit_position_plus_gradient(
    X: np.ndarray,
    V: np.ndarray,
    *,
    k: int,
    T: int,
    eta_g: float,
    theta: float,
    kappa: float,
    random_state: int,
) -> dict[str, np.ndarray]:
    """Run gradient-aware manifold fitting from positions and gradient vectors."""

    fitter = VelocityManifoldFitter(
        X,
        V,
        d_mode="adaptive",
        adaptive_variance_threshold=0.85,
        adaptive_d_min=2,
        k=k,
        T=T,
        eta_g=eta_g,
        theta=theta,
        kappa=kappa,
        bandwidth_mode="variable",
        use_PCA=False,
        random_state=random_state,
    )
    result = fitter.fit(update_mode="normal_only", return_dict=True)
    return {
        "position": result["X"],
        "gradient": result["V"],
    }


def choose_path_indices(n_points: int, max_paths: int, random_state: int) -> np.ndarray:
    max_paths = min(max(int(max_paths), 1), int(n_points))
    if max_paths == n_points:
        return np.arange(n_points)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_points, size=max_paths, replace=False))


def set_shared_limits(axes: np.ndarray, *arrays: np.ndarray) -> None:
    Z = np.vstack([array[:, :2] for array in arrays])
    mins = Z.min(axis=0)
    maxs = Z.max(axis=0)
    centers = (mins + maxs) / 2
    radius = max(float(np.max(maxs - mins) / 2), np.finfo(float).eps)
    pad = 0.10 * radius
    for ax in axes:
        ax.set_xlim(centers[0] - radius - pad, centers[0] + radius + pad)
        ax.set_ylim(centers[1] - radius - pad, centers[1] + radius + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")


def scale_vectors_for_display(Z: np.ndarray, V: np.ndarray, arrow_frac: float = 0.07) -> np.ndarray:
    """Scale projected gradient vectors to a readable visual length."""

    span = max(np.ptp(Z[:, 0]), np.ptp(Z[:, 1]), np.finfo(float).eps)
    norms = np.linalg.norm(V, axis=1)
    nonzero = norms[norms > np.finfo(float).eps]
    if nonzero.size == 0:
        return V
    robust_norm = np.percentile(nonzero, 95)
    return V * (arrow_frac * span / robust_norm)


def plot_three_panel(
    Z_before: np.ndarray,
    Z_after: np.ndarray,
    V_before: np.ndarray,
    V_after: np.ndarray,
    fitness: np.ndarray,
    path_indices: np.ndarray,
    output_path: Path,
) -> None:
    vmax = np.nanpercentile(fitness, 99)
    vmin = np.nanpercentile(fitness, 1)
    V_before_plot = scale_vectors_for_display(Z_before, V_before)
    V_after_plot = scale_vectors_for_display(Z_after, V_after)
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 5.8), constrained_layout=False)

    axes[0].scatter(
        Z_before[:, 0],
        Z_before[:, 1],
        c=fitness,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=24,
        alpha=0.82,
        linewidths=0,
    )
    axes[0].quiver(
        Z_before[path_indices, 0],
        Z_before[path_indices, 1],
        V_before_plot[path_indices, 0],
        V_before_plot[path_indices, 1],
        angles="xy",
        scale_units="xy",
        scale=0.55,
        color="#111827",
        width=0.0032,
        alpha=0.78,
    )
    axes[0].set_title("Before manifold fitting")

    axes[1].scatter(
        Z_after[:, 0],
        Z_after[:, 1],
        c=fitness,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=24,
        alpha=0.82,
        linewidths=0,
    )
    axes[1].quiver(
        Z_after[path_indices, 0],
        Z_after[path_indices, 1],
        V_after_plot[path_indices, 0],
        V_after_plot[path_indices, 1],
        angles="xy",
        scale_units="xy",
        scale=0.55,
        color="#111827",
        width=0.0032,
        alpha=0.78,
    )
    axes[1].set_title("After position + gradient MANFIT")

    axes[2].scatter(
        Z_before[path_indices, 0],
        Z_before[path_indices, 1],
        s=19,
        color="#b8bec5",
        alpha=0.65,
        label="before",
    )
    axes[2].scatter(
        Z_after[path_indices, 0],
        Z_after[path_indices, 1],
        s=21,
        color="#e23d62",
        alpha=0.9,
        label="after",
    )
    for idx in path_indices:
        axes[2].plot(
            [Z_before[idx, 0], Z_after[idx, 0]],
            [Z_before[idx, 1], Z_after[idx, 1]],
            color="#9ca3af",
            alpha=0.34,
            linewidth=0.85,
        )
    axes[2].set_title("Movement path in PCA space")
    axes[2].legend(frameon=False, loc="upper right")

    set_shared_limits(axes, Z_before, Z_after)
    fig.suptitle("P450 fitness landscape: PCA before/after manifold fitting", y=0.98)
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.11, top=0.86, wspace=0.16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X, t50 = read_p450_data(args.data_dir)
    scalar = zscore(t50)
    gradient = estimate_gradient_from_neighbors(
        X,
        scalar,
        n_neighbors=args.gradient_neighbors,
        ridge=args.gradient_ridge,
    )
    fit = fit_position_plus_gradient(
        X,
        gradient,
        k=args.fit_neighbors,
        T=args.fit_iterations,
        eta_g=args.eta_g,
        theta=args.theta,
        kappa=args.kappa,
        random_state=args.random_state,
    )

    pca = PCA(n_components=2, random_state=args.random_state).fit(X)
    Z_before = pca.transform(X)
    Z_after = pca.transform(fit["position"])
    V_before = gradient @ pca.components_.T
    V_after = fit["gradient"] @ pca.components_.T
    path_indices = choose_path_indices(X.shape[0], args.max_paths, args.random_state)

    output_path = args.output_dir / "fitness_landscape_pca_three_panel_before_after.png"
    plot_three_panel(Z_before, Z_after, V_before, V_after, t50, path_indices, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
