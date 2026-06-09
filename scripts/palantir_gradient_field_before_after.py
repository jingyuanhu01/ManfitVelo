"""Visualize Palantir entropy gradients before and after manifold fitting.

This example reads the processed Palantir bone marrow data from the sibling
``potential_curvature`` project and compares:

* before: local finite-difference entropy gradients in raw PCA space
* after: self-consistent MANFIT entropy gradients in PCA space

The H5AD is not copied into this repo. Run with the potential_curvature venv if
the local environment does not have anndata installed:

    /Users/jh/Projects/potential_curvature/.venv-potential-curvature/bin/python \
        scripts/palantir_gradient_field_before_after.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POTENTIAL_CURVATURE_ROOT = Path("~/Projects/potential_curvature").expanduser()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scalar_potential_manfit import (  # noqa: E402
    estimate_gradient_from_neighbors,
    fit_self_consistent_gradient_manfit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5ad",
        type=Path,
        default=POTENTIAL_CURVATURE_ROOT
        / "data/palantir_bone_marrow/marrow_palantir_processed.h5ad",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/real_data_palantir",
    )
    parser.add_argument("--grid-size", type=int, default=30)
    parser.add_argument("--pca-dims", type=int, default=30)
    parser.add_argument("--gradient-neighbors", type=int, default=42)
    parser.add_argument("--grid-neighbors", type=int, default=18)
    parser.add_argument("--outer-iterations", type=int, default=4)
    parser.add_argument("--fit-neighbors", type=int, default=15)
    parser.add_argument("--inner-iterations", type=int, default=2)
    parser.add_argument("--eta-g", type=float, default=0.35)
    parser.add_argument("--theta", type=float, default=0.2)
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def read_palantir_h5ad(path: Path, pca_dims: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read only the fields needed from the processed Palantir H5AD."""

    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError(
            "anndata is required to read the Palantir H5AD. Run this script with "
            "/Users/jh/Projects/potential_curvature/.venv-potential-curvature/bin/python."
        ) from exc

    data = ad.read_h5ad(path, backed="r")
    X_pca = np.asarray(data.obsm["X_pca"], dtype=float)
    X_pca = X_pca[:, : min(int(pca_dims), X_pca.shape[1])]
    entropy = data.obs["palantir_entropy"].to_numpy(dtype=float)
    pseudotime = data.obs["palantir_pseudotime"].to_numpy(dtype=float)
    return X_pca, entropy, pseudotime


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - np.mean(values)) / (np.std(values) + 1e-12)


def make_display_grid(
    X_display: np.ndarray,
    grid_size: int = 30,
    pad_fraction: float = 0.04,
) -> np.ndarray:
    """Create a regular PC1/PC2 grid and keep points inside the display hull."""

    pad_x = pad_fraction * np.ptp(X_display[:, 0])
    pad_y = pad_fraction * np.ptp(X_display[:, 1])
    gx = np.linspace(X_display[:, 0].min() - pad_x, X_display[:, 0].max() + pad_x, grid_size)
    gy = np.linspace(X_display[:, 1].min() - pad_y, X_display[:, 1].max() + pad_y, grid_size)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    inside = Delaunay(X_display).find_simplex(grid) >= 0
    return grid[inside]


def smooth_cell_gradients_to_grid(
    X_display: np.ndarray,
    cell_gradient_display: np.ndarray,
    grid: np.ndarray,
    n_neighbors: int = 18,
) -> np.ndarray:
    """Interpolate local cell gradients onto grid points with Gaussian weights."""

    n_neighbors = min(int(n_neighbors), X_display.shape[0])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_display)
    distances, indices = nbrs.kneighbors(grid)
    bandwidth = np.median(distances[:, -1]) + 1e-12
    weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    weights /= np.sum(weights, axis=1, keepdims=True) + 1e-12
    return np.einsum("gk,gkd->gd", weights, cell_gradient_display[indices])


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vectors, axis=1)
    return vectors / (norms[:, None] + eps), norms


def plot_gradient_panel(
    X_raw_display: np.ndarray,
    X_fit_display: np.ndarray,
    entropy: np.ndarray,
    raw_grid: np.ndarray,
    fitted_grid: np.ndarray,
    raw_grid_gradient: np.ndarray,
    fitted_grid_gradient: np.ndarray,
    output_path: Path,
) -> None:
    """Write a before/after gradient-field panel."""

    raw_unit, raw_norm = normalize_vectors(raw_grid_gradient)
    fitted_unit, fitted_norm = normalize_vectors(fitted_grid_gradient)
    vmax = np.nanpercentile(entropy, 99)
    vmin = np.nanpercentile(entropy, 1)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    panel_specs = [
        (axes[0], X_raw_display, raw_grid, raw_unit, raw_norm, "Before fitting: PCA local entropy gradient"),
        (axes[1], X_fit_display, fitted_grid, fitted_unit, fitted_norm, "After fitting: PCA MANFIT entropy gradient"),
    ]

    for ax, X_display, grid, gradient_unit, gradient_norm, title in panel_specs:
        scatter = ax.scatter(
            X_display[:, 0],
            X_display[:, 1],
            c=entropy,
            s=7,
            alpha=0.34,
            linewidths=0,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.quiver(
            grid[:, 0],
            grid[:, 1],
            gradient_unit[:, 0],
            gradient_unit[:, 1],
            color="#111827",
            angles="xy",
            scale_units="xy",
            scale=2.2,
            width=0.0032,
            headwidth=3.6,
            headlength=4.6,
            headaxislength=4.0,
            alpha=0.86,
        )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    fig.colorbar(scatter, ax=axes, fraction=0.035, pad=0.02, label="Palantir fate entropy")
    fig.suptitle("Palantir Bone Marrow: PCA Entropy Gradient Field Before and After Manifold Fitting", y=1.03)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X_pca, entropy, pseudotime = read_palantir_h5ad(args.h5ad, pca_dims=args.pca_dims)
    entropy_z = zscore(entropy)

    raw_cell_gradient = estimate_gradient_from_neighbors(
        X_pca,
        entropy_z,
        n_neighbors=args.gradient_neighbors,
        ridge=5e-2,
    )

    fitted = fit_self_consistent_gradient_manfit(
        X_pca,
        entropy_z,
        outer_iterations=args.outer_iterations,
        gradient_n_neighbors=args.gradient_neighbors,
        gradient_ridge=5e-2,
        k=args.fit_neighbors,
        inner_T=args.inner_iterations,
        eta_g=args.eta_g,
        theta=args.theta,
        kappa=args.kappa,
        random_state=args.random_state,
    )

    fitted_position = fitted["position"]
    fitted_cell_gradient = fitted["gradient"]
    raw_display = X_pca[:, :2]
    fitted_display = fitted_position[:, :2]

    raw_grid = make_display_grid(raw_display, grid_size=args.grid_size)
    fitted_grid = make_display_grid(fitted_display, grid_size=args.grid_size)
    raw_grid_gradient = smooth_cell_gradients_to_grid(
        raw_display,
        raw_cell_gradient[:, :2],
        raw_grid,
        n_neighbors=args.grid_neighbors,
    )
    fitted_grid_gradient = smooth_cell_gradients_to_grid(
        fitted_display,
        fitted_cell_gradient[:, :2],
        fitted_grid,
        n_neighbors=args.grid_neighbors,
    )

    panel_path = args.output_dir / "palantir_entropy_gradient_pca_before_after_manifold_fit.png"
    plot_gradient_panel(
        raw_display,
        fitted_display,
        entropy,
        raw_grid,
        fitted_grid,
        raw_grid_gradient,
        fitted_grid_gradient,
        panel_path,
    )

    raw_vectors = pd.DataFrame(
        {
            "panel": "before_pca_local_gradient",
            "grid_pc_1": raw_grid[:, 0],
            "grid_pc_2": raw_grid[:, 1],
            "grad_pc_1": raw_grid_gradient[:, 0],
            "grad_pc_2": raw_grid_gradient[:, 1],
            "grad_norm": np.linalg.norm(raw_grid_gradient, axis=1),
        }
    )
    fitted_vectors = pd.DataFrame(
        {
            "panel": "after_pca_manfit_gradient",
            "grid_pc_1": fitted_grid[:, 0],
            "grid_pc_2": fitted_grid[:, 1],
            "grad_pc_1": fitted_grid_gradient[:, 0],
            "grad_pc_2": fitted_grid_gradient[:, 1],
            "grad_norm": np.linalg.norm(fitted_grid_gradient, axis=1),
        }
    )
    vectors = pd.concat([raw_vectors, fitted_vectors], ignore_index=True)
    vectors.to_csv(args.output_dir / "palantir_entropy_gradient_pca_before_after_vectors.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "n_cells": X_pca.shape[0],
                "pca_dims": X_pca.shape[1],
                "entropy_min": float(np.min(entropy)),
                "entropy_max": float(np.max(entropy)),
                "pseudotime_min": float(np.min(pseudotime)),
                "pseudotime_max": float(np.max(pseudotime)),
                "outer_iterations": int(args.outer_iterations),
                "fit_neighbors": int(args.fit_neighbors),
                "inner_iterations": int(args.inner_iterations),
                "raw_grid_vectors": int(raw_grid.shape[0]),
                "fitted_grid_vectors": int(fitted_grid.shape[0]),
                "mean_fit_confidence": float(np.mean(fitted["confidence"])),
            }
        ]
    )
    summary.to_csv(args.output_dir / "palantir_entropy_gradient_pca_before_after_summary.csv", index=False)

    print(panel_path)
    print(args.output_dir / "palantir_entropy_gradient_pca_before_after_vectors.csv")
    print(args.output_dir / "palantir_entropy_gradient_pca_before_after_summary.csv")


if __name__ == "__main__":
    main()
