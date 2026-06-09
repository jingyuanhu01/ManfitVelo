"""Visualize Palantir entropy gradients before and after potential fitting.

This example reads the processed Palantir bone marrow data from the sibling
``potential_curvature`` project and compares:

* before: local finite-difference entropy gradients on the observed UMAP
* after: analytic gradients from a TPS-fitted entropy potential on UMAP

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
POTENTIAL_CURVATURE_SRC = POTENTIAL_CURVATURE_ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if POTENTIAL_CURVATURE_SRC.exists() and str(POTENTIAL_CURVATURE_SRC) not in sys.path:
    sys.path.insert(0, str(POTENTIAL_CURVATURE_SRC))

from scripts.scalar_potential_manfit import estimate_gradient_from_neighbors  # noqa: E402
from potential_curvature import ThinPlatePotential  # noqa: E402


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
    parser.add_argument("--gradient-neighbors", type=int, default=42)
    parser.add_argument("--grid-neighbors", type=int, default=18)
    parser.add_argument("--dof", type=float, default=30.0)
    parser.add_argument("--control-points", type=int, default=900)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def read_palantir_h5ad(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read only the fields needed from the processed Palantir H5AD."""

    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError(
            "anndata is required to read the Palantir H5AD. Run this script with "
            "/Users/jh/Projects/potential_curvature/.venv-potential-curvature/bin/python."
        ) from exc

    data = ad.read_h5ad(path, backed="r")
    X_umap = np.asarray(data.obsm["X_umap"], dtype=float)
    entropy = data.obs["palantir_entropy"].to_numpy(dtype=float)
    pseudotime = data.obs["palantir_pseudotime"].to_numpy(dtype=float)
    return X_umap, entropy, pseudotime


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - np.mean(values)) / (np.std(values) + 1e-12)


def make_grid(X: np.ndarray, grid_size: int = 34, pad_fraction: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """Create a regular grid over X and keep points inside the UMAP hull."""

    pad_x = pad_fraction * np.ptp(X[:, 0])
    pad_y = pad_fraction * np.ptp(X[:, 1])
    gx = np.linspace(X[:, 0].min() - pad_x, X[:, 0].max() + pad_x, grid_size)
    gy = np.linspace(X[:, 1].min() - pad_y, X[:, 1].max() + pad_y, grid_size)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    inside = Delaunay(X).find_simplex(grid) >= 0
    return grid[inside], inside


def smooth_cell_gradients_to_grid(
    X: np.ndarray,
    cell_gradients: np.ndarray,
    grid: np.ndarray,
    n_neighbors: int = 18,
) -> np.ndarray:
    """Interpolate local cell gradients onto grid points with Gaussian weights."""

    n_neighbors = min(int(n_neighbors), X.shape[0])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    distances, indices = nbrs.kneighbors(grid)
    bandwidth = np.median(distances[:, -1]) + 1e-12
    weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    weights /= np.sum(weights, axis=1, keepdims=True) + 1e-12
    return np.einsum("gk,gkd->gd", weights, cell_gradients[indices])


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vectors, axis=1)
    return vectors / (norms[:, None] + eps), norms


def plot_gradient_panel(
    X_umap: np.ndarray,
    entropy: np.ndarray,
    grid: np.ndarray,
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
        (axes[0], raw_unit, raw_norm, "Before fitting: local entropy gradient"),
        (axes[1], fitted_unit, fitted_norm, "After fitting: TPS entropy gradient"),
    ]

    for ax, gradient_unit, gradient_norm, title in panel_specs:
        scatter = ax.scatter(
            X_umap[:, 0],
            X_umap[:, 1],
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
            gradient_norm,
            cmap="magma",
            angles="xy",
            scale_units="xy",
            scale=11,
            width=0.0045,
            headwidth=3.8,
            headlength=4.8,
            headaxislength=4.2,
            alpha=0.92,
        )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    fig.colorbar(scatter, ax=axes, fraction=0.035, pad=0.02, label="Palantir fate entropy")
    fig.suptitle("Palantir Bone Marrow: Entropy Gradient Field Before and After Fitting", y=1.03)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X_umap, entropy, pseudotime = read_palantir_h5ad(args.h5ad)
    entropy_z = zscore(entropy)

    raw_cell_gradient = estimate_gradient_from_neighbors(
        X_umap,
        entropy_z,
        n_neighbors=args.gradient_neighbors,
        ridge=5e-2,
    )

    model = ThinPlatePotential(
        X_umap,
        entropy_z,
        dof=args.dof,
        n_control_points=min(args.control_points, X_umap.shape[0]),
        random_state=args.random_state,
    ).fit()

    grid, _ = make_grid(X_umap, grid_size=args.grid_size)
    raw_grid_gradient = smooth_cell_gradients_to_grid(
        X_umap,
        raw_cell_gradient,
        grid,
        n_neighbors=args.grid_neighbors,
    )
    fitted_grid_gradient = model.gradient(grid)

    panel_path = args.output_dir / "palantir_entropy_gradient_before_after_manifold_fit.png"
    plot_gradient_panel(
        X_umap,
        entropy,
        grid,
        raw_grid_gradient,
        fitted_grid_gradient,
        panel_path,
    )

    vectors = pd.DataFrame(
        {
            "grid_umap_1": grid[:, 0],
            "grid_umap_2": grid[:, 1],
            "raw_grad_umap_1": raw_grid_gradient[:, 0],
            "raw_grad_umap_2": raw_grid_gradient[:, 1],
            "fitted_grad_umap_1": fitted_grid_gradient[:, 0],
            "fitted_grad_umap_2": fitted_grid_gradient[:, 1],
            "raw_grad_norm": np.linalg.norm(raw_grid_gradient, axis=1),
            "fitted_grad_norm": np.linalg.norm(fitted_grid_gradient, axis=1),
        }
    )
    vectors.to_csv(args.output_dir / "palantir_entropy_gradient_before_after_vectors.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "n_cells": X_umap.shape[0],
                "entropy_min": float(np.min(entropy)),
                "entropy_max": float(np.max(entropy)),
                "pseudotime_min": float(np.min(pseudotime)),
                "pseudotime_max": float(np.max(pseudotime)),
                "tps_dof": float(args.dof),
                "tps_control_points": int(min(args.control_points, X_umap.shape[0])),
                "n_grid_vectors": int(grid.shape[0]),
            }
        ]
    )
    summary.to_csv(args.output_dir / "palantir_entropy_gradient_before_after_summary.csv", index=False)

    print(panel_path)
    print(args.output_dir / "palantir_entropy_gradient_before_after_vectors.csv")
    print(args.output_dir / "palantir_entropy_gradient_before_after_summary.csv")


if __name__ == "__main__":
    main()
