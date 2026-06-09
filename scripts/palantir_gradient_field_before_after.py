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
from scipy.interpolate import SmoothBivariateSpline
from scipy.spatial import Delaunay

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
    parser.add_argument("--max-arrows", type=int, default=850)
    parser.add_argument("--potential-grid-size", type=int, default=85)
    parser.add_argument("--spline-smoothing", type=float, default=70.0)
    parser.add_argument("--pca-dims", type=int, default=30)
    parser.add_argument("--gradient-neighbors", type=int, default=42)
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


def choose_arrow_cells(n_cells: int, max_arrows: int, random_state: int) -> np.ndarray:
    """Choose actual cell indices for quiver arrows without changing point positions."""

    n_cells = int(n_cells)
    max_arrows = min(max(int(max_arrows), 1), n_cells)
    if max_arrows == n_cells:
        return np.arange(n_cells)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_cells, size=max_arrows, replace=False))


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vectors, axis=1)
    return vectors / (norms[:, None] + eps), norms


def plot_gradient_panel(
    X_raw_display: np.ndarray,
    X_fit_display: np.ndarray,
    entropy: np.ndarray,
    arrow_indices: np.ndarray,
    raw_cell_gradient: np.ndarray,
    fitted_cell_gradient: np.ndarray,
    output_path: Path,
) -> None:
    """Write a before/after gradient-field panel."""

    raw_arrow_gradient = raw_cell_gradient[arrow_indices]
    fitted_arrow_gradient = fitted_cell_gradient[arrow_indices]
    raw_unit, _ = normalize_vectors(raw_arrow_gradient)
    fitted_unit, _ = normalize_vectors(fitted_arrow_gradient)
    vmax = np.nanpercentile(entropy, 99)
    vmin = np.nanpercentile(entropy, 1)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    panel_specs = [
        (axes[0], X_raw_display, raw_unit, "Before fitting: PCA local entropy gradient"),
        (axes[1], X_fit_display, fitted_unit, "After fitting: PCA MANFIT entropy gradient"),
    ]

    for ax, X_display, gradient_unit, title in panel_specs:
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
            X_display[arrow_indices, 0],
            X_display[arrow_indices, 1],
            gradient_unit[:, 0],
            gradient_unit[:, 1],
            color="#111827",
            angles="xy",
            scale_units="xy",
            scale=1.9,
            width=0.0029,
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


def fit_spline_potential_surface(
    X_display: np.ndarray,
    entropy: np.ndarray,
    *,
    grid_size: int,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a smooth bivariate spline potential on PC1/PC2 and evaluate it on a hull-masked grid."""

    x = X_display[:, 0]
    y = X_display[:, 1]
    spline = SmoothBivariateSpline(x, y, entropy, kx=3, ky=3, s=float(smoothing))
    gx = np.linspace(np.min(x), np.max(x), int(grid_size))
    gy = np.linspace(np.min(y), np.max(y), int(grid_size))
    xx, yy = np.meshgrid(gx, gy)
    zz = spline.ev(xx.ravel(), yy.ravel()).reshape(xx.shape)
    zz = np.clip(zz, np.nanmin(entropy), np.nanmax(entropy))
    inside = Delaunay(X_display).find_simplex(np.column_stack([xx.ravel(), yy.ravel()])) >= 0
    zz = np.where(inside.reshape(xx.shape), zz, np.nan)
    return xx, yy, zz


def plot_potential_3d_panel(
    X_raw_display: np.ndarray,
    X_fit_display: np.ndarray,
    entropy: np.ndarray,
    output_path: Path,
    *,
    grid_size: int,
    smoothing: float,
) -> None:
    """Write a side-by-side 3D spline entropy potential landscape."""

    vmax = np.nanpercentile(entropy, 99)
    vmin = np.nanpercentile(entropy, 1)
    z_min = float(np.nanmin(entropy))
    z_max = float(np.nanmax(entropy))

    fig = plt.figure(figsize=(14.0, 6.2), constrained_layout=True)
    panel_specs = [
        (fig.add_subplot(1, 2, 1, projection="3d"), X_raw_display, "Before fitting: PCA spline potential"),
        (fig.add_subplot(1, 2, 2, projection="3d"), X_fit_display, "After fitting: PCA MANFIT spline potential"),
    ]

    mappable = None
    for ax, X_display, title in panel_specs:
        xx, yy, zz = fit_spline_potential_surface(
            X_display,
            entropy,
            grid_size=grid_size,
            smoothing=smoothing,
        )
        surface = ax.plot_surface(
            xx,
            yy,
            zz,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            linewidth=0.0,
            antialiased=True,
            alpha=0.84,
        )
        ax.scatter(
            X_display[:, 0],
            X_display[:, 1],
            entropy,
            c=entropy,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=2.0,
            alpha=0.32,
            depthshade=False,
        )
        mappable = surface
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("Palantir fate entropy")
        ax.set_zlim(z_min, z_max)
        ax.view_init(elev=28, azim=-62)
        ax.grid(False)
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)

    if mappable is not None:
        fig.colorbar(
            mappable,
            ax=[spec[0] for spec in panel_specs],
            fraction=0.035,
            pad=0.02,
            label="Spline entropy potential",
        )
    fig.suptitle("Palantir Bone Marrow: PCA Spline Potential Field Before and After Manifold Fitting", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
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
    arrow_indices = choose_arrow_cells(
        X_pca.shape[0],
        max_arrows=args.max_arrows,
        random_state=args.random_state,
    )

    panel_path = args.output_dir / "palantir_entropy_gradient_pca_before_after_manifold_fit.png"
    plot_gradient_panel(
        raw_display,
        fitted_display,
        entropy,
        arrow_indices,
        raw_cell_gradient[:, :2],
        fitted_cell_gradient[:, :2],
        panel_path,
    )

    potential_3d_path = args.output_dir / "palantir_entropy_potential_pca_3d_before_after_manifold_fit.png"
    plot_potential_3d_panel(
        raw_display,
        fitted_display,
        entropy,
        potential_3d_path,
        grid_size=args.potential_grid_size,
        smoothing=args.spline_smoothing,
    )

    raw_vectors = pd.DataFrame(
        {
            "panel": "before_pca_local_gradient",
            "cell_index": arrow_indices,
            "pc_1": raw_display[arrow_indices, 0],
            "pc_2": raw_display[arrow_indices, 1],
            "grad_pc_1": raw_cell_gradient[arrow_indices, 0],
            "grad_pc_2": raw_cell_gradient[arrow_indices, 1],
            "grad_norm": np.linalg.norm(raw_cell_gradient[arrow_indices, :2], axis=1),
        }
    )
    fitted_vectors = pd.DataFrame(
        {
            "panel": "after_pca_manfit_gradient",
            "cell_index": arrow_indices,
            "pc_1": fitted_display[arrow_indices, 0],
            "pc_2": fitted_display[arrow_indices, 1],
            "grad_pc_1": fitted_cell_gradient[arrow_indices, 0],
            "grad_pc_2": fitted_cell_gradient[arrow_indices, 1],
            "grad_norm": np.linalg.norm(fitted_cell_gradient[arrow_indices, :2], axis=1),
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
                "quiver_arrows": int(arrow_indices.shape[0]),
                "potential_grid_size": int(args.potential_grid_size),
                "spline_smoothing": float(args.spline_smoothing),
                "mean_fit_confidence": float(np.mean(fitted["confidence"])),
            }
        ]
    )
    summary.to_csv(args.output_dir / "palantir_entropy_gradient_pca_before_after_summary.csv", index=False)

    print(panel_path)
    print(potential_3d_path)
    print(args.output_dir / "palantir_entropy_gradient_pca_before_after_vectors.csv")
    print(args.output_dir / "palantir_entropy_gradient_pca_before_after_summary.csv")


if __name__ == "__main__":
    main()
