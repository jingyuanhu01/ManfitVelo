"""Run synthetic geometry/velocity benchmark comparisons.

The default run is intentionally small enough for iteration. Increase
``--n-samples`` and add datasets with ``--datasets all`` for longer runs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.geometry_velocity_metrics import (  # noqa: E402
    average_local_spectrum,
    displacement_summary,
    mean_distance_to_clean_cloud,
    mean_l2_to_clean,
    median_distance_to_clean_cloud,
    metric_dict,
    normal_energy_ratio,
    quantile_distance_to_clean_cloud,
    rmse_to_clean,
    tangent_energy_ratio,
    local_effective_dimension,
    local_spectral_gap,
    velocity_neighbor_direction_agreement,
    velocity_smoothness,
    velocity_tangent_alignment,
)
from scripts.html_report_utils import write_html_report  # noqa: E402
from scripts.manfit_ours import manfit_ours  # noqa: E402
from scripts.pca_denoisers import (  # noqa: E402
    global_pca_denoise,
    global_pca_denoise_variance,
    local_pca_denoise,
    project_vectors_with_pca_info,
)
from scripts.velocity_manifold_fitter import VelocityManifoldFitter  # noqa: E402
from simulation.flat_manifold_potential_fields import (  # noqa: E402
    FlatPotentialFieldConfig,
    make_flat_manifold_potential_field,
)
from simulation.flat_manifold_vector_fields import (  # noqa: E402
    FlatVectorFieldConfig,
    make_flat_manifold_vector_field,
)
from simulation.manifold_velocity_flows import (  # noqa: E402
    ManifoldVelocityFlowConfig,
    make_manifold_velocity_flow,
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    simulation: dict[str, object]
    intrinsic_dim: int
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="flat_rotation,s_curve,flat_saddle")
    parser.add_argument("--n-seeds", "--n_seeds", dest="n_seeds", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=120)
    parser.add_argument("--position-noise", type=float, default=0.3)
    parser.add_argument("--velocity-noise", type=float, default=0.3)
    parser.add_argument("--extra-dims", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=25)
    parser.add_argument("--fit-neighbors", type=int, default=25)
    parser.add_argument("--fit-iterations", type=int, default=5)
    parser.add_argument("--eta-g", type=float, default=0.35)
    parser.add_argument("--theta", type=float, default=0.15)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--include-local-pca", action="store_true")
    parser.add_argument("--skip-position-manfit", dest="include_position_manfit", action="store_false")
    parser.set_defaults(include_position_manfit=True)
    parser.add_argument("--position-manfit-max-n", type=int, default=160)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "simulation_benchmark")
    return parser.parse_args()


def pad_to_dim(X: np.ndarray, dim: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.shape[1] > dim:
        raise ValueError("cannot pad to a smaller dimension")
    if X.shape[1] == dim:
        return X
    return np.hstack([X, np.zeros((X.shape[0], dim - X.shape[1]))])


def build_dataset_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    requested = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if requested == ["all"]:
        requested = [
            "flat_rotation",
            "flat_spiral",
            "flat_saddle",
            "s_curve",
            "swiss_roll",
            "half_sphere_rotation",
            "potential_saddle_surface_saddle",
        ]

    specs: list[DatasetSpec] = []
    for offset, key in enumerate(requested):
        seed = args.seed + 17 * offset
        if key == "flat_rotation":
            sim = make_flat_manifold_vector_field(
                FlatVectorFieldConfig(
                    field_name="rotation",
                    n_samples=args.n_samples,
                    position_noise=args.position_noise,
                    velocity_noise=args.velocity_noise,
                    extra_dims=args.extra_dims,
                    seed=seed,
                )
            )
            specs.append(DatasetSpec(key, "Flat rotation vector field", sim, 2, "Clean manifold is a 2D plane with tangent rotation."))
        elif key == "flat_spiral":
            sim = make_flat_manifold_vector_field(
                FlatVectorFieldConfig(
                    field_name="spiral",
                    n_samples=args.n_samples,
                    position_noise=args.position_noise,
                    velocity_noise=args.velocity_noise,
                    extra_dims=args.extra_dims,
                    seed=seed,
                )
            )
            specs.append(DatasetSpec(key, "Flat spiral vector field", sim, 2, "2D plane with expanding spiral velocities."))
        elif key == "flat_saddle":
            sim = make_flat_manifold_vector_field(
                FlatVectorFieldConfig(
                    field_name="saddle",
                    n_samples=args.n_samples,
                    position_noise=args.position_noise,
                    velocity_noise=args.velocity_noise,
                    extra_dims=args.extra_dims,
                    seed=seed,
                )
            )
            specs.append(DatasetSpec(key, "Flat saddle vector field", sim, 2, "2D plane with a saddle flow."))
        elif key == "s_curve":
            sim = make_manifold_velocity_flow(
                ManifoldVelocityFlowConfig(
                    manifold_name="s_curve",
                    field_name="velocity_flow",
                    n_samples=args.n_samples,
                    position_noise=args.position_noise,
                    velocity_noise=args.velocity_noise,
                    extra_dims=args.extra_dims,
                    seed=seed,
                )
            )
            specs.append(DatasetSpec(key, "S-curve velocity flow", sim, 2, "Curved 2D manifold embedded in noisy high-dimensional coordinates."))
        elif key == "swiss_roll":
            sim = make_manifold_velocity_flow(
                ManifoldVelocityFlowConfig(
                    manifold_name="swiss_roll",
                    field_name="velocity_flow",
                    n_samples=args.n_samples,
                    position_noise=args.position_noise,
                    velocity_noise=args.velocity_noise,
                    extra_dims=args.extra_dims,
                    seed=seed,
                )
            )
            specs.append(DatasetSpec(key, "Swiss roll velocity flow", sim, 2, "Nonlinear 2D manifold where global PCA is expected to be limited."))
        elif key == "half_sphere_rotation":
            sim = make_manifold_velocity_flow(
                ManifoldVelocityFlowConfig(
                    manifold_name="half_sphere",
                    field_name="rotation",
                    n_samples=args.n_samples,
                    position_noise=args.position_noise,
                    velocity_noise=args.velocity_noise,
                    extra_dims=args.extra_dims,
                    seed=seed,
                )
            )
            specs.append(DatasetSpec(key, "Half-sphere rotation field", sim, 2, "Curved sphere patch with tangent rotation."))
        elif key == "potential_saddle_surface_saddle":
            sim = make_flat_manifold_potential_field(
                FlatPotentialFieldConfig(
                    manifold_name="saddle_surface",
                    field_name="saddle",
                    n_samples=args.n_samples,
                    position_noise=args.position_noise,
                    potential_noise=args.velocity_noise,
                    extra_dims=args.extra_dims,
                    seed=seed,
                )
            )
            specs.append(DatasetSpec(key, "Saddle-surface scalar potential", sim, 2, "Experimental potential-field case; V is generated from the clean gradient flow."))
        else:
            raise ValueError(f"unknown dataset key: {key}")
    return specs


def pca_velocity(V: np.ndarray, info: dict[str, object]) -> np.ndarray:
    return project_vectors_with_pca_info(V, info)


def run_methods(spec: DatasetSpec, args: argparse.Namespace) -> list[dict[str, object]]:
    sim = spec.simulation
    X = np.asarray(sim["X"], dtype=float)
    V = np.asarray(sim["V"], dtype=float)
    rows: list[dict[str, object]] = []

    def add_success(method: str, X_hat: np.ndarray, V_hat: np.ndarray, info: dict[str, object] | None = None):
        rows.append(
            {
                "dataset": spec.key,
                "dataset_label": spec.label,
                "method": method,
                "status": "ok",
                "error": "",
                "X": X_hat,
                "V": V_hat,
                "info": info or {},
            }
        )

    def add_failure(method: str, exc: BaseException):
        rows.append(
            {
                "dataset": spec.key,
                "dataset_label": spec.label,
                "method": method,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=3),
                "X": None,
                "V": None,
                "info": {},
            }
        )

    add_success("raw_noisy", X.copy(), V.copy(), {"baseline": "observed noisy positions and velocities"})

    rank_methods = [
        ("pca_rank_d", spec.intrinsic_dim),
        ("pca_rank_2d", 2 * spec.intrinsic_dim),
        ("pca_rank_5d", 5 * spec.intrinsic_dim),
    ]
    for name, rank in rank_methods:
        try:
            X_hat, info = global_pca_denoise(X, rank=rank, return_info=True)
            add_success(name, X_hat, pca_velocity(V, info), info)
        except Exception as exc:  # noqa: BLE001
            add_failure(name, exc)

    for threshold in (0.9, 0.95):
        name = f"pca_{int(threshold * 100)}pct"
        try:
            X_hat, info = global_pca_denoise_variance(X, variance_threshold=threshold, return_info=True)
            add_success(name, X_hat, pca_velocity(V, info), info)
        except Exception as exc:  # noqa: BLE001
            add_failure(name, exc)

    if args.include_local_pca:
        try:
            X_hat, info = local_pca_denoise(X, spec.intrinsic_dim, n_neighbors=args.n_neighbors, return_info=True)
            add_success("local_pca_position", X_hat, V.copy(), info)
        except Exception as exc:  # noqa: BLE001
            add_failure("local_pca_position", exc)

    if args.include_position_manfit:
        try:
            if X.shape[0] > args.position_manfit_max_n:
                raise RuntimeError(
                    f"position-only MANFIT skipped because n={X.shape[0]} exceeds "
                    f"--position-manfit-max-n={args.position_manfit_max_n}"
                )
            X_hat = manfit_ours(X, sig=max(float(args.position_noise), 1e-3), sample_init=X)
            add_success("position_only_manfit", X_hat, V.copy(), {"baseline": "scripts.manfit_ours"})
        except Exception as exc:  # noqa: BLE001
            add_failure("position_only_manfit", exc)

    try:
        fitter = VelocityManifoldFitter(
            X,
            V,
            d_mode="global",
            global_d=spec.intrinsic_dim,
            k=args.fit_neighbors,
            T=args.fit_iterations,
            eta_g=args.eta_g,
            theta=args.theta,
            kappa=args.kappa,
            bandwidth_mode="variable",
            use_PCA=False,
            random_state=args.seed,
        )
        result = fitter.fit(update_mode="normal_only", return_dict=True)
        add_success(
            "velocity_manifold_fitter",
            result["X"],
            result["V"],
            {
                "d_mode": "global",
                "global_d": spec.intrinsic_dim,
                "fit_history": result.get("history", []),
                "mean_local_dim": float(np.mean(result.get("local_dims", [spec.intrinsic_dim]))),
            },
        )
    except Exception as exc:  # noqa: BLE001
        add_failure("velocity_manifold_fitter", exc)

    return rows


def compute_metrics(spec: DatasetSpec, method_row: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    base = {
        "dataset": spec.key,
        "method": method_row["method"],
        "status": method_row["status"],
        "error": method_row.get("error", ""),
    }
    if method_row["status"] != "ok":
        return base

    X_hat = np.asarray(method_row["X"], dtype=float)
    V_hat = np.asarray(method_row["V"], dtype=float)
    X_raw = np.asarray(spec.simulation["X"], dtype=float)
    X_clean = pad_to_dim(np.asarray(spec.simulation["X_gt"], dtype=float), X_hat.shape[1])

    base.update(
        {
            "rmse_to_clean": rmse_to_clean(X_hat, X_clean),
            "mean_l2_to_clean": mean_l2_to_clean(X_hat, X_clean),
            "mean_distance_to_clean_cloud_approx": mean_distance_to_clean_cloud(X_hat, X_clean),
            "median_distance_to_clean_cloud_approx": median_distance_to_clean_cloud(X_hat, X_clean),
            "q90_distance_to_clean_cloud_approx": quantile_distance_to_clean_cloud(X_hat, X_clean, q=0.9),
        }
    )
    base.update(metric_dict("normal_energy_ratio", normal_energy_ratio(X_hat, spec.intrinsic_dim, args.n_neighbors)))
    base.update(metric_dict("tangent_energy_ratio", tangent_energy_ratio(X_hat, spec.intrinsic_dim, args.n_neighbors)))
    base.update(metric_dict("local_spectral_gap", local_spectral_gap(X_hat, spec.intrinsic_dim, args.n_neighbors)))
    base.update(metric_dict("local_effective_dimension", local_effective_dimension(X_hat, args.n_neighbors)))
    base.update(metric_dict("velocity_tangent_alignment", velocity_tangent_alignment(X_hat, V_hat, spec.intrinsic_dim, args.n_neighbors)))
    base.update(metric_dict("velocity_neighbor_direction_agreement", velocity_neighbor_direction_agreement(X_hat, V_hat, args.n_neighbors)))
    base.update(metric_dict("velocity_smoothness", velocity_smoothness(X_hat, V_hat, args.n_neighbors)))
    base.update(metric_dict("displacement", displacement_summary(X_raw, X_hat, args.n_neighbors)))
    return base


def display_basis(*arrays: np.ndarray) -> PCA:
    joined = np.vstack([np.asarray(array, dtype=float) for array in arrays])
    return PCA(n_components=2, random_state=0).fit(joined)


def plot_dataset_overview(spec: DatasetSpec, method_rows: list[dict[str, object]], assets_dir: Path) -> Path:
    raw = next(row for row in method_rows if row["method"] == "raw_noisy" and row["status"] == "ok")
    clean = pad_to_dim(np.asarray(spec.simulation["X_gt"], dtype=float), np.asarray(raw["X"]).shape[1])
    pca_row = next((row for row in method_rows if row["method"] == "pca_rank_d" and row["status"] == "ok"), None)
    vmf_row = next((row for row in method_rows if row["method"] == "velocity_manifold_fitter" and row["status"] == "ok"), None)
    arrays = [clean, raw["X"]]
    if pca_row is not None:
        arrays.append(pca_row["X"])
    if vmf_row is not None:
        arrays.append(vmf_row["X"])
    pca = display_basis(*arrays)

    panels = [("clean", clean, None), ("raw noisy", raw["X"], raw["V"])]
    if pca_row is not None:
        panels.append(("PCA rank d", pca_row["X"], pca_row["V"]))
    if vmf_row is not None:
        panels.append(("VelocityManifoldFitter", vmf_row["X"], vmf_row["V"]))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 3.8), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    color = np.asarray(spec.simulation.get("true_time", np.arange(clean.shape[0])), dtype=float)
    for ax, (title, X_panel, V_panel) in zip(axes, panels):
        Z = pca.transform(np.asarray(X_panel, dtype=float))
        ax.scatter(Z[:, 0], Z[:, 1], c=color, s=13, cmap="viridis", linewidths=0, alpha=0.85)
        if V_panel is not None:
            V2 = np.asarray(V_panel) @ pca.components_.T
            step = max(1, Z.shape[0] // 35)
            scale = np.percentile(np.linalg.norm(V2, axis=1), 90) + 1e-12
            span = max(np.ptp(Z[:, 0]), np.ptp(Z[:, 1]), 1e-12)
            ax.quiver(
                Z[::step, 0],
                Z[::step, 1],
                V2[::step, 0] * span * 0.05 / scale,
                V2[::step, 1] * span * 0.05 / scale,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.003,
                color="#2f3a4a",
                alpha=0.65,
            )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    path = assets_dir / f"{spec.key}_overview.png"
    fig.suptitle(spec.label)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_movement(spec: DatasetSpec, method_rows: list[dict[str, object]], assets_dir: Path) -> Path | None:
    raw = next(row for row in method_rows if row["method"] == "raw_noisy" and row["status"] == "ok")
    vmf_row = next((row for row in method_rows if row["method"] == "velocity_manifold_fitter" and row["status"] == "ok"), None)
    if vmf_row is None:
        return None
    pca = display_basis(raw["X"], vmf_row["X"])
    Z0 = pca.transform(raw["X"])
    Z1 = pca.transform(vmf_row["X"])
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    step = max(1, Z0.shape[0] // 140)
    for i in range(0, Z0.shape[0], step):
        ax.plot([Z0[i, 0], Z1[i, 0]], [Z0[i, 1], Z1[i, 1]], color="#8a99ad", linewidth=0.6, alpha=0.55)
    color = np.asarray(spec.simulation.get("true_time", np.arange(Z0.shape[0])), dtype=float)
    ax.scatter(Z0[:, 0], Z0[:, 1], c=color, s=12, cmap="viridis", alpha=0.55, label="raw")
    ax.scatter(Z1[:, 0], Z1[:, 1], c=color, s=13, cmap="viridis", alpha=0.9, marker="x", label="VMF")
    ax.set_title(f"{spec.label}: raw to velocity-aware fit")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False)
    path = assets_dir / f"{spec.key}_movement.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_metric_bars(spec: DatasetSpec, metrics_df: pd.DataFrame, assets_dir: Path) -> Path:
    subset = metrics_df[(metrics_df["dataset"] == spec.key) & (metrics_df["status"] == "ok")].copy()
    metrics = ["rmse_to_clean", "normal_energy_ratio_mean", "velocity_tangent_alignment_mean"]
    if "seed" in subset.columns:
        subset = subset.groupby("method", as_index=False)[metrics].mean(numeric_only=True)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 4.0), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        plot_df = subset[["method", metric]].dropna().sort_values(metric)
        ax.barh(plot_df["method"], plot_df[metric], color="#4f7cac")
        ax.set_title(metric)
        ax.tick_params(axis="y", labelsize=8)
    path = assets_dir / f"{spec.key}_metric_bars.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def warnings_for_dataset(spec: DatasetSpec, metrics_df: pd.DataFrame) -> list[str]:
    rows = metrics_df[metrics_df["dataset"] == spec.key]
    warnings_out: list[str] = []
    failed = rows[rows["status"] != "ok"]
    for _, row in failed.iterrows():
        warnings_out.append(f"{row['method']} failed or was skipped: {row.get('error', '')}")

    raw = rows[(rows["method"] == "raw_noisy") & (rows["status"] == "ok")]
    vmf = rows[(rows["method"] == "velocity_manifold_fitter") & (rows["status"] == "ok")]
    if not raw.empty and not vmf.empty:
        raw_rmse = float(raw.iloc[0]["rmse_to_clean"])
        vmf_rmse = float(vmf.iloc[0]["rmse_to_clean"])
        if vmf_rmse > raw_rmse:
            warnings_out.append(f"VelocityManifoldFitter RMSE ({vmf_rmse:.4g}) is worse than raw noisy RMSE ({raw_rmse:.4g}).")
        raw_align = float(raw.iloc[0]["velocity_tangent_alignment_mean"])
        vmf_align = float(vmf.iloc[0]["velocity_tangent_alignment_mean"])
        if vmf_align + 1e-6 < raw_align:
            warnings_out.append(
                f"VelocityManifoldFitter velocity-tangent alignment ({vmf_align:.4g}) is lower than raw ({raw_align:.4g})."
            )
    return warnings_out


def interpretation_for_dataset(spec: DatasetSpec, metrics_df: pd.DataFrame) -> str:
    subset = metrics_df[(metrics_df["dataset"] == spec.key) & (metrics_df["status"] == "ok")]
    if subset.empty:
        return "No successful method rows were available for interpretation."
    best_rmse = subset.sort_values("rmse_to_clean").iloc[0]
    best_alignment = subset.sort_values("velocity_tangent_alignment_mean", ascending=False).iloc[0]
    best_normal = subset.sort_values("normal_energy_ratio_mean").iloc[0]
    return (
        f"Best clean-RMSE method: {best_rmse['method']} ({best_rmse['rmse_to_clean']:.4g}). "
        f"Best velocity-tangent alignment: {best_alignment['method']} "
        f"({best_alignment['velocity_tangent_alignment_mean']:.4g}). "
        f"Lowest normal energy ratio: {best_normal['method']} ({best_normal['normal_energy_ratio_mean']:.4g}). "
        "Approximate clean-cloud distances use nearest neighbors in the clean sampled cloud, not an analytic projection."
    )


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["dataset", "method"]
    numeric_cols = [
        col
        for col in metrics_df.columns
        if col not in {"seed", "dataset", "method", "status", "error"} and pd.api.types.is_numeric_dtype(metrics_df[col])
    ]
    ok = metrics_df[metrics_df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame(columns=id_cols)
    summary = ok.groupby(id_cols)[numeric_cols].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    failures = (
        metrics_df[metrics_df["status"] != "ok"]
        .groupby(id_cols)
        .size()
        .rename("failure_count")
        .reset_index()
    )
    summary = summary.merge(failures, on=id_cols, how="left")
    summary["failure_count"] = summary["failure_count"].fillna(0).astype(int)
    return summary


def _best_summary(summary_df: pd.DataFrame, dataset: str, metric: str, ascending: bool = True) -> pd.Series | None:
    metric_col = f"{metric}_mean"
    subset = summary_df[(summary_df["dataset"] == dataset) & summary_df[metric_col].notna()]
    if subset.empty:
        return None
    return subset.sort_values(metric_col, ascending=ascending).iloc[0]


def _summary_value(summary_df: pd.DataFrame, dataset: str, method: str, metric: str) -> float | None:
    metric_col = f"{metric}_mean"
    subset = summary_df[(summary_df["dataset"] == dataset) & (summary_df["method"] == method)]
    if subset.empty or metric_col not in subset:
        return None
    value = subset.iloc[0][metric_col]
    return None if pd.isna(value) else float(value)


def dataset_result_paragraph(spec: DatasetSpec, summary_df: pd.DataFrame) -> str:
    best_rmse = _best_summary(summary_df, spec.key, "rmse_to_clean", ascending=True)
    best_cloud = _best_summary(summary_df, spec.key, "mean_distance_to_clean_cloud_approx", ascending=True)
    best_align = _best_summary(summary_df, spec.key, "velocity_tangent_alignment_mean", ascending=False)
    best_normal = _best_summary(summary_df, spec.key, "normal_energy_ratio_mean", ascending=True)
    if best_rmse is None:
        return f"{spec.label}: no successful method rows were available."

    pca_rmse = _summary_value(summary_df, spec.key, "pca_rank_d", "rmse_to_clean")
    vmf_rmse = _summary_value(summary_df, spec.key, "velocity_manifold_fitter", "rmse_to_clean")
    vmf_disp = _summary_value(summary_df, spec.key, "velocity_manifold_fitter", "displacement_relative_to_local_scale_mean")
    pieces = [
        f"{spec.label}: best RMSE was {best_rmse['method']} ({best_rmse['rmse_to_clean_mean']:.4g}); "
        f"best approximate clean-cloud distance was {best_cloud['method']} "
        f"({best_cloud['mean_distance_to_clean_cloud_approx_mean']:.4g}); "
        f"best velocity-tangent alignment was {best_align['method']} "
        f"({best_align['velocity_tangent_alignment_mean_mean']:.4g})."
    ]
    if spec.key in {"flat_rotation", "flat_saddle"} and best_rmse["method"] == "pca_rank_d":
        pieces.append(
            "This is expected because the true synthetic manifold is globally linear, making PCA-rank-d close to an oracle baseline."
        )
    if spec.key in {"s_curve", "swiss_roll"} and pca_rmse is not None and vmf_rmse is not None:
        if vmf_rmse < pca_rmse:
            pieces.append(
                f"VMF shows a modest RMSE improvement over PCA-rank-d ({vmf_rmse:.4g} vs {pca_rmse:.4g}), "
                "but the effect size should be checked across more seeds and parameter settings."
            )
        else:
            pieces.append(
                f"In this setting, VMF does not clearly outperform PCA-rank-d on RMSE ({vmf_rmse:.4g} vs {pca_rmse:.4g}) "
                "under the current parameter choices."
            )
    if best_normal is not None and best_normal["method"] == "pca_rank_d" and spec.key not in {"flat_rotation", "flat_saddle"}:
        pieces.append(
            "PCA-rank-d has the lowest normal energy, but this should not be interpreted as true nonlinear manifold recovery, "
            "because PCA-rank-d can reduce normal energy by collapsing data into a global linear rank-d subspace."
        )
    if vmf_disp is not None and vmf_disp > 0.5:
        pieces.append(f"VMF movement cost is nontrivial at {vmf_disp:.3g} local scales on average.")
    return " ".join(pieces)


def cross_dataset_interpretation(specs: list[DatasetSpec], summary_df: pd.DataFrame, metrics_df: pd.DataFrame) -> list[str]:
    paragraphs = [dataset_result_paragraph(spec, summary_df) for spec in specs]
    unresolved: list[str] = []
    failed = metrics_df[metrics_df["status"] != "ok"]
    if failed.empty:
        unresolved.append("No method failures were recorded in this run.")
    else:
        for _, row in failed[["dataset", "method", "error"]].drop_duplicates().iterrows():
            unresolved.append(f"{row['dataset']} / {row['method']}: {row['error']}")
    paragraphs.append(
        "Cross-dataset summary: PCA-rank-d is strongest on globally linear flat simulations, while VMF should be judged mainly on nonlinear geometry and velocity-utility metrics. Low-dimensionality metrics alone are not sufficient evidence because global PCA can win them by construction."
    )
    paragraphs.append(
        "Warning: if PCA-rank-d has the best normal energy on nonlinear datasets, that can reflect low-rank collapse into a global linear subspace rather than recovery of the curved manifold."
    )
    paragraphs.append("Unresolved issues: " + " ".join(unresolved))
    return paragraphs


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    if int(args.n_seeds) < 1:
        raise ValueError("--n-seeds must be at least 1")

    base_seed = int(args.seed)
    specs = build_dataset_specs(args)
    method_rows_by_dataset: dict[str, list[dict[str, object]]] = {}
    metric_rows: list[dict[str, object]] = []
    plot_specs_by_key: dict[str, DatasetSpec] = {}
    for seed_index in range(int(args.n_seeds)):
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = base_seed + seed_index
        seed_specs = build_dataset_specs(seed_args)
        if seed_index == 0:
            specs = seed_specs
            plot_specs_by_key = {spec.key: spec for spec in seed_specs}
        for spec in seed_specs:
            method_rows = run_methods(spec, seed_args)
            if seed_index == 0:
                method_rows_by_dataset[spec.key] = method_rows
            for row in method_rows:
                metric_row = compute_metrics(spec, row, seed_args)
                metric_row["seed"] = seed_args.seed
                metric_rows.append(metric_row)

    metrics_df = pd.DataFrame(metric_rows)
    summary_df = summarize_metrics(metrics_df)
    metrics_csv = output_dir / "simulation_results_long.csv"
    summary_csv = output_dir / "simulation_results_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    metrics_df.to_csv(output_dir / "simulation_metrics.csv", index=False)

    config = vars(args).copy()
    config["output_dir"] = str(config["output_dir"])
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    sections: list[dict[str, object]] = [
        {
            "heading": "Run Configuration",
            "text": [
                f"Generated at {datetime.now().isoformat(timespec='seconds')}.",
                "This report compares raw noisy data, global PCA denoising baselines, optional position-only MANFIT, and VelocityManifoldFitter.",
            ],
            "tables": [pd.DataFrame([config])],
        },
        {
            "heading": "Metric Definitions",
            "text": [
                "RMSE and mean L2 use the padded clean synthetic coordinates. Clean-cloud distances are nearest-neighbor approximations.",
                "Normal energy ratio is the local covariance energy outside the intrinsic dimension; lower is better unless the method collapses.",
                "Velocity-tangent alignment measures how much velocity lies in the locally estimated tangent space; higher is better.",
            ],
        },
        {
            "heading": "All Method Metrics",
            "tables": [
                {"caption": "Summary across seeds", "data": summary_df},
                {"caption": "Long-form per-seed metrics", "data": metrics_df},
            ],
        },
        {
            "heading": "Result-based interpretation",
            "text": cross_dataset_interpretation(specs, summary_df, metrics_df),
        },
    ]

    for spec in specs:
        plot_spec = plot_specs_by_key.get(spec.key, spec)
        overview = plot_dataset_overview(plot_spec, method_rows_by_dataset[spec.key], assets_dir)
        movement = plot_movement(plot_spec, method_rows_by_dataset[spec.key], assets_dir)
        bars = plot_metric_bars(spec, metrics_df, assets_dir)
        images = [
            {"path": overview, "caption": "Clean, noisy, PCA, and velocity-aware positions"},
            {"path": bars, "caption": "Key quantitative metrics"},
        ]
        if movement is not None:
            images.insert(1, {"path": movement, "caption": "Raw-to-VMF movement paths"})
        sections.append(
            {
                "heading": spec.label,
                "text": [spec.note, interpretation_for_dataset(spec, metrics_df)],
                "warnings": warnings_for_dataset(spec, metrics_df),
                "tables": [
                    metrics_df[metrics_df["dataset"] == spec.key].drop(columns=["dataset"], errors="ignore")
                ],
                "images": images,
            }
        )

    report_path = write_html_report("Simulation Benchmark: Geometry and Velocity Utility", sections, output_dir / "index.html")
    return {"report_path": report_path, "metrics_csv": metrics_csv, "summary_csv": summary_csv, "metrics": metrics_df, "summary": summary_df}


def main() -> None:
    result = run_benchmark(parse_args())
    print(result["report_path"])
    print(result["metrics_csv"])
    print(result["summary_csv"])


if __name__ == "__main__":
    main()
