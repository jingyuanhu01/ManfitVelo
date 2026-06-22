"""Run real-data geometry and velocity utility report.

The default report uses the committed cell-cycle matrices. External datasets
are only included when their expected paths exist.
"""

from __future__ import annotations

import argparse
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
    estimate_local_projectors,
    knn_overlap,
    local_effective_dimension,
    local_spectral_gap,
    metric_dict,
    normal_energy_ratio,
    pairwise_distance_correlation,
    tangent_energy_ratio,
    trustworthiness_score,
    velocity_neighbor_direction_agreement,
    velocity_smoothness,
    velocity_tangent_alignment,
)
from scripts.html_report_utils import write_html_report  # noqa: E402
from scripts.pca_denoisers import (  # noqa: E402
    global_pca_denoise,
    global_pca_denoise_variance,
    local_pca_denoise,
    project_vectors_with_pca_info,
)
from scripts.velocity_manifold_fitter import VelocityManifoldFitter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cell_cycle")
    parser.add_argument("--representation-dim", type=int, default=30)
    parser.add_argument("--intrinsic-dim", type=int, default=2)
    parser.add_argument("--max-cells", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-neighbors", type=int, default=25)
    parser.add_argument("--fit-neighbors", type=int, default=25)
    parser.add_argument("--fit-iterations", type=int, default=5)
    parser.add_argument("--eta-g", type=float, default=0.35)
    parser.add_argument("--theta", type=float, default=0.15)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--no-local-pca", dest="include_local_pca", action="store_false")
    parser.set_defaults(include_local_pca=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "application_geometry")
    return parser.parse_args()


def load_cell_cycle(max_cells: int, seed: int) -> dict[str, object]:
    data_dir = ROOT / "data" / "cell_cycle"
    X = np.load(data_dir / "X_cc.npy").astype(float)
    V = np.load(data_dir / "V_cc.npy").astype(float)
    color_path = data_dir / "color_cell_cycle_relativePos.npy"
    color = np.load(color_path).astype(float) if color_path.exists() else np.arange(X.shape[0])
    selected = np.arange(X.shape[0])
    if max_cells and X.shape[0] > int(max_cells):
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(X.shape[0], size=int(max_cells), replace=False))
        X = X[selected]
        V = V[selected]
        color = color[selected]
    return {
        "key": "cell_cycle",
        "label": "Cell-cycle RNA velocity",
        "X": X,
        "V": V,
        "color": color,
        "selected_cells": selected,
        "source": str(data_dir),
    }


def pca_representation(X: np.ndarray, V: np.ndarray, n_components: int, seed: int) -> tuple[np.ndarray, np.ndarray, PCA]:
    n_components = min(int(n_components), X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    X_repr = pca.fit_transform(X)
    V_repr = V @ pca.components_.T
    return X_repr, V_repr, pca


def run_methods(dataset: dict[str, object], args: argparse.Namespace) -> list[dict[str, object]]:
    X_raw, V_raw, pca = pca_representation(dataset["X"], dataset["V"], args.representation_dim, args.seed)
    rows: list[dict[str, object]] = []

    def add_success(method: str, X_hat: np.ndarray, V_hat: np.ndarray, info: dict[str, object] | None = None):
        rows.append({"method": method, "status": "ok", "error": "", "X": X_hat, "V": V_hat, "info": info or {}})

    def add_failure(method: str, exc: BaseException):
        rows.append(
            {
                "method": method,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=3),
                "X": None,
                "V": None,
                "info": {},
            }
        )

    add_success(
        "raw_pca_representation",
        X_raw,
        V_raw,
        {"representation_dim": X_raw.shape[1], "source_pca_explained_variance": pca.explained_variance_ratio_},
    )

    for rank in (10, 20, 30):
        method = f"pca_rank_{rank}"
        try:
            if rank > min(X_raw.shape):
                raise ValueError(f"rank {rank} exceeds representation rank {min(X_raw.shape)}")
            X_hat, info = global_pca_denoise(X_raw, rank=rank, return_info=True)
            add_success(method, X_hat, project_vectors_with_pca_info(V_raw, info), info)
        except Exception as exc:  # noqa: BLE001
            add_failure(method, exc)

    for threshold in (0.9, 0.95):
        method = f"pca_{int(threshold * 100)}pct"
        try:
            X_hat, info = global_pca_denoise_variance(X_raw, variance_threshold=threshold, return_info=True)
            add_success(method, X_hat, project_vectors_with_pca_info(V_raw, info), info)
        except Exception as exc:  # noqa: BLE001
            add_failure(method, exc)

    if args.include_local_pca:
        try:
            X_hat, info = local_pca_denoise(
                X_raw,
                intrinsic_dim=args.intrinsic_dim,
                n_neighbors=args.n_neighbors,
                return_info=True,
            )
            add_success("local_pca_position", X_hat, V_raw.copy(), info)
        except Exception as exc:  # noqa: BLE001
            add_failure("local_pca_position", exc)

    try:
        fitter = VelocityManifoldFitter(
            X_raw,
            V_raw,
            d_mode="adaptive",
            adaptive_variance_threshold=0.85,
            adaptive_d_min=args.intrinsic_dim,
            adaptive_d_max=min(max(2 * args.intrinsic_dim, args.intrinsic_dim), X_raw.shape[1]),
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
                "d_mode": "adaptive",
                "mean_local_dim": float(np.mean(result.get("local_dims", [args.intrinsic_dim]))),
                "fit_history": result.get("history", []),
            },
        )
    except Exception as exc:  # noqa: BLE001
        add_failure("velocity_manifold_fitter", exc)

    return rows


def compute_metrics(method_rows: list[dict[str, object]], args: argparse.Namespace) -> pd.DataFrame:
    raw = next(row for row in method_rows if row["method"] == "raw_pca_representation" and row["status"] == "ok")
    X_raw = np.asarray(raw["X"], dtype=float)
    rows: list[dict[str, object]] = []
    for row in method_rows:
        record = {"method": row["method"], "status": row["status"], "error": row.get("error", "")}
        if row["status"] != "ok":
            rows.append(record)
            continue
        X = np.asarray(row["X"], dtype=float)
        V = np.asarray(row["V"], dtype=float)
        record.update(metric_dict("normal_energy_ratio", normal_energy_ratio(X, args.intrinsic_dim, args.n_neighbors)))
        record.update(metric_dict("tangent_energy_ratio", tangent_energy_ratio(X, args.intrinsic_dim, args.n_neighbors)))
        record.update(metric_dict("local_spectral_gap", local_spectral_gap(X, args.intrinsic_dim, args.n_neighbors)))
        record.update(metric_dict("local_effective_dimension", local_effective_dimension(X, args.n_neighbors)))
        record.update(metric_dict("velocity_tangent_alignment", velocity_tangent_alignment(X, V, args.intrinsic_dim, args.n_neighbors)))
        record.update(metric_dict("velocity_neighbor_direction_agreement", velocity_neighbor_direction_agreement(X, V, args.n_neighbors)))
        record.update(metric_dict("velocity_smoothness", velocity_smoothness(X, V, args.n_neighbors)))
        record.update(metric_dict("displacement", displacement_summary(X_raw, X, args.n_neighbors)))
        record.update(metric_dict("knn_overlap", knn_overlap(X_raw, X, args.n_neighbors)))
        record["pairwise_distance_correlation"] = pairwise_distance_correlation(X_raw, X, random_state=args.seed)
        record["trustworthiness"] = trustworthiness_score(X_raw, X, n_neighbors=args.n_neighbors)
        rows.append(record)
    return pd.DataFrame(rows)


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in metrics_df.columns
        if col not in {"method", "status", "error"} and pd.api.types.is_numeric_dtype(metrics_df[col])
    ]
    ok = metrics_df[metrics_df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame(columns=["method"])
    summary = ok.groupby("method")[numeric_cols].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    failures = (
        metrics_df[metrics_df["status"] != "ok"]
        .groupby("method")
        .size()
        .rename("failure_count")
        .reset_index()
    )
    summary = summary.merge(failures, on="method", how="left")
    summary["failure_count"] = summary["failure_count"].fillna(0).astype(int)
    return summary


def alignment_values(X: np.ndarray, V: np.ndarray, intrinsic_dim: int, n_neighbors: int) -> np.ndarray:
    projectors = estimate_local_projectors(X, intrinsic_dim, n_neighbors=n_neighbors)
    projected = np.einsum("nij,nj->ni", projectors, V)
    denom = np.linalg.norm(V, axis=1)
    values = np.full(X.shape[0], np.nan, dtype=float)
    valid = denom > 1e-12
    values[valid] = np.linalg.norm(projected[valid], axis=1) / denom[valid]
    return values


def plot_overview(dataset: dict[str, object], method_rows: list[dict[str, object]], assets_dir: Path) -> Path:
    raw = next(row for row in method_rows if row["method"] == "raw_pca_representation" and row["status"] == "ok")
    preferred = ["raw_pca_representation", "pca_rank_10", "pca_95pct", "velocity_manifold_fitter"]
    panels = [row for name in preferred for row in method_rows if row["method"] == name and row["status"] == "ok"]
    color = np.asarray(dataset["color"], dtype=float)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.7), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    raw_basis = np.asarray(raw["X"])[:, :2]
    span = max(np.ptp(raw_basis[:, 0]), np.ptp(raw_basis[:, 1]), 1e-12)
    for ax, row in zip(axes, panels):
        X = np.asarray(row["X"])
        V = np.asarray(row["V"])
        Z = X[:, :2]
        ax.scatter(Z[:, 0], Z[:, 1], c=color, cmap="viridis", s=10, linewidths=0, alpha=0.8)
        V2 = V[:, :2]
        step = max(1, Z.shape[0] // 60)
        robust = np.percentile(np.linalg.norm(V2, axis=1), 90) + 1e-12
        ax.quiver(
            Z[::step, 0],
            Z[::step, 1],
            V2[::step, 0] * span * 0.04 / robust,
            V2[::step, 1] * span * 0.04 / robust,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.0028,
            alpha=0.55,
            color="#2f3a4a",
        )
        ax.set_title(row["method"])
        ax.set_xticks([])
        ax.set_yticks([])
    path = assets_dir / "cell_cycle_method_overview.png"
    fig.suptitle(dataset["label"])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_movement(method_rows: list[dict[str, object]], assets_dir: Path) -> Path | None:
    raw = next(row for row in method_rows if row["method"] == "raw_pca_representation" and row["status"] == "ok")
    vmf = next((row for row in method_rows if row["method"] == "velocity_manifold_fitter" and row["status"] == "ok"), None)
    if vmf is None:
        return None
    X0 = np.asarray(raw["X"])[:, :2]
    X1 = np.asarray(vmf["X"])[:, :2]
    fig, ax = plt.subplots(figsize=(5.0, 4.2), constrained_layout=True)
    step = max(1, X0.shape[0] // 180)
    for i in range(0, X0.shape[0], step):
        ax.plot([X0[i, 0], X1[i, 0]], [X0[i, 1], X1[i, 1]], color="#8a99ad", linewidth=0.55, alpha=0.5)
    ax.scatter(X0[:, 0], X0[:, 1], s=8, alpha=0.35, label="raw PCA")
    ax.scatter(X1[:, 0], X1[:, 1], s=8, alpha=0.6, label="VMF")
    ax.legend(frameon=False)
    ax.set_title("Cell-cycle raw PCA to VelocityManifoldFitter movement")
    ax.set_xticks([])
    ax.set_yticks([])
    path = assets_dir / "cell_cycle_vmf_movement.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_spectrum(method_rows: list[dict[str, object]], args: argparse.Namespace, assets_dir: Path) -> Path:
    selected = ["raw_pca_representation", "pca_rank_10", "pca_95pct", "velocity_manifold_fitter"]
    fig, ax = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
    for row in method_rows:
        if row["method"] not in selected or row["status"] != "ok":
            continue
        spectrum = average_local_spectrum(row["X"], n_neighbors=args.n_neighbors)
        n_plot = min(15, spectrum.size)
        ax.plot(np.arange(1, n_plot + 1), spectrum[:n_plot], marker="o", label=row["method"])
    ax.set_yscale("log")
    ax.set_xlabel("local eigenvalue index")
    ax.set_ylabel("mean local covariance eigenvalue")
    ax.set_title("Local spectrum decay")
    ax.legend(frameon=False, fontsize=8)
    path = assets_dir / "cell_cycle_local_spectrum.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_alignment_distribution(method_rows: list[dict[str, object]], args: argparse.Namespace, assets_dir: Path) -> Path:
    selected = ["raw_pca_representation", "pca_rank_10", "pca_95pct", "velocity_manifold_fitter"]
    fig, ax = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
    bins = np.linspace(0, 1, 31)
    for row in method_rows:
        if row["method"] not in selected or row["status"] != "ok":
            continue
        values = alignment_values(row["X"], row["V"], args.intrinsic_dim, args.n_neighbors)
        values = values[np.isfinite(values)]
        ax.hist(values, bins=bins, histtype="step", density=True, linewidth=1.6, label=row["method"])
    ax.set_xlabel("velocity-tangent alignment")
    ax.set_ylabel("density")
    ax.set_title("Velocity compatibility distribution")
    ax.legend(frameon=False, fontsize=8)
    path = assets_dir / "cell_cycle_alignment_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def application_interpretation(metrics_df: pd.DataFrame) -> str:
    ok = metrics_df[metrics_df["status"] == "ok"]
    if ok.empty:
        return "No successful real-data method rows were available."
    raw = ok[ok["method"] == "raw_pca_representation"].iloc[0]
    vmf_rows = ok[ok["method"] == "velocity_manifold_fitter"]
    if vmf_rows.empty:
        return "VelocityManifoldFitter did not complete, so interpretation is limited to PCA-style baselines."
    vmf = vmf_rows.iloc[0]
    pca10 = ok[ok["method"] == "pca_rank_10"]
    pca_text = ""
    if not pca10.empty:
        pca10 = pca10.iloc[0]
        if vmf["velocity_tangent_alignment_mean"] > pca10["velocity_tangent_alignment_mean"]:
            pca_text = (
                f" VMF exceeds PCA-rank-10 on velocity-tangent alignment "
                f"({vmf['velocity_tangent_alignment_mean']:.4g} vs {pca10['velocity_tangent_alignment_mean']:.4g})."
            )
        else:
            pca_text = (
                f" In this setting, VMF does not clearly outperform PCA-rank-10 on velocity-tangent alignment "
                f"({vmf['velocity_tangent_alignment_mean']:.4g} vs {pca10['velocity_tangent_alignment_mean']:.4g})."
            )
    tradeoff = ""
    if vmf["displacement_relative_to_local_scale_mean"] > 0.5 or vmf["knn_overlap_mean"] < 0.6:
        tradeoff = (
            " VMF improves velocity-geometry compatibility, but this comes with a movement and "
            "neighborhood-preservation tradeoff."
        )
    return (
        "Because no clean ground-truth manifold is available, these results should be interpreted as "
        "geometry-aware and velocity-aware utility metrics rather than reconstruction accuracy. "
        f"Compared with raw PCA, VelocityManifoldFitter changed mean normal energy ratio from "
        f"{raw['normal_energy_ratio_mean']:.4g} to {vmf['normal_energy_ratio_mean']:.4g}, "
        f"velocity-tangent alignment from {raw['velocity_tangent_alignment_mean']:.4g} to "
        f"{vmf['velocity_tangent_alignment_mean']:.4g}, and kNN overlap was "
        f"{vmf['knn_overlap_mean']:.4g}. Mean displacement was "
        f"{vmf['displacement_mean']:.4g}, or {vmf['displacement_relative_to_local_scale_mean']:.4g} times the local scale."
        f"{pca_text}{tradeoff}"
    )


def skipped_external_notes() -> list[str]:
    notes = []
    palantir = Path("~/Projects/potential_curvature/data/palantir_bone_marrow/marrow_palantir_processed.h5ad").expanduser()
    if not palantir.exists():
        notes.append(f"Skipped Palantir because {palantir} was not available.")
    protein = ROOT / "data" / "protein_latent_paper" / "p450_numeric_onehot.npy"
    if not protein.exists():
        notes.append(f"Skipped P450 protein landscape because {protein} was not available.")
    return notes


def run_report(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_cell_cycle(max_cells=args.max_cells, seed=args.seed)
    method_rows = run_methods(dataset, args)
    metrics_df = compute_metrics(method_rows, args)
    summary_df = summarize_metrics(metrics_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "application_results_long.csv"
    summary_csv = output_dir / "application_results_summary.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    metrics_df.to_csv(output_dir / "application_geometry_metrics.csv", index=False)
    config = vars(args).copy()
    config["output_dir"] = str(config["output_dir"])
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    images = [
        {"path": plot_overview(dataset, method_rows, assets_dir), "caption": "PCA-space method overview"},
        {"path": plot_spectrum(method_rows, args, assets_dir), "caption": "Mean local spectrum decay"},
        {"path": plot_alignment_distribution(method_rows, args, assets_dir), "caption": "Velocity-tangent alignment distribution"},
    ]
    movement = plot_movement(method_rows, assets_dir)
    if movement is not None:
        images.insert(1, {"path": movement, "caption": "Raw-to-VMF movement paths"})

    warnings_out = []
    for row in method_rows:
        if row["status"] != "ok":
            warnings_out.append(f"{row['method']} failed or was skipped: {row['error']}")
    warnings_out.extend(skipped_external_notes())

    sections = [
        {
            "heading": "Run Configuration",
            "text": [
                f"Generated at {datetime.now().isoformat(timespec='seconds')}.",
                f"Cell-cycle data source: {dataset['source']}. Rows analyzed: {len(dataset['selected_cells'])}.",
                "Metrics are computed in the selected PCA representation, not against a ground-truth manifold.",
            ],
            "tables": [pd.DataFrame([config])],
        },
        {
            "heading": "Metric Definitions",
            "text": [
                "Local low-dimensionality metrics summarize neighborhood covariance spectra. Lower normal energy and lower effective dimension can indicate cleaner local geometry, but excessive reduction can indicate collapse.",
                "Velocity compatibility metrics measure whether velocity vectors lie in local tangent spaces, point toward nearby states, and vary smoothly among neighbors.",
                "Movement and preservation metrics quantify how aggressively a method moves cells and whether it preserves the raw PCA neighborhood structure.",
            ],
        },
        {
            "heading": dataset["label"],
            "warnings": warnings_out,
            "tables": [
                {"caption": "Summary metrics", "data": summary_df},
                {"caption": "Long-form metrics", "data": metrics_df},
            ],
            "images": images,
        },
        {
            "heading": "Geometry and velocity utility interpretation",
            "text": [application_interpretation(metrics_df)],
        },
    ]

    report_path = write_html_report("Application Geometry Report", sections, output_dir / "index.html")
    return {"report_path": report_path, "metrics_csv": metrics_csv, "summary_csv": summary_csv, "metrics": metrics_df, "summary": summary_df}


def main() -> None:
    result = run_report(parse_args())
    print(result["report_path"])
    print(result["metrics_csv"])
    print(result["summary_csv"])


if __name__ == "__main__":
    main()
