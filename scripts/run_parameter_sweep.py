"""Quick parameter sweep for VelocityManifoldFitter tradeoffs.

The sweep is intentionally scoped to VMF parameters. It does not replace the
full simulation benchmark; it helps identify whether utility gains require
large movement or neighborhood disruption.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.geometry_velocity_metrics import (  # noqa: E402
    displacement_summary,
    knn_overlap,
    mean_distance_to_clean_cloud,
    mean_l2_to_clean,
    metric_dict,
    normal_energy_ratio,
    quantile_distance_to_clean_cloud,
    rmse_to_clean,
    velocity_neighbor_direction_agreement,
    velocity_smoothness,
    velocity_tangent_alignment,
)
from scripts.html_report_utils import write_html_report  # noqa: E402
from scripts.run_simulation_benchmark import build_dataset_specs, pad_to_dim  # noqa: E402
from scripts.velocity_manifold_fitter import VelocityManifoldFitter  # noqa: E402


PARAM_COLUMNS = ["eta_g", "theta", "k", "T", "adaptive_variance_threshold"]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use a small default grid suitable for local iteration.")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--n-seeds", "--n_seeds", dest="n_seeds", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--position-noise", type=float, default=0.3)
    parser.add_argument("--velocity-noise", type=float, default=0.3)
    parser.add_argument("--extra-dims", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=25)
    parser.add_argument("--eta-g-values", default=None)
    parser.add_argument("--theta-values", default=None)
    parser.add_argument("--k-values", default=None)
    parser.add_argument("--T-values", default=None)
    parser.add_argument("--adaptive-threshold-values", default=None)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "parameter_sweep")
    args = parser.parse_args()

    if args.quick:
        args.datasets = args.datasets or "s_curve,swiss_roll,half_sphere_rotation"
        args.n_seeds = 2 if args.n_seeds is None else args.n_seeds
        args.n_samples = 100 if args.n_samples is None else args.n_samples
        args.eta_g_values = args.eta_g_values or "0.2,0.35,0.5"
        args.theta_values = args.theta_values or "0.05,0.15,0.3"
        args.k_values = args.k_values or "15,25"
        args.T_values = args.T_values or "3"
        args.adaptive_threshold_values = args.adaptive_threshold_values or "0.8,0.9"
    else:
        args.datasets = args.datasets or "s_curve,swiss_roll,half_sphere_rotation,potential_saddle_surface_saddle"
        args.n_seeds = 3 if args.n_seeds is None else args.n_seeds
        args.n_samples = 120 if args.n_samples is None else args.n_samples
        args.eta_g_values = args.eta_g_values or "0.2,0.35,0.5"
        args.theta_values = args.theta_values or "0.05,0.15,0.3"
        args.k_values = args.k_values or "15,25"
        args.T_values = args.T_values or "3,5"
        args.adaptive_threshold_values = args.adaptive_threshold_values or "0.8,0.9"

    args.eta_g_grid = parse_float_list(args.eta_g_values)
    args.theta_grid = parse_float_list(args.theta_values)
    args.k_grid = parse_int_list(args.k_values)
    args.T_grid = parse_int_list(args.T_values)
    args.adaptive_threshold_grid = parse_float_list(args.adaptive_threshold_values)
    if args.n_seeds < 1:
        raise ValueError("--n-seeds must be at least 1")
    return args


def dataset_args(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        datasets=args.datasets,
        n_samples=args.n_samples,
        position_noise=args.position_noise,
        velocity_noise=args.velocity_noise,
        extra_dims=args.extra_dims,
        seed=seed,
    )


def parameter_grid(args: argparse.Namespace) -> list[dict[str, float | int]]:
    return [
        {
            "eta_g": eta_g,
            "theta": theta,
            "k": k,
            "T": T,
            "adaptive_variance_threshold": threshold,
        }
        for eta_g in args.eta_g_grid
        for theta in args.theta_grid
        for k in args.k_grid
        for T in args.T_grid
        for threshold in args.adaptive_threshold_grid
    ]


def run_one_fit(spec, params: dict[str, float | int], seed: int, args: argparse.Namespace) -> dict[str, object]:
    base = {
        "dataset": spec.key,
        "dataset_label": spec.label,
        "seed": seed,
        **params,
        "status": "ok",
        "error": "",
    }
    X = np.asarray(spec.simulation["X"], dtype=float)
    V = np.asarray(spec.simulation["V"], dtype=float)
    try:
        fitter = VelocityManifoldFitter(
            X,
            V,
            d_mode="adaptive",
            adaptive_variance_threshold=float(params["adaptive_variance_threshold"]),
            adaptive_d_min=spec.intrinsic_dim,
            adaptive_d_max=min(max(2 * spec.intrinsic_dim, spec.intrinsic_dim), X.shape[1]),
            k=int(params["k"]),
            T=int(params["T"]),
            eta_g=float(params["eta_g"]),
            theta=float(params["theta"]),
            kappa=args.kappa,
            bandwidth_mode="variable",
            use_PCA=False,
            random_state=seed,
        )
        result = fitter.fit(update_mode="normal_only", return_dict=True)
        X_hat = np.asarray(result["X"], dtype=float)
        V_hat = np.asarray(result["V"], dtype=float)
        X_clean = pad_to_dim(np.asarray(spec.simulation["X_gt"], dtype=float), X_hat.shape[1])
        base.update(
            {
                "rmse_to_clean": rmse_to_clean(X_hat, X_clean),
                "mean_l2_to_clean": mean_l2_to_clean(X_hat, X_clean),
                "mean_distance_to_clean_cloud_approx": mean_distance_to_clean_cloud(X_hat, X_clean),
                "q90_distance_to_clean_cloud_approx": quantile_distance_to_clean_cloud(X_hat, X_clean, q=0.9),
                "mean_local_dim": float(np.mean(result.get("local_dims", [spec.intrinsic_dim]))),
            }
        )
        base.update(metric_dict("normal_energy_ratio", normal_energy_ratio(X_hat, spec.intrinsic_dim, args.n_neighbors)))
        base.update(metric_dict("velocity_tangent_alignment", velocity_tangent_alignment(X_hat, V_hat, spec.intrinsic_dim, args.n_neighbors)))
        base.update(metric_dict("velocity_neighbor_direction_agreement", velocity_neighbor_direction_agreement(X_hat, V_hat, args.n_neighbors)))
        base.update(metric_dict("velocity_smoothness", velocity_smoothness(X_hat, V_hat, args.n_neighbors)))
        base.update(metric_dict("displacement", displacement_summary(X, X_hat, args.n_neighbors)))
        base.update(metric_dict("knn_overlap", knn_overlap(X, X_hat, args.n_neighbors)))
    except Exception as exc:  # noqa: BLE001
        base["status"] = "failed"
        base["error"] = f"{type(exc).__name__}: {exc}"
        base["traceback"] = traceback.format_exc(limit=3)
    return base


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in results_df.columns
        if col not in {"dataset", "dataset_label", "seed", "status", "error", "traceback", *PARAM_COLUMNS}
        and pd.api.types.is_numeric_dtype(results_df[col])
    ]
    ok = results_df[results_df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame(columns=["dataset", *PARAM_COLUMNS])
    summary = ok.groupby(["dataset", *PARAM_COLUMNS])[numeric_cols].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    failures = (
        results_df[results_df["status"] != "ok"]
        .groupby(["dataset", *PARAM_COLUMNS])
        .size()
        .rename("failure_count")
        .reset_index()
    )
    summary = summary.merge(failures, on=["dataset", *PARAM_COLUMNS], how="left")
    summary["failure_count"] = summary["failure_count"].fillna(0).astype(int)
    return add_balanced_rank(summary)


def add_balanced_rank(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    out = summary_df.copy()
    out["balanced_rank"] = np.nan
    for dataset, idx in out.groupby("dataset").groups.items():
        subset = out.loc[idx]
        ranks = []
        rank_specs = [
            ("rmse_to_clean_mean", True),
            ("mean_distance_to_clean_cloud_approx_mean", True),
            ("velocity_tangent_alignment_mean_mean", False),
            ("displacement_relative_to_local_scale_mean_mean", True),
            ("knn_overlap_mean_mean", False),
        ]
        for col, ascending in rank_specs:
            if col in subset:
                ranks.append(subset[col].rank(ascending=ascending, method="average"))
        if ranks:
            out.loc[idx, "balanced_rank"] = pd.concat(ranks, axis=1).mean(axis=1)
    return out


def params_text(row: pd.Series) -> str:
    def get(name: str):
        if hasattr(row, name):
            return getattr(row, name)
        return row[name]

    return (
        f"eta_g={get('eta_g')}, theta={get('theta')}, k={int(get('k'))}, "
        f"T={int(get('T'))}, adaptive={get('adaptive_variance_threshold')}"
    )


def best_row(summary_df: pd.DataFrame, dataset: str, metric: str, ascending: bool) -> pd.Series | None:
    subset = summary_df[(summary_df["dataset"] == dataset) & summary_df[metric].notna()]
    if subset.empty:
        return None
    return subset.sort_values(metric, ascending=ascending).iloc[0]


def interpretation_paragraphs(summary_df: pd.DataFrame, results_df: pd.DataFrame) -> list[str]:
    paragraphs: list[str] = []
    for dataset in sorted(summary_df["dataset"].unique()):
        best_align = best_row(summary_df, dataset, "velocity_tangent_alignment_mean_mean", ascending=False)
        best_disp = best_row(summary_df, dataset, "displacement_relative_to_local_scale_mean_mean", ascending=True)
        best_knn = best_row(summary_df, dataset, "knn_overlap_mean_mean", ascending=False)
        best_balanced = best_row(summary_df, dataset, "balanced_rank", ascending=True)
        if best_align is None:
            paragraphs.append(f"{dataset}: no successful sweep rows were available.")
            continue
        text = (
            f"{dataset}: best velocity-tangent alignment is {best_align['velocity_tangent_alignment_mean_mean']:.4g} "
            f"with {params_text(best_align)}, displacement {best_align['displacement_relative_to_local_scale_mean_mean']:.4g} "
            f"local scales, and kNN overlap {best_align['knn_overlap_mean_mean']:.4g}. "
            f"Lowest movement is {best_disp['displacement_relative_to_local_scale_mean_mean']:.4g} local scales with "
            f"{params_text(best_disp)}. Best kNN preservation is {best_knn['knn_overlap_mean_mean']:.4g} with "
            f"{params_text(best_knn)}. Balanced rank favors {params_text(best_balanced)}."
        )
        if best_align["displacement_relative_to_local_scale_mean_mean"] > 0.5:
            text += " Utility improves only with a nontrivial movement cost for the alignment-optimal setting."
        paragraphs.append(text)

    failed = results_df[results_df["status"] != "ok"]
    if failed.empty:
        paragraphs.append("No parameter settings failed in this sweep.")
    else:
        failures = failed[["dataset", *PARAM_COLUMNS, "error"]].drop_duplicates()
        paragraphs.append("Failed settings: " + "; ".join(f"{row.dataset} {params_text(row)}: {row.error}" for row in failures.itertuples()))
    return paragraphs


def plot_tradeoff(summary_df: pd.DataFrame, dataset: str, assets_dir: Path) -> Path:
    subset = summary_df[summary_df["dataset"] == dataset].copy()
    fig, ax = plt.subplots(figsize=(5.2, 4.1), constrained_layout=True)
    scatter = ax.scatter(
        subset["displacement_relative_to_local_scale_mean_mean"],
        subset["velocity_tangent_alignment_mean_mean"],
        c=subset["rmse_to_clean_mean"],
        cmap="viridis_r",
        s=42,
        alpha=0.8,
        edgecolors="none",
    )
    best = subset.sort_values("balanced_rank").iloc[0]
    ax.scatter(
        [best["displacement_relative_to_local_scale_mean_mean"]],
        [best["velocity_tangent_alignment_mean_mean"]],
        marker="x",
        s=90,
        color="#b91c1c",
        label="best balanced rank",
    )
    ax.set_xlabel("mean displacement / local scale")
    ax.set_ylabel("mean velocity-tangent alignment")
    ax.set_title(f"{dataset}: alignment vs movement tradeoff")
    ax.legend(frameon=False)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("RMSE to clean")
    path = assets_dir / f"{dataset}_alignment_movement_tradeoff.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run_sweep(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    grid = parameter_grid(args)

    rows: list[dict[str, object]] = []
    for seed_index in range(args.n_seeds):
        seed = int(args.seed) + seed_index
        specs = build_dataset_specs(dataset_args(args, seed))
        for spec in specs:
            for params in grid:
                rows.append(run_one_fit(spec, params, seed, args))

    results_df = pd.DataFrame(rows)
    summary_df = summarize_results(results_df)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "parameter_sweep_results.csv"
    summary_csv = output_dir / "parameter_sweep_summary.csv"
    results_df.to_csv(results_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    config = vars(args).copy()
    for key in ["eta_g_grid", "theta_grid", "k_grid", "T_grid", "adaptive_threshold_grid"]:
        config[key] = list(config[key])
    config["output_dir"] = str(config["output_dir"])
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    images = [
        {"path": plot_tradeoff(summary_df, dataset, assets_dir), "caption": f"{dataset} tradeoff"}
        for dataset in sorted(summary_df["dataset"].unique())
    ]
    top_balanced = (
        summary_df.sort_values(["dataset", "balanced_rank"])
        .groupby("dataset", as_index=False)
        .head(5)
        .loc[
            :,
            [
                "dataset",
                *PARAM_COLUMNS,
                "balanced_rank",
                "rmse_to_clean_mean",
                "mean_distance_to_clean_cloud_approx_mean",
                "velocity_tangent_alignment_mean_mean",
                "displacement_relative_to_local_scale_mean_mean",
                "knn_overlap_mean_mean",
            ],
        ]
    )
    sections = [
        {
            "heading": "Run Configuration",
            "text": [
                f"Generated at {datetime.now().isoformat(timespec='seconds')}.",
                "This sweep varies VMF parameters and reports utility tradeoffs, not a single winner.",
            ],
            "tables": [pd.DataFrame([config])],
        },
        {
            "heading": "Parameter Sweep Interpretation",
            "text": interpretation_paragraphs(summary_df, results_df),
            "tables": [
                {"caption": "Top five balanced settings per dataset", "data": top_balanced},
                {"caption": "Full summary", "data": summary_df},
            ],
            "images": images,
        },
    ]
    report_path = write_html_report("VelocityManifoldFitter Parameter Sweep", sections, output_dir / "index.html")
    return {"report_path": report_path, "results_csv": results_csv, "summary_csv": summary_csv, "results": results_df, "summary": summary_df}


def main() -> None:
    result = run_sweep(parse_args())
    print("Parameter sweep completed.")
    print(result["report_path"])
    print(result["results_csv"])
    print(result["summary_csv"])


if __name__ == "__main__":
    main()
