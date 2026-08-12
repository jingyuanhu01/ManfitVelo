"""P0.2: Half-sphere / closed-curved-surface anomaly diagnosis.

Context (see simulation/current_plan.md P0.2): Local PCA (M4) beats Position-only
MANFIT (M5) and ManfitVelo (M6) on half_sphere_tangent, and the gap does not
shrink with n (confirmed via Scan A/B). This script runs the three
development-seed-only diagnostic tasks the plan asks for, in order to decide
whether the gap is (a) a reasonable trade-off of the pooled shared
hyperparameters on this specific closed/positive-curvature geometry, or (b)
an implementation problem.

1. Per-iteration normal-mean-shift trajectory: does the update oscillate or
   overshoot near the z=0 boundary?
2. A half-sphere-specific small (T, eta_g) grid vs. the pooled/frozen value,
   on half-sphere's own development (tuning) seeds -- everything else
   (theta, kappa, theta_schedule, lambda_v, k) held at the current frozen
   values.
3. Boundary (z near 0) implementation check: does any point's z coordinate
   flip sign during fitting (which would indicate the update is pushing
   points across the sphere's equator incorrectly)?

Development seeds only (TUNING_SEEDS); no final seeds, no ground truth used
for any of the three tasks except the readout metric in task 2
(clean_point_rmse_rel), which the frozen protocol already treats as a
legitimate readout metric for tier-3 hyperparameter comparisons -- see
tune_shared_vmf/tune_shared_position_only, which score the exact same way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import os  # noqa: E402

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_scenarios import vector_data  # noqa: E402
from scripts.velocity_manifold_fitter import VelocityManifoldFitter  # noqa: E402
from simulation.benchmark_core import load_frozen_config, observed_tau  # noqa: E402
from simulation.run_manfitvelo_benchmark import (  # noqa: E402
    TUNING_SEEDS,
    relative_state_metrics,
    shared_knn_graph,
    state_metrics,
    tuning_score,
)

SCENARIO = "half_sphere_tangent"
OUT_DIR = ROOT / "results" / "half_sphere_diagnosis"
EXTENDED_T = 15  # diagnostic-only, longer than the frozen T=3, to see whether
# any oscillation near the boundary keeps growing past the production horizon
BOUNDARY_Z = 0.10  # "near the equator" threshold for task 1/3 bookkeeping


def frozen_vmf_config() -> dict:
    frozen = load_frozen_config()
    cfg = dict(frozen["velocity_manifold_fitter"][SCENARIO])
    return cfg


def make_fitter(Y: np.ndarray, field: np.ndarray, d: int, cfg: dict, seed: int) -> VelocityManifoldFitter:
    return VelocityManifoldFitter(
        Y,
        field,
        d_mode="global",
        global_d=d,
        k=cfg["k"],
        T=cfg["T"],
        eta_g=cfg["eta_g"],
        theta=cfg["theta"],
        kappa=cfg.get("kappa", 1.0),
        bandwidth_mode=cfg.get("bandwidth_mode", "variable"),
        use_PCA=False,
        candidate_mult=cfg.get("candidate_mult", 4),
        random_state=seed,
        lambda_v=cfg.get("lambda_v", 0.0),
        velocity_covariance_mode=cfg.get("velocity_covariance_mode", "centered"),
        velocity_trace_normalization=cfg.get("velocity_trace_normalization", "match_position_trace"),
    )


def run_traced_fit(Y: np.ndarray, field: np.ndarray, d: int, cfg: dict, seed: int, T: int) -> dict:
    """Manually drive VelocityManifoldFitter's update loop (same body as
    VelocityManifoldFitter.fit(update_mode="normal_only")), recording the
    full position snapshot and z-coordinate bookkeeping at every iteration.
    Mirrors the manual-loop technique already used for the theta_schedule
    "ramp" branch in scripts/benchmark_scenarios.py::
    fit_vmf_variant, just extended to snapshot positions."""
    f = make_fitter(Y, field, d, {**cfg, "T": T}, seed)
    working_velocity = f.W.copy()
    f._build_neighbors(working_velocity)

    snapshots = [f.X.copy()]
    per_iteration = []
    sign_flip_events = []
    z0 = f.X[:, 2].copy()

    for t in range(T):
        f._update_weights(velocity_mode="projected")
        f._compute_local_tangent(diagnostic_iteration=t, diagnostic_phase="pre_update")
        f._project_velocity(working_velocity)
        _, mean_shift = f._local_mean_shift()
        tangent_shift = np.einsum("nij,nj->ni", f.P, mean_shift)
        normal_shift = mean_shift - tangent_shift
        steps = f._cap_steps(f.eta_g * normal_shift)
        z_before = f.X[:, 2].copy()
        f.X = f.X + steps
        working_velocity = f.v.copy()
        z_after = f.X[:, 2].copy()

        flips = np.flatnonzero(np.sign(z_before) != np.sign(z_after))
        if len(flips):
            sign_flip_events.append(
                {
                    "iteration": t,
                    "n_flips": int(len(flips)),
                    "point_indices": flips.tolist(),
                    "z_before": z_before[flips].tolist(),
                    "z_after": z_after[flips].tolist(),
                }
            )

        near_boundary = np.abs(z0) < BOUNDARY_Z
        step_norm = np.linalg.norm(steps, axis=1)
        per_iteration.append(
            {
                "iteration": t,
                "mean_step_norm": float(np.mean(step_norm)),
                "mean_step_norm_near_boundary": float(np.mean(step_norm[near_boundary])) if np.any(near_boundary) else float("nan"),
                "mean_step_norm_away_from_boundary": float(np.mean(step_norm[~near_boundary])) if np.any(~near_boundary) else float("nan"),
                "n_near_boundary": int(np.sum(near_boundary)),
                "min_z": float(np.min(f.X[:, 2])),
                "n_negative_z": int(np.sum(f.X[:, 2] < 0)),
            }
        )
        snapshots.append(f.X.copy())

    return {
        "snapshots": snapshots,
        "per_iteration": per_iteration,
        "sign_flip_events": sign_flip_events,
        "z0": z0,
        "n": len(z0),
    }


def task1_trajectory(cfg: dict) -> tuple[pd.DataFrame, list[dict], dict]:
    rows = []
    flip_rows = []
    example_snapshots = None
    for seed in TUNING_SEEDS:
        data = vector_data(SCENARIO, seed)
        for label, T in (("frozen_T", cfg["T"]), ("extended_T", EXTENDED_T)):
            traced = run_traced_fit(data["Y"], data["field"], data["d"], cfg, seed, T)
            for row in traced["per_iteration"]:
                rows.append({"seed": seed, "run": label, **row})
            for event in traced["sign_flip_events"]:
                flip_rows.append({"seed": seed, "run": label, **event})
            if seed == TUNING_SEEDS[0] and label == "extended_T":
                example_snapshots = traced["snapshots"]

    frame = pd.DataFrame(rows)
    return frame, flip_rows, {"example_snapshots": example_snapshots}


def plot_trajectory(frame: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, run in zip(axes, ("frozen_T", "extended_T")):
        sub = frame[frame.run == run]
        for seed, group in sub.groupby("seed"):
            group = group.sort_values("iteration")
            ax.plot(group.iteration, group.mean_step_norm_near_boundary, "o-", label=f"seed {seed} (near boundary)")
            ax.plot(group.iteration, group.mean_step_norm_away_from_boundary, "s--", alpha=0.5, label=f"seed {seed} (away)")
        ax.set_title(f"{run}: mean normal-step norm per iteration")
        ax.set_xlabel("iteration")
        ax.set_ylabel("mean |normal step|")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def task2_local_grid(cfg: dict) -> tuple[pd.DataFrame, dict]:
    grid = [
        {"T": T, "eta_g": eta_g}
        for T in (3, 5, 8)
        for eta_g in (0.35, 0.5, 0.7)
    ]
    rows = []
    for index, candidate in enumerate(grid):
        candidate_cfg = dict(cfg)
        candidate_cfg["T"] = candidate["T"]
        candidate_cfg["eta_g"] = candidate["eta_g"]
        for seed in TUNING_SEEDS:
            data = vector_data(SCENARIO, seed)
            noisy_neighbors = shared_knn_graph(data["Y"], candidate_cfg["k"])
            tau, _, _ = observed_tau(data["Y"], data["field"], noisy_neighbors)
            baseline = state_metrics(SCENARIO, data["Y"], data["field"], data, tau)

            f = make_fitter(data["Y"], data["field"], data["d"], candidate_cfg, seed)
            result = f.fit(update_mode="normal_only", return_dict=True)
            result["V"] = np.einsum("nij,nj->ni", result["P"], data["field"])
            absolute = state_metrics(SCENARIO, result["X"], result["V"], data, tau)
            relative = relative_state_metrics(absolute, baseline)
            rows.append(
                {
                    "candidate_index": index,
                    "T": candidate["T"],
                    "eta_g": candidate["eta_g"],
                    "seed": seed,
                    **relative,
                    "tuning_score": tuning_score(relative),
                }
            )

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["candidate_index", "T", "eta_g"])
        .agg(
            mean_clean_point_rmse_rel=("clean_point_rmse_rel", "mean"),
            mean_tuning_score=("tuning_score", "mean"),
        )
        .reset_index()
        .sort_values("mean_clean_point_rmse_rel")
    )
    best = summary.iloc[0]
    # NOTE: must use bracket indexing for the "T" column -- `summary.T` is
    # pandas' DataFrame.T (transpose) attribute and silently shadows a column
    # literally named "T", which produced bogus (all-NaN) comparisons here
    # until this was caught while reviewing the first run's output.
    pooled = summary[(summary["T"] == cfg["T"]) & (summary["eta_g"] == cfg["eta_g"])].iloc[0]
    comparison = {
        "pooled_T_eta_g": [cfg["T"], cfg["eta_g"]],
        "pooled_mean_clean_point_rmse_rel": float(pooled["mean_clean_point_rmse_rel"]),
        "half_sphere_best_T_eta_g": [int(best["T"]), float(best["eta_g"])],
        "half_sphere_best_mean_clean_point_rmse_rel": float(best["mean_clean_point_rmse_rel"]),
        "relative_gap_pct": float(
            100.0 * (pooled["mean_clean_point_rmse_rel"] - best["mean_clean_point_rmse_rel"])
            / best["mean_clean_point_rmse_rel"]
        ),
    }
    return frame, summary, comparison


def task3_boundary_check(flip_rows: list[dict], cfg: dict) -> dict:
    frozen_flips = [e for e in flip_rows if e["run"] == "frozen_T"]
    extended_flips = [e for e in flip_rows if e["run"] == "extended_T"]

    # Independent check: does the noise model itself ever produce z<0 before
    # fitting (it shouldn't -- see module docstring reasoning)?
    negative_z_in_noisy_input = 0
    for seed in TUNING_SEEDS:
        data = vector_data(SCENARIO, seed)
        negative_z_in_noisy_input += int(np.sum(data["Y"][:, 2] < 0))

    # Does the final fitted output (frozen T=3 config) ever land at z<0?
    negative_z_in_fitted_output = 0
    for seed in TUNING_SEEDS:
        data = vector_data(SCENARIO, seed)
        f = make_fitter(data["Y"], data["field"], data["d"], cfg, seed)
        result = f.fit(update_mode="normal_only", return_dict=True)
        negative_z_in_fitted_output += int(np.sum(result["X"][:, 2] < 0))

    return {
        "boundary_z_threshold": BOUNDARY_Z,
        "sign_flip_events_frozen_T": frozen_flips,
        "sign_flip_events_extended_T": extended_flips,
        "n_sign_flip_events_frozen_T": len(frozen_flips),
        "n_sign_flip_events_extended_T": len(extended_flips),
        "negative_z_in_noisy_input_pooled_over_dev_seeds": negative_z_in_noisy_input,
        "negative_z_in_frozen_fit_output_pooled_over_dev_seeds": negative_z_in_fitted_output,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = frozen_vmf_config()
    print(f"Frozen half_sphere_tangent VMF config: {cfg}")

    print("Task 1: per-iteration trajectory ...")
    traj_frame, flip_rows, extras = task1_trajectory(cfg)
    traj_frame.to_csv(OUT_DIR / "task1_trajectory.csv", index=False)
    plot_trajectory(traj_frame, OUT_DIR / "task1_trajectory.png")

    print("Task 2: half-sphere-specific (T, eta_g) grid ...")
    grid_long, grid_summary, comparison = task2_local_grid(cfg)
    grid_long.to_csv(OUT_DIR / "task2_grid_long.csv", index=False)
    grid_summary.to_csv(OUT_DIR / "task2_grid_summary.csv", index=False)

    print("Task 3: boundary (z near 0) implementation check ...")
    boundary = task3_boundary_check(flip_rows, cfg)
    (OUT_DIR / "task3_boundary_check.json").write_text(json.dumps(boundary, indent=2))

    notes = {
        "scenario": SCENARIO,
        "frozen_vmf_config": cfg,
        "task2_comparison": comparison,
        "task3_summary": {
            k: v for k, v in boundary.items() if not k.startswith("sign_flip_events")
        },
    }
    (OUT_DIR / "p0_2_summary.json").write_text(json.dumps(notes, indent=2))

    print(json.dumps(notes, indent=2))
    print(f"\nWrote outputs to {OUT_DIR}/")


if __name__ == "__main__":
    main()
