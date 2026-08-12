"""P0.1: select a single global neighbor-count constant C.

Replaces the current per-dimension NEIGHBOR_SCALING_CONSTANT = {1: C_1, 2: C_2}
(simulation/benchmark_core.py) -- reverse-engineered by forcing k(n0, d) = 40 on
one anchor scenario per dimension -- with a single dimension-independent C in

    k(n, d) = clip(ceil(C * n**(4 / (d + 4))), k_min, k_max)

Method (confirmed with the user 2026-08-12; see simulation/current_plan.md P0.1
"方法论已定" for the full rationale):

- Candidates: C in {0.30, 0.45, 0.60, 0.75, 0.90} -- 5 values, evenly spaced,
  chosen to cover and straddle both current anchor values (C_1 ~= 0.361,
  C_2 ~= 0.713), not to bracket a wider or narrower range.
- For each candidate C, per-scenario k is derived by the SAME two-stage
  procedure already used by the frozen protocol: a Stage-1 ceiling from the
  formula above, then the existing Stage-2 curvature-aware refinement
  (benchmark_core.local_pca_normal_residual / curvature_aware_neighbor_count,
  unchanged) sweeping from a small floor up to that ceiling. Only the Stage-1
  ceiling's constant changes; the refinement mechanism itself is untouched.
- Evaluated on TUNING_SEEDS only (42000-42002), pooled over all 9 canonical
  scenarios. FINAL_SEEDS never enter this selection.
- The other VMF / Position-only MANFIT hyperparameters (T, eta_g, theta,
  kappa, theta_schedule, lambda_v, velocity_covariance_mode,
  velocity_trace_normalization) are held at their CURRENTLY FROZEN values
  (results/manfitvelo_benchmark/selected_hyperparameters.json) rather than
  re-tuned per candidate C. C is a tier-2 (data-adaptive) choice and is
  selected first; once a winning C is confirmed, the tier-3
  T/eta_g/theta/kappa/theta_schedule grid (shared_vmf_grid /
  shared_position_only_grid in run_manfitvelo_benchmark.py) must be re-run
  once against that C as a separate follow-up step -- not done by this
  script.
- Scored by tuning_score (mean log of the 4 relative headline metrics:
  clean_point_rmse_rel, velocity_rmse_id_rel, velocity_angle_mae_id_rel,
  joint_euler_state_rmse_rel), the same function shared_vmf_grid /
  shared_position_only_grid already use.
- Primary selection criterion: ManfitVelo (M6)'s pooled tuning_score (mean
  over 9 scenarios x 3 tuning seeds), matching how every other shared
  hyperparameter is currently selected. Position-only MANFIT (M5)'s pooled
  tuning_score is computed and reported alongside for transparency but is not
  part of the selection rule.

This is a standalone diagnostic/selection script, not part of the frozen
main() pipeline: it does not write to results/manfitvelo_benchmark/ and does
not modify simulation/benchmark_core.py's NEIGHBOR_SCALING_CONSTANT. Once a
winning C is confirmed with the user, benchmark_core.py is updated and the
full protocol is re-frozen as a separate follow-up (rerun of
run_manfitvelo_benchmark.py, run_sphere_scalability.py, run_stress_scans.py,
run_lambda_sensitivity.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_scenarios import (  # noqa: E402
    fit_vmf_variant,
    position_only_trajectory,
    vector_data,
)
from simulation.benchmark_core import (  # noqa: E402
    NEIGHBOR_COUNT_CLIP,
    SCENARIO_LABELS,
    curvature_aware_neighbor_count,
    load_frozen_config,
    local_pca_normal_residual,
)
from simulation.run_manfitvelo_benchmark import (  # noqa: E402
    SCENARIOS,
    TUNING_SEEDS,
    relative_state_metrics,
    shared_knn_graph,
    state_metrics,
    tuning_score,
)
from scripts.benchmark_scenarios import SETS  # noqa: E402
from simulation.benchmark_core import observed_tau  # noqa: E402

CANDIDATE_C = (0.30, 0.45, 0.60, 0.75, 0.90)
CURVATURE_GRID_POINTS = 14
OUT_DIR = ROOT / "results" / "c_selection"


def candidate_stage1_k(n: int, d: int, C: float) -> int:
    """Stage-1 ceiling k(n, d) under a candidate single global C."""
    exponent = 4.0 / (d + 4)
    raw = C * (float(n) ** exponent)
    k_min, k_max = NEIGHBOR_COUNT_CLIP
    return int(np.clip(np.ceil(raw), k_min, k_max))


def candidate_k_grid(n: int, d: int, C: float, *, num: int = CURVATURE_GRID_POINTS) -> list[int]:
    """Same shape as benchmark_core.curvature_probe_k_grid, but ceiling comes
    from a candidate global C instead of the frozen NEIGHBOR_SCALING_CONSTANT."""
    ceiling = candidate_stage1_k(n, d, C)
    floor = min(max(2 * d + 2, 8), ceiling)
    if floor >= ceiling:
        return [ceiling]
    return sorted({int(round(v)) for v in np.geomspace(floor, ceiling, num=num)})


def curvature_aware_k_for_candidate(scenario: str, C: float) -> tuple[int, dict]:
    """Stage-1 (candidate C) + Stage-2 (unchanged curvature-aware refinement)."""
    setting = SETS[scenario]
    k_grid = candidate_k_grid(setting.n, setting.d, C)
    curves = [
        local_pca_normal_residual(vector_data(scenario, seed)["Y"], setting.d, k_grid)
        for seed in TUNING_SEEDS
    ]
    return curvature_aware_neighbor_count(k_grid, curves)


def evaluate_candidate(C: float, frozen: dict) -> list[dict]:
    vmf_base = frozen["velocity_manifold_fitter"]  # per-scenario dict, T/eta_g/theta/... identical, k differs
    pos_base = frozen["position_only_manfit"]

    rows: list[dict] = []
    for scenario in SCENARIOS:
        safe_k, diag = curvature_aware_k_for_candidate(scenario, C)
        stage1_ceiling = candidate_stage1_k(SETS[scenario].n, SETS[scenario].d, C)

        vmf_cfg = dict(vmf_base[scenario])
        vmf_cfg["k"] = safe_k
        pos_T, pos_eta_g = pos_base[scenario]["T"], pos_base[scenario]["eta_g"]

        for seed in TUNING_SEEDS:
            data = vector_data(scenario, seed)
            noisy_neighbors = shared_knn_graph(data["Y"], safe_k)
            tau, _, _ = observed_tau(data["Y"], data["field"], noisy_neighbors)
            baseline = state_metrics(scenario, data["Y"], data["field"], data, tau)

            vmf_result = fit_vmf_variant(data["Y"], data["field"], data["d"], vmf_cfg, seed)
            vmf_absolute = state_metrics(scenario, vmf_result["X"], vmf_result["V"], data, tau)
            vmf_relative = relative_state_metrics(vmf_absolute, baseline)

            pos_trajectory = position_only_trajectory(
                data["Y"], data["field"], data["d"], safe_k, pos_T, pos_eta_g
            )
            _, pos_X, pos_V = pos_trajectory[-1]
            pos_absolute = state_metrics(scenario, pos_X, pos_V, data, tau)
            pos_relative = relative_state_metrics(pos_absolute, baseline)

            rows.append(
                {
                    "candidate_C": C,
                    "scenario": scenario,
                    "seed": seed,
                    "stage1_ceiling_k": stage1_ceiling,
                    "curvature_aware_k": safe_k,
                    "turn_index": diag["turn_index"],
                    "method": "manfitvelo",
                    **vmf_relative,
                    "tuning_score": tuning_score(vmf_relative),
                }
            )
            rows.append(
                {
                    "candidate_C": C,
                    "scenario": scenario,
                    "seed": seed,
                    "stage1_ceiling_k": stage1_ceiling,
                    "curvature_aware_k": safe_k,
                    "turn_index": diag["turn_index"],
                    "method": "position_only_manfit",
                    **pos_relative,
                    "tuning_score": tuning_score(pos_relative),
                }
            )
    return rows


def main() -> None:
    frozen = load_frozen_config()
    all_rows: list[dict] = []
    for C in CANDIDATE_C:
        print(f"Evaluating candidate C={C} ...")
        all_rows.extend(evaluate_candidate(C, frozen))

    long_frame = pd.DataFrame(all_rows)

    summary = (
        long_frame.groupby(["candidate_C", "method"])
        .agg(mean_tuning_score=("tuning_score", "mean"), n_rows=("tuning_score", "size"))
        .reset_index()
    )

    # Per-scenario k under each candidate, for audit (should match the
    # curvature-aware, per-seed-averaged k already used by the frozen
    # protocol when C happens to coincide with the current anchors).
    k_table = (
        long_frame[long_frame.method == "manfitvelo"]
        .drop_duplicates(["candidate_C", "scenario"])[
            ["candidate_C", "scenario", "stage1_ceiling_k", "curvature_aware_k"]
        ]
        .sort_values(["candidate_C", "scenario"])
    )

    vmf_summary = summary[summary.method == "manfitvelo"].sort_values("mean_tuning_score")
    winner = float(vmf_summary.iloc[0]["candidate_C"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_frame.to_csv(OUT_DIR / "c_selection_long.csv", index=False)
    summary.to_csv(OUT_DIR / "c_selection_summary.csv", index=False)
    k_table.to_csv(OUT_DIR / "c_selection_k_table.csv", index=False)
    (OUT_DIR / "c_selection_notes.json").write_text(
        json.dumps(
            {
                "candidates": list(CANDIDATE_C),
                "selection_uses_final_seeds": False,
                "scoring_method": "manfitvelo (M6) pooled tuning_score, mean over 9 scenarios x 3 tuning seeds",
                "winner_by_manfitvelo_tuning_score": winner,
                "note": (
                    "Diagnostic/selection run only; benchmark_core.py "
                    "NEIGHBOR_SCALING_CONSTANT not modified by this script."
                ),
            },
            indent=2,
        )
    )

    print("\n=== ManfitVelo (M6) pooled tuning_score by candidate C (primary criterion) ===")
    print(vmf_summary.to_string(index=False))
    print("\n=== Position-only MANFIT (M5) pooled tuning_score by candidate C (reported, not selection rule) ===")
    print(summary[summary.method == "position_only_manfit"].sort_values("mean_tuning_score").to_string(index=False))
    print(f"\nWinning candidate by M6 tuning_score: C = {winner}")
    print(f"\nWrote {OUT_DIR}/c_selection_long.csv, c_selection_summary.csv, c_selection_k_table.csv, c_selection_notes.json")


if __name__ == "__main__":
    main()
