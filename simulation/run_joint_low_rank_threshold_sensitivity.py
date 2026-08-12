"""P2.2: Joint Low-Rank (M3) threshold sensitivity.

Sweep variance_threshold in {0.80, 0.90, 0.95, 0.99} (the frozen protocol uses
0.90, scripts/simulation_baselines.JOINT_LOW_RANK_VARIANCE_THRESHOLD). Single
question: is M3's curved-geometry failure pattern specific to q=0.90, or does
it hold at every threshold? If all four tell the same qualitative story,
freeze M3 as-is (current_plan.md P2.2) -- this is a robustness/reporting check
with a pre-committed decision rule, not a performance-based selection of q
(q stays 0.90 regardless of what this finds, unless it turns up a genuine
implementation problem), so it is run on FINAL_SEEDS like the other
supplement/reporting tables in this pipeline (e.g. GraphVelo raw-vs-
standardized), not TUNING_SEEDS.

Also reports each threshold's per-scenario selected rank r* (out of the
joint [X,V] matrix's ambient rank, min(n, 2*ambient_dim) = 6 for every
canonical scenario here), and a low-cost mechanism diagnostic: the rank a
V-only SVD would need to reach the same variance thresholds, compared
against the joint [X,V] rank -- explains why M3's velocity reconstruction is
disproportionately strong (Round 3, log.md) without promoting this into a
full eighth baseline.

    python simulation/run_joint_low_rank_threshold_sensitivity.py
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

from scripts.simulation_baselines import JOINT_LOW_RANK_VARIANCE_THRESHOLD, joint_low_rank_state, shared_knn_graph  # noqa: E402
from simulation.benchmark_core import observed_tau  # noqa: E402
from simulation.run_manfitvelo_benchmark import (  # noqa: E402
    FINAL_SEEDS,
    SCENARIOS,
    relative_state_metrics,
    state_metrics,
    vector_data,
)

THRESHOLDS = (0.80, 0.90, 0.95, 0.99)
OUT_DIR = ROOT / "results" / "joint_low_rank_threshold_sensitivity"

HEADLINE_METRICS = (
    "clean_point_rmse_rel",
    "distance_to_manifold_rel",
    "velocity_rmse_id_rel",
    "velocity_rmse_loc_rel",
)


def v_only_rank(V: np.ndarray, variance_threshold: float) -> int:
    """Rank a V-only SVD would need to reach variance_threshold, for the
    same [X,V] joint block-normalization convention (v_centered/v_scale) so
    the comparison is apples-to-apples with joint_low_rank_state's own rank."""
    v_centered = V - V.mean(axis=0)
    v_scale = float(np.linalg.norm(v_centered))
    v_scale = v_scale if v_scale > 1e-12 else 1.0
    _, singular_values, _ = np.linalg.svd(v_centered / v_scale, full_matrices=False)
    energy = singular_values**2
    cumulative = np.cumsum(energy) / max(float(np.sum(energy)), 1e-12)
    rank = int(np.searchsorted(cumulative, variance_threshold) + 1)
    return min(rank, len(singular_values))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in SCENARIOS:
        for seed in FINAL_SEEDS:
            data = vector_data(scenario, seed)
            k = min(50, len(data["Y"]) - 1)
            graph = shared_knn_graph(data["Y"], k)
            tau, _, _ = observed_tau(data["Y"], data["field"], graph)
            baseline = state_metrics(scenario, data["Y"], data["field"], data, tau)
            for q in THRESHOLDS:
                Xhat, Vhat, info = joint_low_rank_state(data["Y"], data["field"], variance_threshold=q)
                absolute = state_metrics(scenario, Xhat, Vhat, data, tau)
                relative = relative_state_metrics(absolute, baseline)
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "variance_threshold": q,
                        "joint_rank": info["rank"],
                        "joint_max_rank": info["n_singular_values"],
                        "v_only_rank": v_only_rank(data["field"], q),
                        **relative,
                    }
                )

    long_frame = pd.DataFrame(rows)
    long_frame.to_csv(OUT_DIR / "threshold_sensitivity_long.csv", index=False)

    rank_summary = (
        long_frame.groupby(["scenario", "variance_threshold"])
        .agg(
            median_joint_rank=("joint_rank", "median"),
            median_v_only_rank=("v_only_rank", "median"),
            joint_max_rank=("joint_max_rank", "first"),
        )
        .reset_index()
    )
    metric_summary = (
        long_frame.groupby(["scenario", "variance_threshold"])[list(HEADLINE_METRICS)]
        .median()
        .reset_index()
    )
    summary = rank_summary.merge(metric_summary, on=["scenario", "variance_threshold"])
    summary.to_csv(OUT_DIR / "threshold_sensitivity_summary.csv", index=False)

    # Qualitative-consistency check: does every threshold agree on whether
    # M3 is worse than noisy input (clean_point_rmse_rel > 1) on each
    # scenario, and does the curved-vs-flat rank pattern hold at every q?
    consistency_rows = []
    for scenario in SCENARIOS:
        sub = summary[summary.scenario == scenario]
        worse_than_noisy = (sub.clean_point_rmse_rel > 1.0).tolist()
        consistency_rows.append(
            {
                "scenario": scenario,
                "worse_than_noisy_at_each_threshold": dict(zip(sub.variance_threshold, worse_than_noisy)),
                "qualitatively_consistent_across_thresholds": bool(len(set(worse_than_noisy)) == 1),
                "rank_range": [int(sub.median_joint_rank.min()), int(sub.median_joint_rank.max())],
                "joint_max_rank": int(sub.joint_max_rank.iloc[0]),
            }
        )
    all_consistent = all(row["qualitatively_consistent_across_thresholds"] for row in consistency_rows)

    notes = {
        "thresholds": list(THRESHOLDS),
        "frozen_threshold": JOINT_LOW_RANK_VARIANCE_THRESHOLD,
        "final_seeds_used": True,
        "note": "Robustness/reporting check with a pre-committed decision rule; q not re-selected by performance.",
        "all_scenarios_qualitatively_consistent_across_thresholds": all_consistent,
        "per_scenario_consistency": consistency_rows,
        "decision": "freeze M3 as-is (q=0.90)" if all_consistent else "inconsistent across thresholds -- needs review before freezing M3",
    }
    (OUT_DIR / "p2_2_summary.json").write_text(json.dumps(notes, indent=2, default=str))

    print(json.dumps(notes, indent=2, default=str))
    print(f"\nWrote {OUT_DIR}/threshold_sensitivity_long.csv, threshold_sensitivity_summary.csv, p2_2_summary.json")


if __name__ == "__main__":
    main()
