"""P4.1: separate local-regression error from joint geometric-fitting error
in the scalar-field pipeline.

current_plan.md P4.1. Pipeline: (X_i, s_i) -> grad_hat(X_i) -> ManfitVelo-type
joint fitting (fit_scalar_gradient_manfit). Total error has two sources:

    ||grad_hat - grad_true||        (local regression: estimating the
                                      gradient from noisy scalar
                                      observations)
    + joint geometric fitting error  (the manifold-fitting stage itself)

The oracle-gradient ablation isolates the second term: feed the exact true
gradient directly into fit_scalar_gradient_manfit's oracle_gradient=
parameter, which replaces every re-estimation step with the ground truth
(confidence fixed at 1.0) while leaving every other mechanic (k, T,
lambda_v, ...) unchanged. Whatever error remains under oracle_gradient is
attributable to joint fitting alone; the gap between the realistic
(estimated-gradient) pipeline and the oracle pipeline is attributable to
local-regression error propagating downstream.

Also runs both lambda_v in {0.0, 1.0} (with velocity_covariance_mode=
"uncentered" when lambda_v=1.0, matching the frozen vector-field M6
protocol) under both gradient sources, since P4.0/P4.1 explicitly flag that
whether the vector-field-tuned lambda_v=1.0 is even appropriate for scalar
gradients has not been validated -- not a selection (nothing here changes
any frozen value), a diagnostic reported alongside the oracle/estimated
split.

Uses the two existing scalar scenarios (scalar_s_curve, scalar_saddle --
S1/S2 controlled scalar experiments are separate, later P4 work). 15 final
seeds, reporting only.

    python simulation/run_p4_1_scalar_oracle_ablation.py
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

from scripts.benchmark_scenarios import distance_to_manifold, scalar_data  # noqa: E402
from scripts.scalar_potential_manfit import estimate_gradient_from_neighbors, fit_scalar_gradient_manfit  # noqa: E402
from simulation.benchmark_core import angle_mae, neighbor_count, vector_rmse  # noqa: E402
from simulation.run_manfitvelo_benchmark import FINAL_SEEDS  # noqa: E402

SCENARIOS = ("scalar_s_curve", "scalar_saddle")
LAMBDA_VARIANTS = (0.0, 1.0)  # 1.0 matches the frozen vector-field M6 protocol; 0.0 = no covariance blend
# lambda_v_confidence_scaling="power" exponents to probe, on top of the flat
# lambda_v in {0,1} comparison above -- see VelocityManifoldFitter's
# lambda_v_confidence_scaling docstring. Not a selection: reported as a
# diagnostic showing the mechanism's effect size, not used to pick a winner.
CONFIDENCE_POWERS = (1.0, 2.0, 4.0, 8.0, 16.0)
OUT_DIR = ROOT / "results" / "p4_1_scalar_oracle_ablation"


def position_metrics(name: str, Xhat: np.ndarray, clean_P: np.ndarray) -> dict:
    valid = np.ones(len(Xhat), dtype=bool)
    return {
        "clean_point_rmse": float(np.sqrt(np.mean(np.sum((Xhat - clean_P) ** 2, axis=1)))),
        "distance_to_manifold": distance_to_manifold(name, Xhat),
    }


def gradient_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict:
    valid = np.ones(len(estimate), dtype=bool)
    return {
        "gradient_rmse": vector_rmse(estimate, truth, valid),
        "gradient_angle_mae": angle_mae(estimate, truth, valid),
    }


def evaluate_condition(name: str, seed: int, k: int) -> list[dict]:
    data = scalar_data(name, seed)
    Y, scalar_obs, truth, clean_P = data["Y"], data["scalar"], data["truth"], data["P"]
    rows = []

    # (a) Raw local regression alone -- no manifold fitting at all.
    grad_hat_raw = estimate_gradient_from_neighbors(Y, scalar_obs, n_neighbors=k, ridge=5e-2)
    rows.append(
        {
            "scenario": name, "seed": seed, "k": k, "pipeline": "raw_local_regression",
            "lambda_v": None, "gradient_source": "estimated",
            "clean_point_rmse": float("nan"), "distance_to_manifold": float("nan"),
            **gradient_metrics(grad_hat_raw, truth),
        }
    )

    # (b)-(e): estimated vs oracle gradient source, crossed with lambda_v in {0, 1}.
    for gradient_source in ("estimated", "oracle"):
        for lambda_v in LAMBDA_VARIANTS:
            kwargs = dict(k=k, lambda_v=lambda_v)
            if lambda_v > 0:
                kwargs["velocity_covariance_mode"] = "uncentered"
            if gradient_source == "oracle":
                kwargs["oracle_gradient"] = truth
            result = fit_scalar_gradient_manfit(Y, scalar_obs, **kwargs)
            rows.append(
                {
                    "scenario": name, "seed": seed, "k": k,
                    "pipeline": f"{gradient_source}_lambda{lambda_v}",
                    "lambda_v": lambda_v, "gradient_source": gradient_source,
                    "lambda_v_confidence_scaling": "none", "lambda_v_confidence_power": None,
                    **position_metrics(name, result["position"], clean_P),
                    **gradient_metrics(result["gradient"], truth),
                }
            )

    # (f) Confidence-scaled lambda_v (2026-08-12 follow-up, same day as the
    # oracle_gradient ablation): discount lambda_v=1.0 per point by
    # confidence**power instead of applying it flatly, on the "estimated"
    # gradient source only (oracle confidence is uniformly 1.0 by
    # construction, so scaling is a no-op there -- not run for oracle).
    # Kept as a validated, documented alternative even after (g) below was
    # added -- not deleted, see current_plan.md P4.1 follow-up "redesign" note.
    for power in CONFIDENCE_POWERS:
        result = fit_scalar_gradient_manfit(
            Y, scalar_obs, k=k, lambda_v=1.0, velocity_covariance_mode="uncentered",
            lambda_v_confidence_scaling="power", lambda_v_confidence_power=power,
        )
        rows.append(
            {
                "scenario": name, "seed": seed, "k": k,
                "pipeline": f"estimated_lambda1.0_power{power}",
                "lambda_v": 1.0, "gradient_source": "estimated",
                "lambda_v_confidence_scaling": "power", "lambda_v_confidence_power": power,
                **position_metrics(name, result["position"], clean_P),
                **gradient_metrics(result["gradient"], truth),
            }
        )

    # (g) "inverse_error" (2026-08-12, same day, added after user review of
    # (f): "power" needs its own separately-tuned shape hyperparameter,
    # which does not match the original intent of a decreasing function of
    # the error itself). lambda_v_effective = lambda_v / (1 +
    # relative_error), relative_error = the local regression's own
    # ss_res/ss_tot, already computed inside estimate_gradient_confidence_
    # from_neighbors -- no new estimation, no extra tunable shape parameter.
    result = fit_scalar_gradient_manfit(
        Y, scalar_obs, k=k, lambda_v=1.0, velocity_covariance_mode="uncentered",
        lambda_v_confidence_scaling="inverse_error",
    )
    rows.append(
        {
            "scenario": name, "seed": seed, "k": k,
            "pipeline": "estimated_lambda1.0_inverse_error",
            "lambda_v": 1.0, "gradient_source": "estimated",
            "lambda_v_confidence_scaling": "inverse_error", "lambda_v_confidence_power": None,
            **position_metrics(name, result["position"], clean_P),
            **gradient_metrics(result["gradient"], truth),
        }
    )

    # (h) "rank" (2026-08-12, same day, direct response to the user pointing
    # out that (f)'s "best" power=16 was never actually tuning-seed
    # selected -- just the smallest value in an exploratory grid evaluated
    # on final seeds). lambda_v_effective = lambda_v * (1 -
    # percentile_rank(relative_error)) -- zero free parameters like (g), but
    # invariant to the absolute numeric scale of relative_error (only
    # within-batch ordering matters), which is what made (g) weak.
    result = fit_scalar_gradient_manfit(
        Y, scalar_obs, k=k, lambda_v=1.0, velocity_covariance_mode="uncentered",
        lambda_v_confidence_scaling="rank",
    )
    rows.append(
        {
            "scenario": name, "seed": seed, "k": k,
            "pipeline": "estimated_lambda1.0_rank",
            "lambda_v": 1.0, "gradient_source": "estimated",
            "lambda_v_confidence_scaling": "rank", "lambda_v_confidence_power": None,
            **position_metrics(name, result["position"], clean_P),
            **gradient_metrics(result["gradient"], truth),
        }
    )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in SCENARIOS:
        first = scalar_data(name, FINAL_SEEDS[0])
        k = neighbor_count(len(first["Y"]), first["d"])
        for seed in FINAL_SEEDS:
            rows.extend(evaluate_condition(name, seed, k))

    long_frame = pd.DataFrame(rows)
    long_frame.to_csv(OUT_DIR / "p4_1_long.csv", index=False)

    summary = (
        long_frame.groupby(["scenario", "pipeline", "lambda_v", "gradient_source"], dropna=False)
        .agg(
            k=("k", "first"),
            median_clean_point_rmse=("clean_point_rmse", "median"),
            median_distance_to_manifold=("distance_to_manifold", "median"),
            median_gradient_rmse=("gradient_rmse", "median"),
            median_gradient_angle_mae=("gradient_angle_mae", "median"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "p4_1_summary.csv", index=False)

    # Headline decomposition, per scenario, at the frozen lambda_v=1.0:
    # raw local-regression error vs. estimated-pipeline vs. oracle-pipeline.
    decomposition = []
    for name in SCENARIOS:
        sub = summary[summary.scenario == name]
        raw = sub[sub.pipeline == "raw_local_regression"].iloc[0]
        est = sub[sub.pipeline == "estimated_lambda1.0"].iloc[0]
        oracle = sub[sub.pipeline == "oracle_lambda1.0"].iloc[0]
        decomposition.append(
            {
                "scenario": name,
                "raw_local_regression_gradient_rmse": raw.median_gradient_rmse,
                "estimated_pipeline_position_rmse": est.median_clean_point_rmse,
                "oracle_pipeline_position_rmse": oracle.median_clean_point_rmse,
                "position_rmse_gap_attributable_to_gradient_estimation": est.median_clean_point_rmse - oracle.median_clean_point_rmse,
                "estimated_pipeline_gradient_rmse": est.median_gradient_rmse,
                "oracle_pipeline_gradient_rmse": oracle.median_gradient_rmse,
            }
        )
    (OUT_DIR / "p4_1_decomposition.json").write_text(json.dumps(decomposition, indent=2, default=str))

    print(json.dumps(decomposition, indent=2, default=str))
    print(f"\nWrote {OUT_DIR}/p4_1_long.csv, p4_1_summary.csv, p4_1_decomposition.json")


if __name__ == "__main__":
    main()
