"""Scalar-branch lambda_v selection (tuning seeds only), with the scaling MODE
already fixed to "rank" by explicit user decision (2026-08-12, current_plan.md P4.1
follow-up, third iteration) -- this script does not re-litigate mode choice, it
only selects the numeric lambda_v magnitude to pair with "rank".

Mirrors `run_lambda_sensitivity.py`'s own selection procedure for the
vector-field lambda_v (Round 5, see log.md): sweep a small grid on TUNING_SEEDS
only, pool a headline score across scenarios, and apply a safeguard that no
candidate may score worse than the lambda_v=0 baseline on ANY single scenario
-- final seeds never enter the selection.

Why this step exists even though "rank" itself needed no tuning: every
`rank`/`power`/`inverse_error` comparison run so far (see current_plan.md) used
lambda_v=1.0 unconditionally, copied from the vector-field M6 frozen value,
never actually chosen for the scalar branch. It's already known that
lambda_v=1.0 + rank underperforms the safe lambda_v=0 baseline on
scalar_saddle -- freezing that combination without checking whether some
other lambda_v magnitude does better (or whether 0 really is the honest
answer) would mean freezing a config already known to be worse than doing
nothing.

`scalar_s_curve` is included for parity with the scenario set used everywhere
else in P4.1, even though it's known to be flat/uninformative for lambda_v
(z=0 degenerate geometry) -- it will not tip the pooled score either way, but
omitting it would be an unexplained scope narrowing.

UPDATE (2026-08-12, same day, parameter_rules.md SS3c): theta/kappa now reuse
the vector-field M6's own frozen shared values (0.02, 0.0) rather than
fit_scalar_gradient_manfit's own standalone defaults, by direct user
decision (reuse ManfitVelo's neighbor-reranking strength rather than a
separate tier-3 search for it). inner_T/eta_g are deliberately NOT copied
from M6's T=3/eta_g=0.7 -- caught empirically first: since
fit_scalar_gradient_manfit calls fit() outer_iterations=4 times (each with
inner_T steps), copying eta_g=0.7 there means 4x the position-update budget
at M6's own aggressive per-step size, which overshoots badly (scalar_saddle's
lambda_v=0 safe baseline went from clearly-helping at 0.0204 to
no-better-than-raw-noise at ~0.05). inner_T/eta_g stay at their own
already-used values (2, 0.35) -- only the neighbor-reranking parameters
(theta, kappa) are shared, since those are what the "reuse ManfitVelo's
settings" consistency argument was actually about (S1/S2's own headline
finding traced the joint_scalar_aware-vs-geometry_only gap to exactly this
mechanism) -- see SHARED_KWARGS below.

    python simulation/run_scalar_lambda_v_selection.py
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

from scripts.benchmark_scenarios import scalar_data  # noqa: E402
from scripts.scalar_potential_manfit import fit_scalar_gradient_manfit  # noqa: E402
from simulation.benchmark_core import neighbor_count  # noqa: E402
from simulation.run_manfitvelo_benchmark import TUNING_SEEDS, FINAL_SEEDS  # noqa: E402

SCENARIOS = ("scalar_s_curve", "scalar_saddle")
LAMBDA_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)
SCALING = "rank"  # fixed by user decision, not re-selected here

# theta/kappa reused directly from the vector-field M6's own frozen shared
# values (parameter_rules.md SS3c, 2026-08-12 user decision) rather than a
# separate tier-3 search for the scalar branch's neighbor-reranking strength.
# inner_T/eta_g deliberately NOT copied from M6's T=3/eta_g=0.7 -- caught
# empirically that doing so overshoots badly, since fit_scalar_gradient_
# manfit's outer_iterations=4 loop means 4x the position-update budget at
# that eta_g compared to M6's single fit() call (see module docstring).
# inner_T/eta_g stay at fit_scalar_gradient_manfit's own defaults (2, 0.35).
SHARED_KWARGS = dict(theta=0.02, kappa=0.0)

OUT_DIR = ROOT / "results" / "scalar_lambda_v_selection"


def clean_point_rmse(Xhat: np.ndarray, clean_P: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((Xhat - clean_P) ** 2, axis=1))))


def evaluate(scenario: str, lambda_v: float, seeds: tuple[int, ...]) -> list[dict]:
    rows = []
    for seed in seeds:
        data = scalar_data(scenario, seed)
        Y, scalar_obs, clean_P = data["Y"], data["scalar"], data["P"]
        k = neighbor_count(len(Y), data["d"])
        kwargs = dict(k=k, lambda_v=lambda_v, lambda_v_confidence_scaling=SCALING, **SHARED_KWARGS)
        if lambda_v > 0:
            kwargs["velocity_covariance_mode"] = "uncentered"
        result = fit_scalar_gradient_manfit(Y, scalar_obs, **kwargs)
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "lambda_v": lambda_v,
                "clean_point_rmse": clean_point_rmse(result["position"], clean_P),
            }
        )
    return rows


def select_lambda_v(frame: pd.DataFrame) -> tuple[float, pd.DataFrame, bool]:
    """Pick lambda_v by lowest pooled mean log(clean_point_rmse) across scenarios
    (tuning seeds only), subject to a safeguard: no candidate may have a worse
    per-scenario mean clean_point_rmse than the lambda_v=0 baseline for that
    same scenario. Mirrors select_lambda_v() in run_lambda_sensitivity.py.
    """
    per_scenario_mean = frame.groupby(["lambda_v", "scenario"]).clean_point_rmse.mean().unstack()
    baseline = per_scenario_mean.loc[0.0]
    safe = per_scenario_mean.le(baseline).all(axis=1)
    pooled = (
        frame.assign(log_rmse=np.log(frame.clean_point_rmse))
        .groupby("lambda_v")
        .log_rmse.mean()
        .reset_index()
        .rename(columns={"log_rmse": "pooled_log_clean_point_rmse"})
    )
    pooled["safe_for_every_scenario"] = pooled.lambda_v.map(safe)
    candidates = pooled[pooled.safe_for_every_scenario]
    safeguard_triggered = candidates.empty or len(candidates) < len(pooled)
    if candidates.empty:
        candidates = pooled
    best = float(candidates.loc[candidates.pooled_log_clean_point_rmse.idxmin(), "lambda_v"])
    return best, pooled.sort_values("pooled_log_clean_point_rmse"), safeguard_triggered


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tuning_rows = []
    for scenario in SCENARIOS:
        for lam in LAMBDA_GRID:
            tuning_rows.extend(evaluate(scenario, lam, TUNING_SEEDS))
    tuning_frame = pd.DataFrame(tuning_rows)
    tuning_frame.to_csv(OUT_DIR / "tuning_seed_grid.csv", index=False)

    best_lambda_v, pooled, safeguard_triggered = select_lambda_v(tuning_frame)
    pooled.to_csv(OUT_DIR / "tuning_seed_selection_audit.csv", index=False)

    print(f"Grid: {LAMBDA_GRID}, scaling fixed to {SCALING!r}, seeds: TUNING_SEEDS={TUNING_SEEDS} only\n")
    print(pooled.to_string(index=False))
    print(f"\nSelected lambda_v = {best_lambda_v} (safeguard excluded any regressing candidate: {safeguard_triggered})")
    if best_lambda_v == 0.0:
        print(
            "\nNOTE: the winner is lambda_v=0.0 -- i.e. the covariance-blend mechanism itself "
            "(regardless of scaling mode) does not help on these scenarios under the current "
            "gradient-estimation quality. This is a legitimate, reportable outcome, not a bug."
        )

    # Confirmatory-only report on final seeds, for the frozen (lambda_v, "rank") pair and the
    # lambda_v=0 safe baseline -- never used for selection.
    final_rows = []
    for scenario in SCENARIOS:
        final_rows.extend(evaluate(scenario, 0.0, FINAL_SEEDS))
        if best_lambda_v != 0.0:
            final_rows.extend(evaluate(scenario, best_lambda_v, FINAL_SEEDS))
    final_frame = pd.DataFrame(final_rows)
    final_frame.to_csv(OUT_DIR / "final_seed_confirmation.csv", index=False)
    final_summary = final_frame.groupby(["scenario", "lambda_v"]).clean_point_rmse.median().reset_index()
    print("\nFinal-seed confirmatory medians (reporting only, not a selection step):")
    print(final_summary.to_string(index=False))

    audit = {
        "scaling": SCALING,
        "grid": list(LAMBDA_GRID),
        "selected_lambda_v": best_lambda_v,
        "safeguard_triggered": bool(safeguard_triggered),
        "selection_uses_final_seeds": False,
    }
    (OUT_DIR / "sanity_checks.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"\nWrote {OUT_DIR}/tuning_seed_grid.csv, tuning_seed_selection_audit.csv, "
          f"final_seed_confirmation.csv, sanity_checks.json")


if __name__ == "__main__":
    main()
