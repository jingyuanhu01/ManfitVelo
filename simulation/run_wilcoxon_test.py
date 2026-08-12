"""P5: paired Wilcoxon signed-rank test for M5-vs-M6 tie / thin-margin scenarios.

current_plan.md P5's freeze checklist and its "Claim language review" list (SS5,
point 2) both flag this as the one thing standing between the report and a
defensible "ManfitVelo (M6) beats Position-only MANFIT (M5)" claim: several
scenario/metric pairs have thin median margins or split seed-level verdicts,
so a plain "X/9 scenarios" or "M6 wins on scenario Y" claim needs a real
significance test, not just eyeballing medians.

Tests the pairs the plan explicitly names, reading directly from the
already-frozen `results/manfitvelo_benchmark/final_seed_metrics.csv` (no
recomputation -- see log.md's P5-scoping decision, 2026-08-12: nothing here
needs rerunning, the data is already current under the frozen protocol):

    circle, G1 (distance_to_manifold_rel)
    circle, G2 (clean_point_rmse_rel)
    flat_rotation_annulus, V3 (velocity_rmse_loc_rel)
    swiss_roll, G1 (distance_to_manifold_rel)
    swiss_roll, G2 (clean_point_rmse_rel)  -- added: this is the metric that
        actually flipped median-favoring-M5 after the C=0.60 rerun (P0.1
        section), the most consequential of the five pairs to test, even
        though the original planning-time checklist named swiss_roll's G1
        rather than G2.

G1/G2/V3 labels match `metric_definitions.md`. Paired by seed (15 final
seeds, M5 vs M6 on the same noisy draw), using the *_rel metrics (each
scenario/seed's own value divided by that seed's own ambient-noisy
baseline) rather than raw metrics, so the seed pairing also cancels
per-seed noise-level variation -- matches how every relative-metric claim
elsewhere in this pipeline is framed.

Reports both a two-sided test (is there any difference at all) and a
one-sided test with alternative="less" (does M6 score lower / better than
M5), since the pipeline's own headline claim is directional -- reporting
both rather than only the one that looks favorable. zero_method="pratt"
(includes zero-difference pairs in the ranking rather than dropping them,
the generally-recommended default over the classic "wilcox" behavior).

    python simulation/run_wilcoxon_test.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METRIC_LABELS = {
    "G1": "distance_to_manifold_rel",
    "G2": "clean_point_rmse_rel",
    "V3": "velocity_rmse_loc_rel",
}

# (scenario, metric_label) pairs to test -- see module docstring for why
# swiss_roll G2 was added alongside the originally-flagged G1.
PAIRS = [
    ("circle", "G1"),
    ("circle", "G2"),
    ("flat_rotation_annulus", "V3"),
    ("swiss_roll", "G1"),
    ("swiss_roll", "G2"),
]

SOURCE = ROOT / "results" / "manfitvelo_benchmark" / "final_seed_metrics.csv"
OUT_DIR = ROOT / "results" / "wilcoxon_test"


def paired_values(frame: pd.DataFrame, scenario: str, metric_col: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    sub = frame[(frame.scenario == scenario) & (frame.method.isin(["position_only_manfit", "manfitvelo"]))]
    pivot = sub.pivot(index="seed", columns="method", values=metric_col).sort_index()
    if pivot.isna().any().any():
        raise ValueError(f"missing seed(s) for {scenario}/{metric_col}: {pivot[pivot.isna().any(axis=1)]}")
    return pivot["position_only_manfit"].to_numpy(), pivot["manfitvelo"].to_numpy(), list(pivot.index)


def run_test(scenario: str, label: str) -> dict:
    metric_col = METRIC_LABELS[label]
    frame = pd.read_csv(SOURCE)
    m5, m6, seeds = paired_values(frame, scenario, metric_col)
    difference = m6 - m5  # negative => M6 better (lower relative error/metric)
    wins_m6 = int((difference < 0).sum())
    ties = int((difference == 0).sum())
    n = len(difference)

    two_sided = wilcoxon(m6, m5, zero_method="pratt", alternative="two-sided")
    one_sided_m6_better = wilcoxon(m6, m5, zero_method="pratt", alternative="less")

    return {
        "scenario": scenario,
        "metric_label": label,
        "metric_column": metric_col,
        "n_seeds": n,
        "m5_median": float(np.median(m5)),
        "m6_median": float(np.median(m6)),
        "median_paired_difference": float(np.median(difference)),
        "m6_wins": wins_m6,
        "m5_wins": n - wins_m6 - ties,
        "ties": ties,
        "two_sided_statistic": float(two_sided.statistic),
        "two_sided_p_value": float(two_sided.pvalue),
        "one_sided_m6_better_statistic": float(one_sided_m6_better.statistic),
        "one_sided_m6_better_p_value": float(one_sided_m6_better.pvalue),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        raise FileNotFoundError(f"{SOURCE} not found -- run simulation/run_manfitvelo_benchmark.py first")

    results = [run_test(scenario, label) for scenario, label in PAIRS]
    frame = pd.DataFrame(results)
    frame.to_csv(OUT_DIR / "wilcoxon_results.csv", index=False)
    (OUT_DIR / "wilcoxon_results.json").write_text(json.dumps(results, indent=2) + "\n")

    print(frame.to_string(index=False))
    print(f"\nWrote {OUT_DIR}/wilcoxon_results.csv, wilcoxon_results.json")
    print(f"Source (not recomputed): {SOURCE}")


if __name__ == "__main__":
    main()
