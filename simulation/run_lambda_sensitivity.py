"""Lambda_v selection (tuning seeds) and confirmatory sensitivity curve (final seeds).

ManfitVelo's lambda_v controls how much weight the velocity second-moment matrix gets
when it's blended into the local position covariance before the tangent-space
eigendecomposition (`VelocityManifoldFitter._compute_local_tangent`) -- this is the one
parameter that actually implements "use velocity to improve manifold recovery." It was
originally a frozen constant (0.1), carried forward unchanged from a dedicated prior study
(archive/simulation/results_legacy/velocity_augmented_main_benchmark_20260717/) that used
the pre-fairness-fix, pre-curvature-aware-k, 7-scenario protocol.

Note lambda_v=0 does NOT numerically reduce VMF to Position-only MANFIT (M5): M5's
`position_only_trajectory` is an independent implementation using plain Euclidean kNN
frozen at t=0 (never invokes VMF at all), whereas VMF's own `_build_neighbors` reranks
candidates by a velocity/theta-aware distance regardless of lambda_v -- so even at
lambda_v=0, VMF's neighbor selection still uses velocity information; only the
tangent-covariance blending step is switched off. This sweep is therefore the true
single-parameter isolated ablation (everything else -- k, T, eta_g, theta, kappa,
theta_schedule, neighbor selection -- fixed at M6's own canonical values, only lambda_v
varies), complementing (not duplicating) the separate M4-to-M5-to-M6 pipeline-capability
ablation, which compares three different implementations rather than one parameter.

Two-stage protocol (2026-08-11 revision, see log.md Round 5):
  1. --seeds tuning: re-run the archived study's grid {0,0.1,0.25,0.5,1,2} under the
     CURRENT protocol (9 scenarios, curvature-aware k, pooled fair shared hyperparameters)
     using ONLY the 3 tuning seeds, pooled the same way tune_shared_vmf() already pools
     T/eta_g/theta/kappa/theta_schedule (mean tuning_score across all 9 scenarios x 3
     tuning seeds). This is the legitimate selection basis -- final seeds are NEVER used to
     choose a parameter anywhere in this pipeline (selection_uses_final_seeds asserted
     False throughout), so the original --seeds final run (which showed lambda_v=0.1 was
     conservative for 8/9 scenarios) could inform *that a re-check was worth doing* but
     could not itself be used to pick a new default.
     The naive pooled-mean argmin was lambda_v=2.0, but that candidate makes Swiss Roll
     score WORSE than its own lambda_v=0 baseline (0.783 vs 0.741 on clean_point_rmse_rel,
     tuning seeds) -- exactly the "single local flow family does not reliably reinforce
     every weak tangent direction" failure mode the archived study's own safeguard was
     designed to catch. Adding that safeguard back (candidate must not regress any single
     scenario below its own lambda_v=0 baseline) rules out 2.0 and selects lambda_v=1.0,
     which has the best pooled score among the scenario-safe candidates -- see
     select_lambda_v() and lambda_selection_audit.csv.
  2. Update FROZEN_DEFAULT_LAMBDA_V here and the matching constant in
     run_manfitvelo_benchmark.shared_vmf_grid(), rerun the canonical benchmark and all
     scans with the new default.
  3. --seeds final (default): confirmatory sensitivity curve around the new default, for
     the report -- reporting only, not a second selection step.

    python simulation/run_lambda_sensitivity.py --seeds tuning   # selection
    python simulation/run_lambda_sensitivity.py                  # confirmatory curve (final seeds)
    python simulation/run_lambda_sensitivity.py --report-only
"""

from __future__ import annotations

import argparse
import base64
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import simulation.run_manfitvelo_benchmark as m  # noqa: E402
from simulation.benchmark_core import SCENARIO_LABELS, load_frozen_config  # noqa: E402

SCENARIOS = m.SCENARIOS
TUNING_SEEDS = m.TUNING_SEEDS
FINAL_SEEDS = m.FINAL_SEEDS
FROZEN_DEFAULT_LAMBDA_V = 1.0  # updated 2026-08-11; see log.md Round 5 for the tuning-seed selection
LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)  # matches the archived prior study's grid
HEADLINE_METRICS = (
    "clean_point_rmse_rel",
    "distance_to_manifold_rel",
    "velocity_rmse_loc_rel",
    "joint_euler_state_rmse_rel",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", choices=("tuning", "final"), default="final")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def evaluate_lambda(scenario: str, lambda_v: float, base_selected: dict, seeds: tuple[int, ...]) -> pd.DataFrame:
    cfg = dict(base_selected["velocity_manifold_fitter"][scenario])
    cfg["lambda_v"] = float(lambda_v)
    rows = []
    for seed in seeds:
        data = m.vector_data(scenario, seed)
        k = int(base_selected["shared_graph_k"][scenario])
        graph = m.shared_knn_graph(data["Y"], k)
        tau, _, _ = m.observed_tau(data["Y"], data["field"], graph)
        baseline = m.state_metrics(scenario, data["Y"], data["field"], data, tau)
        result = m.fit_vmf_variant(data["Y"], data["field"], data["d"], cfg, seed)
        absolute = m.state_metrics(scenario, result["X"], result["V"], data, tau)
        relative = m.relative_state_metrics(absolute, baseline)
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "lambda_v": lambda_v,
                "tuning_score": m.tuning_score(relative),
                **absolute,
                **relative,
            }
        )
    return pd.DataFrame(rows)


def run(base_selected: dict, seeds: tuple[int, ...]) -> pd.DataFrame:
    frames = [evaluate_lambda(scenario, lam, base_selected, seeds) for scenario in SCENARIOS for lam in LAMBDA_GRID]
    return pd.concat(frames, ignore_index=True)


def select_lambda_v(frame: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Pick lambda_v by lowest mean tuning_score pooled over all scenarios x tuning seeds --
    the same pooling rule tune_shared_vmf() already uses for T/eta_g/theta/kappa/theta_schedule --
    subject to a safeguard: no candidate may score worse (higher tuning_score) than the
    lambda_v=0 baseline for ANY single scenario. Without this safeguard the naive pooled argmin
    picks the largest grid value every time a majority of scenarios keep improving, even if a
    minority regress arbitrarily badly (exactly what happens at lambda_v=2.0, which makes Swiss
    Roll worse than doing nothing) -- the same failure mode the archived prior study's own
    multi-criterion selection (benefit_count + worst-case ratio) was designed to catch.
    """
    baseline = frame[frame.lambda_v == 0.0].groupby("scenario").tuning_score.mean()
    per_scenario = frame.groupby(["lambda_v", "scenario"]).tuning_score.mean().unstack()
    safe = per_scenario.le(baseline).all(axis=1)
    pooled = frame.groupby("lambda_v").tuning_score.mean().reset_index()
    pooled["safe_for_every_scenario"] = pooled.lambda_v.map(safe)
    candidates = pooled[pooled.safe_for_every_scenario]
    if candidates.empty:
        candidates = pooled
    best = float(candidates.loc[candidates.tuning_score.idxmin(), "lambda_v"])
    return best, pooled.sort_values("tuning_score")


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, lam), g in frame.groupby(["scenario", "lambda_v"], sort=False):
        row = {"scenario": scenario, "lambda_v": lam}
        for metric in HEADLINE_METRICS:
            row[f"{metric}_median"] = float(g[metric].median())
            row[f"{metric}_q25"] = float(g[metric].quantile(0.25))
            row[f"{metric}_q75"] = float(g[metric].quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric(summary: pd.DataFrame, metric: str, output: Path, marker_lambda: float, marker_label: str) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    for ax, scenario in zip(axes.ravel(), SCENARIOS):
        sub = summary[summary.scenario == scenario].sort_values("lambda_v")
        ax.plot(sub.lambda_v, sub[f"{metric}_median"], marker="o", color="#1f6f5c")
        ax.fill_between(sub.lambda_v, sub[f"{metric}_q25"], sub[f"{metric}_q75"], color="#1f6f5c", alpha=0.15)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="ambient noisy input")
        ax.axvline(marker_lambda, color="#b8860b", linewidth=1.4, linestyle=":", label=marker_label)
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=10)
        ax.set_xlabel("lambda_v", fontsize=8)
        ax.tick_params(labelsize=7)
    axes.ravel()[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{metric} vs. lambda_v")
    path = output / "figures" / f"lambda_{metric}.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


class AuditParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images: list[str] = []
        self.external = False

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            for key, value in attrs:
                if key == "src":
                    self.images.append(value)
                    if not value.startswith("data:image/png;base64,"):
                        self.external = True


def build_report(
    output: Path, summary: pd.DataFrame, seeds_kind: str, marker_lambda: float, regenerate_figures: bool = True
) -> dict:
    style = "body{margin:0;background:#f4f6f8;color:#17212b;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}main{max-width:1400px;margin:auto;padding:28px 20px 60px}.card{background:#fff;border:1px solid #d8dee7;border-radius:9px;padding:20px;margin:18px 0}img{max-width:100%;height:auto}p{line-height:1.55}"
    (output / "figures").mkdir(parents=True, exist_ok=True)
    marker_label = "selected default" if seeds_kind == "tuning" else "frozen default"
    sections = []
    for metric in HEADLINE_METRICS:
        path = (
            plot_metric(summary, metric, output, marker_lambda, marker_label)
            if regenerate_figures
            else output / "figures" / f"lambda_{metric}.png"
        )
        sections.append(f"<section class='card'><h2>{metric}</h2><img src='{image_uri(path)}'></section>")
    if seeds_kind == "tuning":
        intro = (
            "<h1>lambda_v selection (tuning seeds)</h1>"
            f"<section class='card'><p>Sweeps lambda_v &isin; {LAMBDA_GRID} on the 3 development "
            "(tuning) seeds only, everything else frozen at the canonical shared values. This IS the "
            "selection basis (pooled mean tuning_score across all 9 scenarios &times; 3 tuning seeds, "
            "same rule as the T/eta_g/theta/kappa/theta_schedule search). Final seeds never enter this "
            f"computation.</p></section>"
        )
    else:
        intro = (
            "<h1>lambda_v confirmatory sensitivity curve (final seeds)</h1>"
            f"<section class='card'><p>Sweeps lambda_v &isin; {LAMBDA_GRID} on all 15 final seeds with "
            f"the now-frozen default (lambda_v={marker_lambda:g}, dotted gold line) marked for "
            "reference. Reporting only, not a selection step: the default was already chosen from the "
            "tuning-seed run.</p></section>"
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>ManfitVelo lambda_v {seeds_kind}</title><style>{style}</style></head><body><main>"
        + intro
        + "".join(sections)
        + "</main></body></html>"
    )
    report = output / f"lambda_sensitivity_{seeds_kind}_report.html"
    report.write_text(html, encoding="utf-8")
    parser = AuditParser()
    parser.feed(html)
    return {
        "self_contained_html": all(v.startswith("data:image/png;base64,") for v in parser.images) and not parser.external,
        "embedded_figure_count": len(parser.images),
        "expected_figure_count": len(HEADLINE_METRICS),
        "seeds_kind": seeds_kind,
        "marker_lambda": marker_lambda,
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir or (ROOT / f"results/lambda_sensitivity_{args.seeds}")
    output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        summary = pd.read_csv(output / "summary_metrics.csv")
        audit = build_report(output, summary, args.seeds, FROZEN_DEFAULT_LAMBDA_V, regenerate_figures=False)
        (output / "sanity_checks.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(output / f"lambda_sensitivity_{args.seeds}_report.html")
        return

    base_selected = load_frozen_config()
    seeds = TUNING_SEEDS if args.seeds == "tuning" else FINAL_SEEDS
    frame = run(base_selected, seeds)
    frame.to_csv(output / "lambda_seed_metrics.csv", index=False)

    marker_lambda = FROZEN_DEFAULT_LAMBDA_V
    if args.seeds == "tuning":
        best, pooled = select_lambda_v(frame)
        pooled.to_csv(output / "lambda_selection_audit.csv", index=False)
        marker_lambda = best
        print(f"selected lambda_v = {best} (pooled tuning_score, tuning seeds only)")
        print(pooled.to_string(index=False))

    summary = summarize(frame)
    summary.to_csv(output / "summary_metrics.csv", index=False)
    audit = build_report(output, summary, args.seeds, marker_lambda, regenerate_figures=True)
    audit["selection_uses_final_seeds"] = False if args.seeds == "tuning" else "n/a (confirmatory only)"
    (output / "sanity_checks.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(output / f"lambda_sensitivity_{args.seeds}_report.html")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
