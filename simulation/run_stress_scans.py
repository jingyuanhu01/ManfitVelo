"""Stress-test scans: Scan A (sample size n), Scan B (position noise sigma_X),
Scan C (velocity noise sigma_V).

Weekly Plan v1.1 section 10. Reuses the already-frozen shared VMF / Position-only
MANFIT hyperparameters from results/manfitvelo_benchmark/selected_hyperparameters.json
(tier-3 parameters -- meant to stay frozen across a scan) but recomputes k(n,d) and its
curvature-aware refinement fresh at every scan point from that point's own
development-seed draws (tier-2 data-adaptive rule). This is a hard requirement, not an
implementation convenience -- see parameter_rules.md section 7: reusing the canonical
setting's k across different n/noise levels would silently reintroduce
scenario(-condition)-specific tuning through the back door. (Scan C is the one exception
that's actually principled rather than an oversight: k only depends on position
observations, so it provably reproduces the canonical value when n and sigma_X are held
at canonical -- see evaluate_condition's docstring comment.)

Scan B multipliers are relative to each scenario's own canonical sigma_X (which already
varies scenario-to-scenario, e.g. 0.02 for Near Intersection vs 0.05 for Circle) so every
scenario is stressed by a comparable *relative* amount.

Scan C (redesigned 2026-08-12, current_plan.md P1.1): the original absolute grid
sigma_V in {0.05,...,0.30} was too mild to find a real velocity-noise breakdown point (M6
geometry barely moved -- see log.md Round 6/current_plan.md P1.1 for the pre-redesign numbers).
Replaced with a *relative* grid r_V = sigma_V / median||V_true|| in
{0.05, 0.1, 0.2, 0.4, 0.8, 1.6}, converted per scenario via SCENARIO_VELOCITY_SCALE (median
clean-field speed pooled over TUNING_SEEDS -- development seeds only, computed once, not a
selection), plus one additional non-numeric "shuffled velocity" negative control per
scenario (row-permuted noisy velocity at the scenario's own canonical sigma_V -- same noise
magnitude, but no longer correlated with position at all). Scan C also reports
tangent/normal-component decomposition (mechanism_metrics/true_projector, reused from
run_manfitvelo_benchmark.py -- already scenario-general, no new per-scenario ground truth
needed) for Position-only MANFIT (M5), ManfitVelo (M6), and Local PCA (M4), not just the
headline relative state metrics Scan A/B report.

    python simulation/run_stress_scans.py                    # full run, all three scans
    python simulation/run_stress_scans.py --scans C           # only (re)run Scan C, keep existing A/B rows
    python simulation/run_stress_scans.py --report-only
"""

from __future__ import annotations

import argparse
import base64
import copy
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import sys
import time

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
from simulation.benchmark_core import (  # noqa: E402
    SCENARIO_LABELS,
    curvature_aware_neighbor_count,
    curvature_probe_k_grid,
    load_frozen_config,
    local_pca_normal_residual,
)

TUNING_SEEDS = m.TUNING_SEEDS
FINAL_SEEDS = m.FINAL_SEEDS
SCENARIOS = m.SCENARIOS
METHODS = m.BASE_METHODS + ("manfitvelo",)
METHOD_LABELS = m.BASE_LABELS

SCAN_A_N_VALUES = (200, 400, 800, 1600)
SCAN_B_MULTIPLIERS = (0.5, 1.0, 1.5, 2.0, 3.0)
# Old absolute grid, superseded 2026-08-12 by the relative grid below (current_plan.md P1.1) --
# kept only as a comment for provenance; results/stress_scans/scan_seed_metrics.csv rows
# with scan="C_velocity_noise" from before this date used this grid.
_SCAN_C_SIGMA_V_VALUES_LEGACY = (0.05, 0.10, 0.15, 0.20, 0.30)
SCAN_C_RELATIVE_SIGMA_V = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)

MECHANISM_METHODS = ("position_only_manfit", "manfitvelo", "local_pca")

HEADLINE_METRICS = (
    "clean_point_rmse_rel",
    "distance_to_manifold_rel",
    "velocity_rmse_loc_rel",
    "joint_euler_state_rmse_rel",
)
MECHANISM_METRICS = ("tangential_component_rmse", "normal_component_rmse")
COMPARISON_METHODS = ("position_only_manfit", "manfitvelo", "local_pca")


def scenario_velocity_scale(scenario: str) -> float:
    """Median clean-field speed for a scenario, pooled over TUNING_SEEDS only
    (development seeds; never final seeds). Used solely to size Scan C's
    relative noise grid -- a fixed reference constant, not a selection."""
    speeds = []
    for seed in TUNING_SEEDS:
        data = m.vector_data(scenario, seed)
        speeds.append(np.linalg.norm(data["truth"], axis=1))
    return float(np.median(np.concatenate(speeds)))


def shuffle_velocity_field(field: np.ndarray, seed: int) -> np.ndarray:
    """Row-permute a noisy velocity field, independent of the RNG stream used
    to generate the data itself. Breaks any correspondence between a point's
    position and the velocity observed there, while preserving the noise
    magnitude distribution exactly (same values, different assignment) --
    a genuine "velocity carries zero geometric information" negative control,
    sharper than just increasing sigma_V."""
    rng = np.random.default_rng([int(seed), 20260812])
    return field[rng.permutation(len(field))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/stress_scans")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--scans",
        default="A,B,C",
        help=(
            "Comma-separated subset of {A,B,C} to (re)compute. Scans not listed keep their "
            "existing rows from the output directory's scan_seed_metrics.csv (must already "
            "exist). Default recomputes all three."
        ),
    )
    return parser.parse_args()


def condition_selected(base: dict, scenario: str, k: int) -> dict:
    """Copy the frozen canonical config, overriding only this scenario's k."""
    sel = copy.deepcopy(base)
    sel["shared_graph_k"][scenario] = k
    sel["local_pca"][scenario] = {"k": k}
    sel["position_only_manfit"][scenario] = {**sel["position_only_manfit"][scenario], "k": k}
    sel["velocity_manifold_fitter"][scenario] = {**sel["velocity_manifold_fitter"][scenario], "k": k}
    return sel


def curvature_aware_k_for_condition(scenario: str, d: int, n: int, sigma_x: float) -> tuple[int, dict]:
    k_grid = curvature_probe_k_grid(n, d)
    curves = [
        local_pca_normal_residual(
            m.vector_data(scenario, seed, override_n=n, position_noise=sigma_x)["Y"], d, k_grid
        )
        for seed in TUNING_SEEDS
    ]
    return curvature_aware_neighbor_count(k_grid, curves)


def evaluate_condition(
    scenario: str,
    base_selected: dict,
    *,
    scan: str,
    n: int | None = None,
    sigma_x: float | None = None,
    sigma_v: float | None = None,
    sigma_v_relative: float | None = None,
    shuffle_velocity: bool = False,
) -> tuple[pd.DataFrame, dict]:
    setting = m.SETS[scenario]
    n_eff = int(n) if n else setting.n
    sigma_x_eff = float(sigma_x) if sigma_x is not None else setting.px
    sigma_v_eff = float(sigma_v) if sigma_v is not None else setting.field_noise
    d = setting.d
    compute_mechanism = scan == "C_velocity_noise"
    # k(n,d) and its curvature-aware refinement depend only on the *position* observations
    # (local_pca_normal_residual reads Y, never V), so for Scan C -- which holds n and sigma_X at
    # their canonical values and only varies sigma_V -- this reproduces the canonical k exactly
    # (same seeds, same n, same sigma_X -> identical position draws, since add_noise consumes the
    # position-noise random draw before the velocity-noise one). Still computed explicitly rather
    # than special-cased, so the "recompute k fresh at every scan point" rule has no silent
    # exceptions and the result is verifiable rather than assumed. Shuffling velocity after the
    # fact doesn't touch this either (still purely a position-observation rule).
    k, diag = curvature_aware_k_for_condition(scenario, d, n_eff, sigma_x_eff)
    sel = condition_selected(base_selected, scenario, k)
    rows = []
    for seed in FINAL_SEEDS:
        data = m.vector_data(scenario, seed, override_n=n_eff, position_noise=sigma_x_eff, velocity_noise=sigma_v_eff)
        if shuffle_velocity:
            data["field"] = shuffle_velocity_field(data["field"], seed)
        tau_graph = m.shared_knn_graph(data["Y"], k)
        tau, _, _ = m.observed_tau(data["Y"], data["field"], tau_graph)
        baseline = m.state_metrics(scenario, data["Y"], data["field"], data, tau)
        states, _, _ = m.fit_final_states(scenario, seed, data, sel)
        for method in METHODS:
            Xhat, Vhat = states[method]
            nan_inf_count = int(np.sum(~np.isfinite(Xhat)) + np.sum(~np.isfinite(Vhat)))
            if nan_inf_count:
                Xhat = np.nan_to_num(Xhat, nan=0.0, posinf=0.0, neginf=0.0)
                Vhat = np.nan_to_num(Vhat, nan=0.0, posinf=0.0, neginf=0.0)
            absolute = m.state_metrics(scenario, Xhat, Vhat, data, tau)
            mechanism = (
                m.mechanism_metrics(scenario, Vhat, data, tau_graph)
                if (compute_mechanism and method in MECHANISM_METHODS)
                else {}
            )
            rows.append(
                {
                    "scan": scan,
                    "scenario": scenario,
                    "seed": seed,
                    "method": method,
                    "n": n_eff,
                    "sigma_x": sigma_x_eff,
                    "sigma_x_multiplier": round(sigma_x_eff / setting.px, 4),
                    "sigma_v": sigma_v_eff,
                    "sigma_v_relative": sigma_v_relative,
                    "velocity_shuffled": bool(shuffle_velocity),
                    "k": k,
                    "nan_inf_count": nan_inf_count,
                    **absolute,
                    **m.relative_state_metrics(absolute, baseline),
                    **{f"mechanism_{key}": value for key, value in mechanism.items() if key in MECHANISM_METRICS},
                }
            )
    return pd.DataFrame(rows), {
        "scenario": scenario,
        "scan": scan,
        "n": n_eff,
        "sigma_x": sigma_x_eff,
        "sigma_v": sigma_v_eff,
        "sigma_v_relative": sigma_v_relative,
        "velocity_shuffled": bool(shuffle_velocity),
        **diag,
    }


def run_scans(base_selected: dict, scans: tuple[str, ...] = ("A", "B", "C")) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, diag_rows = [], []
    for scenario in SCENARIOS:
        setting = m.SETS[scenario]
        if "A" in scans:
            for n in SCAN_A_N_VALUES:
                frame, diag = evaluate_condition(scenario, base_selected, scan="A_sample_size", n=n)
                rows.append(frame)
                diag_rows.append(diag)
        if "B" in scans:
            for multiplier in SCAN_B_MULTIPLIERS:
                frame, diag = evaluate_condition(
                    scenario, base_selected, scan="B_position_noise", sigma_x=multiplier * setting.px
                )
                rows.append(frame)
                diag_rows.append(diag)
        if "C" in scans:
            v_scale = scenario_velocity_scale(scenario)
            for r_v in SCAN_C_RELATIVE_SIGMA_V:
                frame, diag = evaluate_condition(
                    scenario,
                    base_selected,
                    scan="C_velocity_noise",
                    sigma_v=r_v * v_scale,
                    sigma_v_relative=r_v,
                )
                rows.append(frame)
                diag_rows.append(diag)
            frame, diag = evaluate_condition(
                scenario,
                base_selected,
                scan="C_velocity_noise",
                shuffle_velocity=True,
            )
            rows.append(frame)
            diag_rows.append(diag)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(diag_rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "scan", "scenario", "method", "n", "sigma_x", "sigma_x_multiplier",
        "sigma_v", "sigma_v_relative", "velocity_shuffled", "k",
    ]
    # dropna=False: Scan A/B rows have sigma_v_relative=NaN by construction
    # (not applicable to those scans) -- default groupby behavior silently
    # drops NaN-keyed groups, which would delete every Scan A/B row here.
    for keys, g in frame.groupby(group_cols, sort=False, dropna=False):
        row = dict(zip(group_cols, keys))
        for metric in HEADLINE_METRICS:
            row[f"{metric}_median"] = float(g[metric].median())
            row[f"{metric}_q25"] = float(g[metric].quantile(0.25))
            row[f"{metric}_q75"] = float(g[metric].quantile(0.75))
        for metric in MECHANISM_METRICS:
            col = f"mechanism_{metric}"
            if col in g.columns and g[col].notna().any():
                row[f"{col}_median"] = float(g[col].median())
                row[f"{col}_q25"] = float(g[col].quantile(0.25))
                row[f"{col}_q75"] = float(g[col].quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_scan(
    summary: pd.DataFrame,
    scan: str,
    x_col: str,
    x_label: str,
    metric: str,
    output: Path,
    *,
    reference_line: float | None = 1.0,
    file_tag: str | None = None,
) -> Path:
    scenarios = list(SCENARIOS)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    colors = {"position_only_manfit": "#b8860b", "manfitvelo": "#1f6f5c", "local_pca": "#7a7a7a"}
    for ax, scenario in zip(axes.ravel(), scenarios):
        sub = summary[(summary.scan == scan) & (summary.scenario == scenario)]
        if "velocity_shuffled" in sub.columns:
            # The shuffled-velocity control has no position on this numeric
            # axis (sigma_v_relative is NaN for it by construction) -- it's
            # reported separately as a table, not plotted on this curve.
            # NOTE (bug fixed 2026-08-12): velocity_shuffled is NaN (not
            # False) for Scan A/B rows, since it's only ever set for Scan C.
            # `sub.velocity_shuffled == False` is False for NaN too (IEEE754
            # NaN comparison semantics), so that filter silently dropped
            # every Scan A/B row -- blank plots for both, discovered when
            # the consolidated report's Scan A/B panels turned out empty.
            # `!= True` correctly keeps NaN (NaN != True is True) while still
            # excluding real shuffled=True rows.
            sub = sub[sub.velocity_shuffled != True]  # noqa: E712
        sub = sub.sort_values(x_col)
        for method in COMPARISON_METHODS:
            ms = sub[sub.method == method]
            if ms.empty or f"{metric}_median" not in ms.columns or ms[f"{metric}_median"].isna().all():
                continue
            ax.plot(ms[x_col], ms[f"{metric}_median"], marker="o", color=colors[method], label=METHOD_LABELS[method])
            ax.fill_between(ms[x_col], ms[f"{metric}_q25"], ms[f"{metric}_q75"], color=colors[method], alpha=0.15)
        if reference_line is not None:
            ax.axhline(reference_line, color="black", linewidth=0.8, linestyle="--")
        if x_col in ("sigma_x_multiplier", "sigma_v"):
            ax.set_xscale("linear")
        else:
            ax.set_xscale("log")
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=10)
        ax.set_xlabel(x_label, fontsize=8)
        ax.tick_params(labelsize=7)
    axes.ravel()[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{metric} vs. {x_label} ({scan})")
    path = output / "figures" / f"{scan}_{file_tag or metric}.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def shuffle_control_table(summary: pd.DataFrame) -> str:
    """HTML table for the non-numeric shuffled-velocity negative control:
    same noise magnitude as canonical sigma_V, but row-permuted so velocity
    carries no position-specific information at all."""
    sub = summary[(summary.scan == "C_velocity_noise") & (summary.velocity_shuffled == True)]  # noqa: E712
    metrics = list(HEADLINE_METRICS) + [f"mechanism_{m}" for m in MECHANISM_METRICS]
    header = "".join(f"<th>{metric}</th>" for metric in metrics)
    rows_html = []
    for scenario in SCENARIOS:
        for method in COMPARISON_METHODS:
            row = sub[(sub.scenario == scenario) & (sub.method == method)]
            if row.empty:
                continue
            row = row.iloc[0]
            cells = "".join(
                f"<td>{row[f'{metric}_median']:.3f}</td>" if f"{metric}_median" in row and pd.notna(row[f"{metric}_median"]) else "<td>—</td>"
                for metric in metrics
            )
            rows_html.append(f"<tr><td>{SCENARIO_LABELS[scenario]}</td><td>{METHOD_LABELS[method]}</td>{cells}</tr>")
    return (
        "<table style='border-collapse:collapse;width:100%;font-size:12px'>"
        f"<thead><tr><th>Scenario</th><th>Method</th>{header}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )


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


def build_report(output: Path, summary: pd.DataFrame, regenerate_figures: bool = True) -> dict:
    style = """
body{margin:0;background:#f4f6f8;color:#17212b;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}main{max-width:1400px;margin:auto;padding:28px 20px 60px}.card{background:#fff;border:1px solid #d8dee7;border-radius:9px;padding:20px;margin:18px 0;overflow:auto}img{max-width:100%;height:auto}p{line-height:1.55}code{background:#edf1f5;padding:1px 4px}
"""
    (output / "figures").mkdir(parents=True, exist_ok=True)
    sections = []
    for scan, x_col, x_label in [
        ("A_sample_size", "n", "sample size n (log scale)"),
        ("B_position_noise", "sigma_x_multiplier", "sigma_X / canonical sigma_X"),
        ("C_velocity_noise", "sigma_v_relative", "r_V = sigma_V / median||V_true|| (log scale)"),
    ]:
        for metric in HEADLINE_METRICS:
            if regenerate_figures:
                path = plot_scan(summary, scan, x_col, x_label, metric, output)
            else:
                path = output / "figures" / f"{scan}_{metric}.png"
            sections.append(f"<section class='card'><h2>{scan} — {metric}</h2><img src='{image_uri(path)}'></section>")
    for metric in MECHANISM_METRICS:
        col = f"mechanism_{metric}"
        tag = col
        if regenerate_figures:
            path = plot_scan(
                summary, "C_velocity_noise", "sigma_v_relative",
                "r_V = sigma_V / median||V_true|| (log scale)", col, output,
                reference_line=None, file_tag=tag,
            )
        else:
            path = output / "figures" / f"C_velocity_noise_{tag}.png"
        sections.append(f"<section class='card'><h2>C_velocity_noise — {col} (absolute, no baseline=1 reference)</h2><img src='{image_uri(path)}'></section>")
    shuffle_table = shuffle_control_table(summary)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>ManfitVelo stress-test scans</title><style>{style}</style></head><body><main>"
        "<h1>Stress-test scans: sample size (A), position noise (B), velocity noise (C)</h1>"
        "<section class='card'><p>Position-only MANFIT (M5), ManfitVelo (M6), and Local PCA (M4) shown; "
        "dashed line = ambient noisy input (relative value 1.0). Shaded band = seed IQR. "
        "k(n,d) and its curvature-aware refinement are recomputed fresh at every scan point from that "
        "point's own development-seed draws (Scan C reuses the canonical k exactly, since it only "
        "depends on position observations, unaffected by velocity noise); shared VMF/Position-only "
        "hyperparameters (T, eta_g, theta, kappa, theta_schedule) stay frozen at their canonical "
        "values throughout. Scan C (redesigned 2026-08-12, current_plan.md P1.1) uses a per-scenario "
        "*relative* velocity-noise grid r_V = sigma_V / median||V_true|| (development-seed reference "
        "scale) instead of one shared absolute grid, and also reports the tangent/normal-component "
        "decomposition of each method's velocity error (mechanism_tangential_component_rmse / "
        "mechanism_normal_component_rmse, absolute units, not baseline-relative).</p></section>"
        + "".join(sections)
        + "<section class='card'><h2>C_velocity_noise — shuffled-velocity negative control</h2>"
        "<p>Same noisy-velocity magnitude as each scenario's own canonical sigma_V, but row-permuted "
        "across points so velocity carries zero position-specific information. All values below are "
        "relative to the ambient-noisy baseline computed under this same shuffled input (headline "
        "metrics) or absolute RMSE (mechanism metrics).</p>"
        + shuffle_table
        + "</section>"
        + "</main></body></html>"
    )
    report = output / "scan_report.html"
    report.write_text(html, encoding="utf-8")
    parser = AuditParser()
    parser.feed(html)
    return {
        "self_contained_html": all(v.startswith("data:image/png;base64,") for v in parser.images) and not parser.external,
        "embedded_figure_count": len(parser.images),
        "expected_figure_count": 3 * len(HEADLINE_METRICS) + len(MECHANISM_METRICS),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        summary = pd.read_csv(output / "summary_metrics.csv")
        audit = build_report(output, summary, regenerate_figures=False)
        (output / "sanity_checks.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(output / "scan_report.html")
        return

    scans = tuple(sorted({s.strip().upper() for s in args.scans.split(",") if s.strip()}))
    scan_full_names = {"A": "A_sample_size", "B": "B_position_noise", "C": "C_velocity_noise"}

    base_selected = load_frozen_config()
    t0 = time.time()
    frame, diagnostics = run_scans(base_selected, scans=scans)

    if set(scans) != {"A", "B", "C"}:
        # Partial rerun: keep existing rows for the scans NOT being recomputed
        # (must already exist on disk -- this only speeds up iterating on one
        # scan's design, it doesn't let you skip the first full run).
        existing_seed_path = output / "scan_seed_metrics.csv"
        existing_diag_path = output / "scan_k_diagnostics.csv"
        if not existing_seed_path.exists():
            raise FileNotFoundError(
                f"--scans {args.scans} requires an existing {existing_seed_path} to merge into; "
                "run with --scans A,B,C first."
            )
        recomputed_full_names = {scan_full_names[s] for s in scans}
        old_frame = pd.read_csv(existing_seed_path)
        old_diag = pd.read_csv(existing_diag_path)
        frame = pd.concat(
            [old_frame[~old_frame.scan.isin(recomputed_full_names)], frame], ignore_index=True
        )
        diagnostics = pd.concat(
            [old_diag[~old_diag.scan.isin(recomputed_full_names)], diagnostics], ignore_index=True
        )

    frame.to_csv(output / "scan_seed_metrics.csv", index=False)
    diagnostics.to_csv(output / "scan_k_diagnostics.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(output / "summary_metrics.csv", index=False)
    audit = build_report(output, summary, regenerate_figures=True)
    audit["runtime_seconds"] = time.time() - t0
    (output / "sanity_checks.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(output / "scan_report.html")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
