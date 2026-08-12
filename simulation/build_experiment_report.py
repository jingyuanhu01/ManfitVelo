"""Consolidated experiment report: methods, metrics, scenarios, parameters, results.

Pulls together methods_config.yaml, scenario_config.yaml, metric_definitions.md, the
canonical single-point results (results/manfitvelo_benchmark/), the stress-scan results
(results/stress_scans/, Scans A/B/C), and the lambda_v selection/confirmation runs
(results/lambda_sensitivity_tuning/, results/lambda_sensitivity_final/) into one
self-contained HTML document.

    python simulation/build_experiment_report.py
"""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "simulation"
CANONICAL = ROOT / "results/manfitvelo_benchmark"
SCANS = ROOT / "results/stress_scans"
LAMBDA_TUNING = ROOT / "results/lambda_sensitivity_tuning"
LAMBDA_FINAL = ROOT / "results/lambda_sensitivity_final"
WILCOXON = ROOT / "results/wilcoxon_test"
AMBIENT_D = ROOT / "results/manifold_dimension_scalability"
V1 = ROOT / "results/v1_field_family"
V2 = ROOT / "results/v2_manifold_family"
SCALAR_LAMBDA_V = ROOT / "results/scalar_lambda_v_selection"
P4_1 = ROOT / "results/p4_1_scalar_oracle_ablation"
S1 = ROOT / "results/s1_scalar_landscape_family"
S2 = ROOT / "results/s2_manifold_landscape_family"
OUTPUT = ROOT / "results/experiment_report"

# Primary, fairly-rankable-on-G1/G2 comparison set shown in the headline table (section 5.1).
PRIMARY_METHOD_ORDER = ("ambient_noisy", "graphvelo", "cosine_kernel", "joint_low_rank", "manfitvelo")
# M4/M5/M6 pipeline-capability ablation (section 5.2) -- three different implementations, not a
# single-parameter sweep (see the ablation section's own explanatory text).
ABLATION_METHOD_ORDER = ("local_pca", "position_only_manfit", "manfitvelo")
METHOD_LABELS = {
    "ambient_noisy": "M0 — Ambient Noisy Input",
    "cosine_kernel": "M2 — Cosine Kernel",
    "graphvelo": "M1 — GraphVelo",
    "joint_low_rank": "M3 — Joint Low-Rank",
    "local_pca": "M4 — Local PCA",
    "position_only_manfit": "M5 — Position-only MANFIT",
    "manfitvelo": "M6 — ManfitVelo",
}
HEADLINE_METRICS = ("clean_point_rmse_rel", "distance_to_manifold_rel", "velocity_rmse_loc_rel", "joint_euler_state_rmse_rel")
HEADLINE_LABELS = {
    "clean_point_rmse_rel": "G2: Clean-point RMSE (rel.)",
    "distance_to_manifold_rel": "G1: Distance to manifold (rel.)",
    "velocity_rmse_loc_rel": "V3: Velocity RMSE, location-anchored (rel.)",
    "joint_euler_state_rmse_rel": "E_flow: One-step Euler forecast RMSE (rel.)",
}

STYLE = """
:root{--bg:#f4f6f8;--card:#ffffff;--border:#d8dee7;--text:#17212b;--muted:#56616f;--accent:#1f6f5c;--head:#edf1f5}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
main{max-width:1400px;margin:auto;padding:32px 20px 80px}
h1{font-size:1.9rem;margin-bottom:4px}
h2{font-size:1.35rem;border-bottom:2px solid var(--accent);padding-bottom:6px;margin-top:0}
h3{font-size:1.05rem;color:var(--accent)}
.subtitle{color:var(--muted);margin-top:0}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:22px 24px;margin:20px 0;overflow-x:auto}
nav.toc{display:flex;gap:14px;flex-wrap:wrap;font-size:0.9rem;margin:18px 0}
nav.toc a{color:var(--accent);text-decoration:none;border:1px solid var(--border);border-radius:6px;padding:4px 10px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:10px 0}
th,td{border:1px solid var(--border);padding:6px 8px;text-align:center}
th{background:var(--head)}
tbody th,td.left{text-align:left}
td.best{background:#cdeed4;font-weight:650}
img{max-width:100%;height:auto;border-radius:6px;border:1px solid var(--border)}
.figgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}
code{background:var(--head);padding:1px 5px;border-radius:3px}
p{line-height:1.6}
.note{border-left:4px solid var(--accent);background:#eef6ff;padding:10px 14px;font-size:0.92rem}
small.src{color:var(--muted)}
ol,ul{line-height:1.7}
"""


def image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def section_methods() -> str:
    cfg = yaml.safe_load((SIM / "methods_config.yaml").read_text())
    rows = [
        ("M0", "Ambient Noisy Input", "no", "no", "reference only, never ranked"),
        ("M1", "GraphVelo", "no", "yes", "official algorithm untouched; input rescaled by a fixed truth-free rule"),
        ("M2", "Cosine Kernel", "no", "yes", "k(n,d) only"),
        ("M3", "Joint Low-Rank", "yes", "yes", f"fixed {cfg['joint_low_rank']['variance_threshold']:.2f} variance threshold"),
        ("M4", "Local PCA", "yes", "yes", "k(n,d) only"),
        ("M5", "Position-only MANFIT", "yes", "yes", "k(n,d) + shared (T, eta_g)"),
        ("M6", "ManfitVelo", "yes", "yes", "k(n,d) + shared (T, eta_g, theta, kappa, theta_schedule, lambda_v)"),
    ]
    table = "<table><thead><tr><th>ID</th><th class='left'>Method</th><th>Updates x&#770;</th><th>Updates v&#770;</th><th class='left'>Tuned?</th></tr></thead><tbody>"
    for r in rows:
        table += f"<tr><th>{r[0]}</th><td class='left'>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td class='left'>{r[4]}</td></tr>"
    table += "</tbody></table>"
    vmf = cfg["velocity_manifold_fitter"]
    return f"""
<section class="card" id="methods">
<h2>1. Methods</h2>
<p>Seven methods answer four questions (Weekly Plan v1.1 &sect;15): <b>Q1</b> does velocity information
improve manifold recovery (M5 vs M6)? <b>Q2a</b> does manifold information at all improve velocity
recovery (M1/M2 vs M6)? <b>Q2b</b> given manifold info is used, which fitting strategy recovers
velocity best (M4 vs M5 vs M6)? <b>Q3</b> does joint recovery improve dynamics (E_flow across all
methods)? M1/M2 never touch position, so their G1/G2 geometry metrics equal M0 by construction and
are not separately ranked on those.</p>
{table}
<p><b>Core fairness principle</b>: every method has exactly one fixed parameter-selection
rule applied identically across all scenarios &mdash; never a hand-picked constant per scenario. Where a
rule is itself a function of the data (k(n,d), its curvature-aware refinement, M3's rank threshold),
it is allowed to output different numbers on different scenarios: that is what makes it
&ldquo;data-adaptive&rdquo; rather than &ldquo;scenario-specific tuning.&rdquo;</p>

<h3>M6 — ManfitVelo, step by step</h3>
<p>Given noisy ambient positions <code>Y</code> and noisy velocities <code>W</code>, iterated <code>T</code> times:</p>
<ol>
<li><b>Velocity-aware neighbor selection.</b> For each point, take a larger Euclidean candidate pool
(<code>k &times; candidate_mult</code>) and rerank it by a distance that blends spatial proximity with
a sigmoid of the cosine similarity between the query's velocity and each candidate's displacement
(sharpness <code>theta</code>, controlled optionally by <code>theta_schedule</code> ramping theta up
over iterations); keep the top <code>k</code>. This step alone already uses velocity information,
independent of lambda_v below.</li>
<li><b>Kernel weights.</b> Combine a spatial term <code>(1&minus;(d/h)&sup2;)^beta</code> (bandwidth
<code>h</code>, fixed or locally variable per <code>bandwidth_mode</code>) with a directional term
<code>exp(kappa&middot;cos)</code> so neighbors whose displacement points the same way as the query's
velocity are upweighted.</li>
<li><b>Joint tangent estimation (the core mechanism).</b> Compute the weighted local covariance of
neighbor <em>positions</em>, then blend in the local covariance of neighbor <em>velocities</em>
(trace-normalized to match the position covariance's scale) with weight <code>lambda_v</code>:
<code>C = C_position + lambda_v &middot; C_velocity</code>. Eigendecompose <code>C</code> to get the
local tangent basis and normal-space projector. <code>lambda_v=0</code> would use position geometry
alone here — this is the one step that actually implements "use velocity to improve manifold
recovery," and its weight is validated separately in &sect;5.2.2 rather than folded into the grid
search below.</li>
<li><b>Velocity projection.</b> Project the (already direction-reweighted) velocity onto the
estimated tangent space.</li>
<li><b>Position update.</b> Remove the tangential component of the local weighted mean-shift, keeping
only the normal-direction correction (<code>update_mode=normal_only</code>), scaled by
<code>eta_g</code> and capped at a fraction of the local bandwidth.</li>
</ol>
<p>Frozen shared values (once-for-all pooled grid search, &sect;4): T={vmf['T']}, eta_g={vmf['eta_g']:g},
theta={vmf['theta']:g}, kappa={vmf['kappa']:g}, theta_schedule={vmf['theta_schedule'] or 'flat'},
lambda_v={vmf['lambda_v']:g} (selected separately, &sect;5.2.2).</p>

<h3>M1 — GraphVelo</h3>
<p>Vendored, unmodified official implementation (n_neighbors={cfg['graphvelo']['n_neighbors']},
a={cfg['graphvelo']['a']:g}, b={cfg['graphvelo']['b']:g}, r={cfg['graphvelo']['r']:g},
loss={cfg['graphvelo']['loss_func']}): builds a cosine-similarity transition kernel over a k-NN graph,
density-corrects it, then solves a per-point ridge-regularized least-squares problem reconstructing
velocity from the graph's local displacement structure (minimizing
<code>a&Vert;&Sigma;&phi;(x_j&minus;x_i)&minus;v_i&Vert;&sup2; + r&Vert;&phi;&Vert;&sup2;</code>). Never
updates position. The reported row applies one fixed, truth-free unit standardization
(median-15-NN-distance / median-noisy-speed) before calling the unmodified official objective &mdash;
necessary because the objective is not scale-invariant, never tuned or selected by performance. Raw
(unstandardized) output retained only as a sensitivity diagnostic.</p>

<h3>M2 — Cosine Kernel Smoothing</h3>
<p>For each point, computes the cosine similarity between its velocity and each k-NN neighbor's
displacement, density-corrects (subtracts the row mean), and reconstructs a smoothed direction as the
weighted sum of neighbor displacements; rescales to the observed noisy speed. Represents the simplest
"local averaging only" baseline — no manifold/tangent estimation, no position update.</p>

<h3>M3 — Joint Low-Rank Denoising (replaces the earlier Global PCA baseline)</h3>
<p>{cfg['joint_low_rank']['rule']} Rank is a per-sample deterministic function of the observed
singular-value spectrum only &mdash; never ground truth, never tuned.</p>

<h3>M4 — Local PCA</h3>
<p>Pointwise local PCA: for each point, take its k nearest neighbors (fixed at t=0, no velocity
involved), fit a rank-d affine subspace, project the point onto it. Velocity is a downstream step —
projected onto the same local tangent basis recomputed at the denoised positions. No global iteration.</p>

<h3>M5 — Position-only MANFIT (Manifold-Aware Only)</h3>
<p>{cfg['position_only_manfit']['tuning'].split('once-for-all')[0]}Independent implementation
(<code>position_only_trajectory</code>, never invokes VMF): plain Euclidean k-NN frozen at t=0,
spatial-only weights <code>(1&minus;(d/h)&sup2;)^beta</code>, iterated normal-only position update —
structurally the same iterative normal-mean-shift *procedure* as M6, but with no velocity anywhere,
not even for neighbor selection. <b>Not</b> numerically equivalent to M6 at lambda_v=0 (see &sect;5.2.2).</p>

<h3>Relationship to the wider RNA velocity / manifold-learning literature</h3>
<p>M1 (GraphVelo), M2 (Cosine Kernel), and M3 (Joint Low-Rank) are the three real competitors directly
comparable to ManfitVelo <em>on this specific task</em>: denoising/reprojecting an already-observed
noisy (position, velocity) pair via manifold structure. Most of the wider RNA velocity literature
(scVelo's steady-state/stochastic/dynamical models, veloVI, DeepVelo, cellDancer, UniTVelo, &hellip;)
instead solves a different problem — <em>estimating</em> velocity from spliced/unspliced count data —
and is not directly comparable without a substantially different, count-level simulation. Dynamo's
vector-field reconstruction (sparseVFC) is methodologically closer in spirit (fitting a smooth,
denoised vector field on a learned manifold from noisy single-cell velocity estimates) but is a
separate algorithmic framework not implemented head-to-head here. M4 (Local PCA) and M5 (Position-only
MANFIT) are not external competitors either — they are internal ablations of ManfitVelo's own
pipeline (&sect;5.2), isolating the incremental contribution of each design choice rather than
representing a distinct published method.</p>
<small class="src">Full definitions: <code>parameter_rules.md</code>, <code>methods_config.yaml</code>.</small>
</section>
"""


def section_metrics() -> str:
    rows = "".join(f"<tr><td class='left'>{k}</td><td class='left'>{v}</td></tr>" for k, v in HEADLINE_LABELS.items())
    return f"""
<section class="card" id="metrics">
<h2>2. Benchmark metrics</h2>
<p>Every metric compares a method's output (x&#770;,v&#770;) against either the original clean generating
point (identity anchoring) or the point on the true manifold nearest the method's own denoised
location, x_proj = &Pi;<sub>M</sub>(x&#770;) (location anchoring). Every metric is reported both as an
absolute value and relative to the Ambient Noisy Input baseline &mdash; <b>values below 1.0 mean
&ldquo;better than doing nothing.&rdquo;</b></p>
<h3>Headline metrics (this report)</h3>
<table><thead><tr><th class='left'>Column</th><th class='left'>Reads as</th></tr></thead><tbody>{rows}</tbody></table>
<p><b>G1/G2</b> (geometry): did the denoised point land back on the manifold, and did it recover the
specific generating point (not just some point on the manifold)? <b>V3</b> (velocity, primary):
is the velocity consistent with where the method says the cell actually is &mdash; the
projection-aware question, more relevant than forcing every method to hit one fixed target.
<b>E_flow</b> (joint): if we forecast one small Euler step from the denoised state, how far off is the
predicted next state from the true one?</p>
<p><b>Projection ambiguity</b>: Curved Hairpin / Near Intersection use an oracle branch-aware
projection (restricted to the same labeled branch) since ordinary nearest-Euclidean projection can
jump to the wrong arm; Y-branch excludes points within radius 0.05 of its non-smooth branch point from
angle-type metrics.</p>
<small class="src">Full definitions incl. mechanism diagnostics: <code>metric_definitions.md</code>.</small>
</section>
"""


def section_scenarios() -> str:
    cfg = yaml.safe_load((SIM / "scenario_config.yaml").read_text())
    group_names = {
        "A_regular_smooth_manifold": "Group A — Regular smooth-manifold benchmarks",
        "B_geometry_stress_test": "Group B — Geometry stress tests (small reach / ambiguous neighborhoods)",
        "C_out_of_assumption": "Group C — Out-of-assumption robustness (non-smooth branch point)",
    }
    body = ""
    for gid, ginfo in cfg["groups"].items():
        body += f"<h3>{group_names.get(gid, gid)}</h3><p>{ginfo['description']}</p>"
        body += "<table><thead><tr><th class='left'>Scenario</th><th>n</th><th>&sigma;_X</th><th>&sigma;_V</th><th>d</th><th class='left'>Geometry</th></tr></thead><tbody>"
        for name in ginfo["scenarios"]:
            s = cfg["scenarios"][name]
            body += (
                f"<tr><th class='left'>{name}</th><td>{s['n']}</td><td>{s['sigma_X']:g}</td>"
                f"<td>{s['sigma_V']:g}</td><td>{s['intrinsic_dimension']}</td>"
                f"<td class='left'>{s['geometry']}</td></tr>"
            )
        body += "</tbody></table>"
    return f"""
<section class="card" id="scenarios">
<h2>3. Scenarios</h2>
<p>9 scenarios spanning curvature sign and magnitude: Flat Rotation Annulus is exactly flat,
Half-sphere-tangent and Swiss Roll are positively/extrinsically curved, Saddle Surface has
negative/mixed Gaussian curvature. Position noise is a single scalar draw per point along the
manifold's own analytic normal direction (not full isotropic noise); velocity noise
(&sigma;_V=0.10) is full ambient-dimensional and identical across every scenario.</p>
{body}
<small class="src">Full generative formulas: <code>scenario_config.yaml</code>.</small>
</section>
"""


def section_parameters() -> str:
    cfg = yaml.safe_load((SIM / "methods_config.yaml").read_text())
    rule = cfg["neighbor_count_rule"]
    k_rows = "".join(f"<tr><th class='left'>{s}</th><td>{k}</td></tr>" for s, k in cfg["shared_graph_k"].items())
    return f"""
<section class="card" id="parameters">
<h2>4. Parameter settings</h2>
<p>Priority order for any parameter (highest first): (1) official default; (2) data-adaptive rule,
a deterministic function of the observed data, allowed to vary by scenario; (3) once-for-all
development tuning, grid-searched once and pooled across all scenarios; (4) scenario-specific
tuning &mdash; <b>forbidden</b> in final experiments.</p>
<h3>Neighborhood size k(n,d) &mdash; two-stage, data-adaptive (tier 2)</h3>
<p><b>Stage 1</b>: <code>{rule['base_formula']}</code>, C={rule['C']:.2f} (single scalar,
dimension-independent since the 2026-08-12 P0.1 global-C selection -- superseded the
earlier per-dimension C_d dict).
<b>Stage 2</b>: {rule['curvature_refinement']}</p>
<p class="note">Applies identically to every scenario &mdash; including Curved Hairpin and
Near Intersection &mdash; with no scenario-specific exception.</p>
<table><thead><tr><th class='left'>Scenario</th><th>Frozen k</th></tr></thead><tbody>{k_rows}</tbody></table>
<h3>Shared (T, eta_g, &hellip;) &mdash; once-for-all pooled grid search (tier 3)</h3>
<p>M6 ManfitVelo: 162-candidate grid (T&times;eta_g&times;kappa&times;theta&times;theta_schedule),
scored by mean tuning_score pooled over all 9 scenarios &times; 3 tuning seeds. M5 Position-only
MANFIT: 9-candidate grid (T&times;eta_g), same pooled scoring. lambda_v is selected separately
(&sect;5.2.2) to keep its audit trail legible. See &sect;1 for the winning values.</p>
<small class="src">Full derivation of every rule: <code>parameter_rules.md</code>; full frozen values:
<code>methods_config.yaml</code>.</small>
</section>
"""


def headline_table(summary: pd.DataFrame, method_order: tuple[str, ...], safeguard_ref: str | None = None) -> str:
    scenarios = summary.scenario.unique().tolist()
    rows = []
    for scenario in scenarios:
        sub = summary[summary.scenario == scenario].set_index("method")
        for metric in HEADLINE_METRICS:
            col = f"{metric}_median"
            values = {method: sub.loc[method, col] for method in method_order if method in sub.index}
            comparable = {k: v for k, v in values.items() if k not in ("ambient_noisy", "graphvelo", "cosine_kernel")}
            best = min(comparable, key=comparable.get) if comparable else None
            cells = "".join(
                f"<td class='{'best' if method == best else ''}'>{values.get(method, float('nan')):.2f}</td>"
                for method in method_order
            )
            rows.append(f"<tr><th class='left'>{scenario if metric == HEADLINE_METRICS[0] else ''}</th><td class='left'>{HEADLINE_LABELS[metric]}</td>{cells}</tr>")
    header = "".join(f"<th>{METHOD_LABELS[m].split(chr(32),1)[0]}</th>" for m in method_order)
    return (
        "<table><thead><tr><th class='left'>Scenario</th><th class='left'>Metric</th>" + header + "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def section_ablation() -> str:
    summary = pd.read_csv(CANONICAL / "summary_metrics.csv")
    ablation_table = headline_table(summary, ABLATION_METHOD_ORDER)

    lambda_tuning_audit = pd.read_csv(LAMBDA_TUNING / "lambda_selection_audit.csv") if (LAMBDA_TUNING / "lambda_selection_audit.csv").exists() else None
    audit_table = ""
    if lambda_tuning_audit is not None:
        rows = "".join(
            f"<tr><td>{r.lambda_v:g}</td><td>{r.tuning_score:.4f}</td><td>{'&check;' if r.safe_for_every_scenario else '&mdash;'}</td></tr>"
            for r in lambda_tuning_audit.sort_values("tuning_score", ascending=False).itertuples()
        )
        audit_table = f"<table><thead><tr><th>lambda_v</th><th>pooled score (tuning seeds)</th><th>safe for every scenario</th></tr></thead><tbody>{rows}</tbody></table>"

    lambda_tuning_figs = "".join(
        f"<div><h4>{p.stem.replace('lambda_', '')} (tuning seeds — selection basis)</h4><img src='{image_uri(p)}'></div>"
        for p in sorted((LAMBDA_TUNING / "figures").glob("lambda_*.png"))
    )
    lambda_final_figs = "".join(
        f"<div><h4>{p.stem.replace('lambda_', '')} (final seeds — confirmatory)</h4><img src='{image_uri(p)}'></div>"
        for p in sorted((LAMBDA_FINAL / "figures").glob("lambda_*.png"))
    )

    return f"""
<section class="card" id="ablation">
<h2>5.2 Ablation: M4 &rarr; M5 &rarr; M6</h2>
<p>M4/M5/M6 are <b>three different implementations</b> (local PCA &rarr; velocity-independent
iterative normal-mean-shift &rarr; velocity-aware neighbor selection + joint tangent estimation), not
one algorithm with a single parameter varied — this is a pipeline-<em>capability</em> ablation, showing
what each successive design choice adds, complementary to the single-parameter lambda_v sweep below.
<b>Caveat</b>: velocity enters M6 through two distinct mechanisms — neighbor reranking (step 1, &sect;1)
and the covariance-blend term weighted by lambda_v (step 3) — and M5-vs-M6's difference is their
combined effect; this ablation was not further split to isolate each mechanism's individual
contribution on the vector-field side. (The scalar-field branch's S1/S2 controlled experiments,
&sect;6.5&ndash;6.6, do isolate them there: at the frozen lambda_v=0 — covariance blend off — the joint
pipeline still clearly beats geometry-only denoising on gradient recovery, showing neighbor reranking
alone carries real signal, independent of the covariance-blend mechanism.)</p>
{ablation_table}

<h3>5.2.2 lambda_v selection and sensitivity (single-parameter isolated ablation)</h3>
<p>Unlike M4/M5/M6 above, this sweep holds every other ManfitVelo setting (k, T, eta_g, theta, kappa,
theta_schedule, and hence neighbor selection) fixed at its own canonical value and varies only
lambda_v — the weight given to the velocity second-moment matrix in the tangent-covariance blend
(&sect;1). lambda_v was originally a carried-forward constant (0.1) from a pre-fairness-fix,
pre-curvature-aware-k prior study. Re-selected under the current protocol on <b>tuning seeds only</b>
(final seeds never enter this computation), pooling each candidate's score across all 9 scenarios via
the mean of log(clean_point_rmse_rel) + log(distance_to_manifold_rel) + log(velocity_rmse_loc_rel) +
log(joint_euler_state_rmse_rel).</p>
{audit_table}
<p class="note">The naive pooled-best (lambda_v=2.0) is only ~1% better than lambda_v=1.0 in aggregate,
but makes Swiss Roll's clean_point_rmse/distance_to_manifold <em>worse than its own lambda_v=0
baseline</em> while gaining almost nothing further on velocity_rmse_loc over 1.0 (0.916 vs 0.917) — a
Pareto-inefficient trade. <b>Selected: lambda_v=1.0</b>, which captures essentially all of the
velocity-metric gain without that regression.</p>
<h4>Selection curve (tuning seeds)</h4>
<div class="figgrid">{lambda_tuning_figs}</div>
<h4>Confirmatory curve (final seeds, reporting only)</h4>
<div class="figgrid">{lambda_final_figs}</div>
<small class="src">Full audit: <code>results/lambda_sensitivity_tuning/</code>,
<code>results/lambda_sensitivity_final/</code>; derivation: <code>parameter_rules.md</code> &sect;3a.</small>
</section>

{section_significance()}
"""


def section_significance() -> str:
    """P5 (current_plan.md): paired Wilcoxon signed-rank test for the M5-vs-M6 tie /
    thin-margin scenario/metric pairs flagged in the pre-freeze claim-language
    review. Reads results/wilcoxon_test/ (see run_wilcoxon_test.py) -- not
    recomputed here, that script is the single source of truth for these
    numbers."""
    path = WILCOXON / "wilcoxon_results.csv"
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    rows = "".join(
        f"<tr><td class='left'>{r.scenario}</td><td>{r.metric_label}</td>"
        f"<td>{r.m5_median:.4f}</td><td>{r.m6_median:.4f}</td>"
        f"<td>{r.m6_wins}/{r.n_seeds}</td>"
        f"<td>{r.two_sided_p_value:.4f}</td><td>{r.one_sided_m6_better_p_value:.4f}</td></tr>"
        for r in df.itertuples()
    )
    return f"""
<section class="card" id="significance">
<h3>5.2.3 Statistical significance — paired Wilcoxon signed-rank test (P5)</h3>
<p>The scenario/metric pairs with the thinnest M5-vs-M6 margins or split seed-level verdicts
elsewhere in this report, tested formally rather than eyeballed. Paired by seed (same 15 final-seed
noisy draw for both methods), using each scenario/seed's own *_rel metric (so the pairing also cancels
per-seed noise-level variation). "M6 wins" counts seeds where M6's paired value is strictly lower
(better); p-values from <code>scipy.stats.wilcoxon</code> (zero_method="pratt"), both two-sided and
one-sided (H1: M6 better) are reported rather than only whichever looks more favorable.</p>
{rows and f"<table><thead><tr><th class='left'>Scenario</th><th>Metric</th><th>M5 median</th><th>M6 median</th><th>M6 wins</th><th>p (two-sided)</th><th>p (one-sided, M6 better)</th></tr></thead><tbody>{rows}</tbody></table>"}
<p><b>Reading these</b>: circle (G1, G2) and swiss_roll (G1) are clearly significant in M6's favor
(p&lt;0.01) despite modest median margins. flat_rotation_annulus (V3) and swiss_roll (G2) have the
thinnest margins in the whole report (11/15 seed wins each) but are still significant at the
conventional p&lt;0.05 threshold. swiss_roll G2 is the most consequential case: its own *marginal*
medians (each method's median computed independently across the 15 seeds) put M5 ahead
(0.7152 vs 0.7269), which is what originally read as a "flip" after the C=0.60 rerun (&sect;5.1) — but
that comparison discards the seed pairing. The <em>paired</em> statistic (median of per-seed
M6&minus;M5 differences, and the Wilcoxon test built on it) tells a different, more relevant story: on
11 of 15 individual noisy draws M6 scores lower (better) than M5 on the *same* draw, and the signed-rank
test on those paired differences is still significant in M6's favor (p=0.048). Marginal-median and
paired comparisons can point in different directions on data with this much seed-to-seed variance
(n=15) — a real statistical subtlety, not an error in either number — and the paired test is the
correct one for this seed-matched design.</p>
<small class="src">Full data: <code>results/wilcoxon_test/wilcoxon_results.csv</code>; script:
<code>simulation/run_wilcoxon_test.py</code>.</small>
</section>
"""


def section_results() -> str:
    summary = pd.read_csv(CANONICAL / "summary_metrics.csv")
    primary_table = headline_table(summary, PRIMARY_METHOD_ORDER)
    state_figs = "".join(
        f"<div><h4>{p.stem.replace('state_', '').replace('_', ' ').title()}</h4><img src='{image_uri(p)}'></div>"
        for p in sorted((CANONICAL / "figures").glob("state_*.png"))
    )
    scan_figs = {}
    for scan in ("A_sample_size", "B_position_noise", "C_velocity_noise"):
        scan_figs[scan] = "".join(
            f"<div><h4>{p.stem.replace(f'{scan}_', '')}</h4><img src='{image_uri(p)}'></div>"
            for p in sorted((SCANS / "figures").glob(f"{scan}_*.png"))
        )
    return f"""
<section class="card" id="results">
<h2>5. Results</h2>

<h3>5.1 Primary comparison — all 9 scenarios, 15 final seeds</h3>
<p>Bold = best among the fairly-rankable-on-G1/G2 methods (M3, M6); M0/M1/M2 shown for reference
(M1/M2 never update position, so their G1/G2 always equal M0 by construction). Values &lt;1.0 = better
than noisy input. M4/M5 (internal ablations, not external competitors) are in &sect;5.2.</p>
{primary_table}
<p><b>Pattern</b>: ManfitVelo (M6) clearly beats noisy input on every scenario (0.2&ndash;0.8) and, with
the re-selected lambda_v=1.0 (&sect;5.2.2), beats Position-only MANFIT (M5) on <b>8/9 scenarios by
median-of-ratios</b> (up from 5/9 at the original lambda_v=0.1) — Swiss Roll is the one exception by
that measure (G2 clean-point median 0.727 vs M5's 0.715), but M6 still wins the paired-seed comparison
there 11/15, and a paired Wilcoxon signed-rank test on the seed-level differences is (marginally)
significant in M6's favor (p=0.048 two-sided, p=0.024 one-sided — &sect;5.2.3). See &sect;5.2 for the
full ablation. M3 fails (worse than
noisy) on every scenario with real ambient curvature &mdash; expected, since a global linear low-rank
subspace cannot represent a manifold that genuinely curves through all 3 ambient dimensions &mdash;
while trading position accuracy for large velocity-accuracy gains on the exactly-flat Flat Rotation
Annulus. M1/M2 help velocity on 1D curve scenarios but make it worse than noisy on the 2D curved ones
(Half-sphere, Swiss Roll, Saddle).</p>
</section>

{section_ablation()}

<section class="card" id="results-figures">
<h3>5.3 Representative state figures (fixed pilot seed)</h3>
<div class="figgrid">{state_figs}</div>

<h3>5.4 Scan A — sample size n &isin; &#123;200,400,800,1600&#125;</h3>
<p>Fixed &sigma;_X, &sigma;_V, D at each scenario's canonical value. k(n,d) and its curvature-aware
refinement are recomputed fresh at every n from that condition's own development-seed draws; shared
(T, eta_g, &hellip;, lambda_v) stay frozen at the canonical values in &sect;4/&sect;5.2.2.</p>
<div class="figgrid">{scan_figs["A_sample_size"]}</div>

<h3>5.5 Scan B — position noise &sigma;_X &isin; &#123;0.5,1,1.5,2,3&#125; &times; canonical</h3>
<p>The most direct test of Q1: does velocity information help recover geometry specifically as
position observations get noisier? Fixed n, &sigma;_V, D. Same recompute-per-scan-point rule for
k(n,d) as Scan A.</p>
<div class="figgrid">{scan_figs["B_position_noise"]}</div>

<h3>5.6 Scan C — relative velocity noise r_V = &sigma;_V / median&Vert;V_true&Vert; &isin;
&#123;0.05,0.1,0.2,0.4,0.8,1.6&#125; (redesigned 2026-08-12, current_plan.md P1.1)</h3>
<p>Original absolute grid (&sigma;_V up to 0.30) was too mild to move M6's geometry at all on most
scenarios. Redesigned as a per-scenario <em>relative</em> grid (r_V = &sigma;_V normalized by that
scenario's own median true-velocity norm, computed on tuning seeds only) plus a randomized/shuffled
velocity negative control. Core question: at what noise level does velocity stop being useful auxiliary
information and start actively hurting? Fixed n, &sigma;_X, D; k(n,d) is unaffected by &sigma;_V and
exactly reproduces the canonical value at every point (verified: M5's clean_point_rmse_rel, which never
uses velocity, is bit-identical across every r_V and shuffle condition).</p>
<p><b>Result</b>: a clean, monotonic M6-vs-M5 crossover on <b>8/9 scenarios</b> (Swiss Roll is already
flipped at the smallest r_V=0.05, consistent with its thin margin elsewhere in this report) — the
"flatter" scenarios where velocity helps least to begin with (Flat Rotation Annulus, Saddle Surface)
cross over earliest (r_V 0.2&rarr;0.4); the scenarios where lambda_v's own selection showed the largest
gain (Circle, S-curve, Curved Hairpin) hold on to an M6 advantage the longest (r_V 0.8&rarr;1.6) — a
consistency check between two independently-run experiments. <b>Shuffled-velocity control</b> at each
scenario's own canonical &sigma;_V (r_V equivalent &asymp;0.09&ndash;0.13): M6 loses to M5 on 7/9
scenarios once velocity direction is fully randomized, confirming M6 is genuinely using velocity
information rather than being immune to it by construction; Curved Hairpin and Flat Rotation Annulus
are exceptions (the latter counterintuitively scores *better* under full shuffle than under its own
real small-noise conditions — observed, not yet mechanistically explained, plausibly related to the
scenario's rotational symmetry).</p>
<div class="figgrid">{scan_figs["C_velocity_noise"]}</div>

<h3>5.7 Interpretation</h3>
<p><b>Q1</b> (does velocity help manifold recovery — M5 vs M6): yes, on <b>8/9 scenarios by
median-of-ratios</b> with the re-selected lambda_v=1.0 (&sect;5.2.1&ndash;5.2.2) — a materially
stronger and more consistent result than the original lambda_v=0.1 default gave (5/9). Swiss Roll is
the one scenario where the median-of-ratios comparison favors M5, but the paired-seed comparison still
favors M6 (11/15) and a paired Wilcoxon signed-rank test on the same seed-level differences is
marginally significant in M6's favor (p=0.048 two-sided — &sect;5.2.3); circle and flat_rotation_annulus's
own thin margins are both clearly significant (p&lt;0.02). Scan B lets Q1 be read as a function of noise
level rather than a single point estimate; Scan C (above) does the same for velocity noise specifically,
and finds M6's advantage is not unconditional — it crosses over to favor M5 once velocity gets noisy
enough, on 8/9 scenarios, in a pattern consistent with lambda_v's own scenario-level selection gains.
<b>Q2a</b> (does manifold information help at all — M1/M2 vs M6): M1/M2 only touch velocity and
underperform M6 on every 2D-curved scenario, supporting a "yes."
<b>Q2b</b> (which manifold-fitting strategy — M4 vs M5 vs M6): see the pipeline-capability ablation in
&sect;5.2 — each successive design choice (velocity-aware neighbor selection, then joint tangent
estimation) contributes incremental improvement on <b>8/9 scenarios</b>; Half-sphere-tangent is the one
exception, but it is a pooled-hyperparameter cost rather than a real capability regression: the shared
(T, eta_g) pooled across all 9 scenarios scores 44.6% worse there than that scenario's own locally-optimal
(T, eta_g) would (results/half_sphere_diagnosis/p0_2_summary.json, current_plan.md P0.2) — with its own
optimal hyperparameters, M6 would actually beat M4 on Half-sphere too.
<b>Q3</b> (does joint recovery improve dynamics — E_flow): tracks the G1/G2 pattern closely for
M4–M6, as expected since E_flow is dominated by the position term at the fixed small &tau; used here.
<b>Q4</b> (regimes where ManfitVelo helps or fails): see Scan A/B/C curves per scenario — the target
finding is a clearly identifiable regime rather than uniform dominance, consistent with the Weekly
Plan's explicit non-goal of "wins everywhere."</p>
</section>
"""


def section_extensions() -> str:
    """Part II: the controlled-experiment and scalar-field extensions built after the
    core 9-scenario benchmark (current_plan.md P1.2, P3, P4). Compact summaries with
    pointers to each experiment's own full self-contained report, rather than
    re-embedding every figure here -- each of these already has its own detailed
    HTML report with figures; this section is the single index tying them together
    with the key numbers and the frozen-protocol context, not a duplicate of them."""
    return f"""
<section class="card" id="ambient-d">
<h2>6. Extensions: ambient dimension, controlled experiments, scalar-field branch</h2>
<p>Built after the core 9-scenario benchmark, all under the same frozen protocol (global C=0.60,
curvature-aware k, lambda_v=1.0 for directly-observed velocity). Each has its own full report with
figures; this section indexes them with the headline numbers.</p>

<h3>6.1 Ambient-dimension scalability (P1.2)</h3>
<p>Circle (d=1) and Saddle Surface (d=2) embedded in ambient dimension D up to 3&ndash;several tens,
with position noise drawn along the manifold's own analytic normal direction (normal_only mode, scale
independent of D by construction — see &sect;8 of parameter_rules.md) — a complementary question to
`run_sphere_scalability.py`'s existing isotropic-Gaussian-noise experiment (which isolates whether that
different noise mechanism works at all, on a single positive-curvature manifold), not a duplicate of
it: this one isolates whether ambient dimension itself hurts recovery, independent of noise mode.
Full report/figures: <code>results/manifold_dimension_scalability/scalability_report.html</code>.</p>

<h3>6.2 Experiment V1 — same manifold, different vector fields (P3)</h3>
<p>Flat unit disk, five vector fields (source, sink, saddle, rotation, nonlinear) sharing every other
setting. Tests whether field structure (linear/affine vs. genuinely nonlinear, divergent vs.
rotational) changes recovery with geometry held fixed. Full report/figures:
<code>results/v1_field_family/v1_report.html</code>.</p>

<h3>6.3 Experiment V2 — same intrinsic dynamics, different manifolds (P3)</h3>
<p>One shared latent dynamics (u&#775;=1, v&#775;=0) pushed forward through four embeddings
(flat_plane, sphere_patch, swiss_roll, saddle_surface) via the Jacobian pushforward
V=D&phi;(u,v)&middot;(1,0), so every embedding observes literally the same intrinsic dynamics and only
extrinsic curvature differs. Full report/figures: <code>results/v2_manifold_family/v2_report.html</code>.</p>

<h3>6.4 Scalar-field branch (P4)</h3>
<p>The scalar analog of the vector-field pipeline: an estimated scalar gradient stands in for velocity,
fed into the same <code>VelocityManifoldFitter</code> machinery
(<code>scripts/scalar_potential_manfit.fit_scalar_gradient_manfit</code>). P4.1's oracle-gradient
ablation separates local-regression error (estimating the gradient from noisy scalar observations)
from joint-fitting error (the manifold-fitting stage itself): on <code>scalar_saddle</code>, raw local
regression has a gradient RMSE around 0.58 (comparable to the gradient's own magnitude); the joint
pipeline's remaining error under an oracle (exact) gradient is much smaller (clean_point_rmse 0.017),
isolating how much of the total error is attributable to each stage. Full decomposition:
<code>results/p4_1_scalar_oracle_ablation/p4_1_decomposition.json</code>.</p>
<p><b>lambda_v for the scalar branch</b>: unlike directly-observed velocity, a vector-field-tuned
lambda_v=1.0 was found to <em>help</em> the oracle-gradient pipeline but <em>hurt</em> the realistic
estimated-gradient one on <code>scalar_saddle</code> (clean_point_rmse 0.0204&rarr;0.0511 at
lambda_v=1.0, worse than not blending covariance at all). Explored three per-point confidence-scaling
mechanisms to discount lambda_v by the gradient estimate's own reliability (<code>"power"</code>,
<code>"inverse_error"</code>, <code>"rank"</code> — see <code>VelocityManifoldFitter</code>'s own
docstring for the full family), then ran a proper tuning-seed selection (mirroring &sect;5.2.2's own
lambda_v procedure) over lambda_v &isin; &#123;0, 0.5, 1, 2, 4&#125; under the <code>"rank"</code>
scaling: <b>selected lambda_v=0.0</b> — every candidate above 0 regressed <code>scalar_saddle</code>
below its own safe baseline. Full audit: <code>results/scalar_lambda_v_selection/</code>;
derivation: <code>parameter_rules.md</code> &sect;3b.</p>

<h3>6.5 Experiment S1 — same manifold, different scalar landscapes (P4)</h3>
<p>Scalar analog of V1: flat unit disk, four landscapes (single basin, double well, saddle, a
log-sum-exp nonlinear/multimodal well) at the frozen lambda_v=0.0 protocol. Across all four landscapes,
<code>joint_scalar_aware</code> (full pipeline) clearly beats <code>geometry_only</code> (Local-PCA
denoise, then post-hoc gradient regression) on gradient recovery — e.g. saddle: 0.265 vs. 0.384 gradient
RMSE — even with the covariance-blend term switched off, showing the gain traces to velocity-aware
neighbor reranking specifically. Full report/figures: <code>results/s1_scalar_landscape_family/s1_report.html</code>.</p>

<h3>6.6 Experiment S2 — same scalar landscape, different manifolds (P4)</h3>
<p>Scalar analog of V2: one shared nonlinear landscape (S1's log-sum-exp well) transported to the same
four V2 embeddings via the local pullback metric (needed since the landscape is defined through the
chart, not read directly off ambient coordinates). Same <code>joint_scalar_aware</code> &gt;
<code>geometry_only</code> pattern holds on all four manifolds, including two
(<code>sphere_patch</code>, <code>swiss_roll</code>) where <code>geometry_only</code> has already lost
its edge over doing nothing at all — on <code>sphere_patch</code>, geometry-first denoising actually
backfires (0.743 vs. 0.660 gradient RMSE for raw), traced to Local-PCA-denoised neighborhoods becoming
nearly coplanar and ill-conditioning the ambient gradient regression (design-matrix condition number
roughly doubles). Full report/figures: <code>results/s2_manifold_landscape_family/s2_report.html</code>.</p>
</section>
"""


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    body = (
        "<h1>ManfitVelo Simulation — Consolidated Experiment Report (v3)</h1>"
        "<p class='subtitle'>Methods, metrics, scenarios, parameters, and results for the formal "
        "9-scenario benchmark, the M4-M5-M6 ablation with lambda_v selection, sample-size / "
        "position-noise / velocity-noise stress tests (Scans A/B/C), significance testing (P5), and "
        "the ambient-dimension / controlled-experiment / scalar-field extensions (P1.2, P3, P4).</p>"
        "<nav class='toc'><a href='#methods'>1. Methods</a><a href='#metrics'>2. Metrics</a>"
        "<a href='#scenarios'>3. Scenarios</a><a href='#parameters'>4. Parameters</a>"
        "<a href='#results'>5. Results</a><a href='#ablation'>5.2 Ablation</a>"
        "<a href='#significance'>5.2.3 Significance</a><a href='#ambient-d'>6. Extensions</a></nav>"
        + section_methods()
        + section_metrics()
        + section_scenarios()
        + section_parameters()
        + section_results()
        + section_extensions()
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>ManfitVelo Experiment Report</title><style>{STYLE}</style></head>"
        f"<body><main>{body}</main></body></html>"
    )
    path = OUTPUT / "index.html"
    path.write_text(html, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
