"""P3 Experiment V1: same manifold, different vector fields.

current_plan.md P3/V1. Fixed flat 2D disk (unit radius, embedded as the z=0 plane
in R^3 -- same embedding/normal convention as flat_rotation_annulus), n, and
noise held constant; only the vector field changes across five field types:

    source     v(x,y) = +(x,y)
    sink       v(x,y) = -(x,y)
    saddle     v(x,y) = (x,-y)
    rotation   v(x,y) = (-y,x)
    nonlinear  v(x,y) = (1, sin(pi x))   -- not of the form Ax+b, unlike the
                                             other four, which are all
                                             globally linear/affine and so
                                             "too global-low-rank" on their
                                             own (current_plan.md's own framing)

A double-well gradient-flow field is explicitly optional in current_plan.md and is
NOT implemented this round (deferred, not forgotten).

Each field is rescaled to a median speed of 1 over its own sampled points
before noise is added, so sigma_V=0.10 means a comparable relative noise
level across field types while preserving each field's own within-field
speed *variation* (e.g. rotation/source/saddle genuinely grow with radius;
nonlinear does not) -- a global rescale, not a per-point unit-speed
normalization, which would have erased exactly the structural differences
this experiment is designed to compare.

Question: with geometry held fixed, how does field structure (linear vs.
nonlinear, rotational vs. divergent) affect recovery?

Frozen shared hyperparameters (T, eta_g, theta, kappa, theta_schedule,
lambda_v) are reused directly from the canonical protocol -- identical
across all 9 canonical scenarios (borrowed from the "circle" entry purely as
a representative source; the values are the same everywhere), only k(n,d) is
freshly computed for this scenario's own (n, d) via
simulation.benchmark_core.neighbor_count, matching the tier-2 (data-adaptive)
/ tier-3 (frozen) split used throughout this pipeline. 15 final seeds,
reporting only -- nothing here informs any hyperparameter selection, so
there is no final-seed-leakage concern.

    python simulation/run_v1_field_family.py
"""

from __future__ import annotations

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

from scripts.graphvelo_official_adapter import graphvelo_velocity_standardized  # noqa: E402
from scripts.benchmark_scenarios import fit_vmf_variant, position_only_trajectory  # noqa: E402
from scripts.simulation_baselines import (  # noqa: E402
    cosine_kernel_projection,
    downstream_velocity,
    joint_low_rank_state,
    local_pca_state,
    restore_noisy_speed,
    shared_knn_graph,
)
from simulation.benchmark_core import angle_mae, joint_error, load_frozen_config, neighbor_count, observed_tau, vector_rmse  # noqa: E402

FINAL_SEEDS = tuple(range(43000, 43015))
N = 480
INTRINSIC_DIMENSION = 2
SIGMA_X = 0.05
SIGMA_V = 0.10
EPS = 1e-12

METHODS = ("ambient_noisy", "cosine_kernel", "graphvelo", "joint_low_rank", "local_pca", "position_only_manfit", "manfitvelo")
LABELS = {
    "ambient_noisy": "Ambient noisy input", "cosine_kernel": "Cosine kernel", "graphvelo": "GraphVelo",
    "joint_low_rank": "Joint Low-Rank (M3)", "local_pca": "Local PCA",
    "position_only_manfit": "Position-only MANFIT", "manfitvelo": "ManfitVelo",
}


def field_source(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x, y


def field_sink(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return -x, -y


def field_saddle(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x, -y


def field_rotation(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return -y, x


def field_nonlinear(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.ones_like(x), np.sin(np.pi * x)


FIELDS = {
    "source": field_source, "sink": field_sink, "saddle": field_saddle,
    "rotation": field_rotation, "nonlinear": field_nonlinear,
}


def raw_field(field_name: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    fx, fy = FIELDS[field_name](x, y)
    return np.c_[fx, fy, np.zeros_like(x)]


def disk_data(field_name: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(0, 1, N))
    theta = rng.uniform(0, 2 * np.pi, N)
    x, y = r * np.cos(theta), r * np.sin(theta)
    X = np.c_[x, y, np.zeros(N)]

    raw = raw_field(field_name, x, y)
    speed = np.linalg.norm(raw, axis=1)
    field_scale = float(np.median(speed[speed > EPS])) if np.any(speed > EPS) else 1.0
    F = raw / max(field_scale, EPS)

    normal = np.tile([0.0, 0.0, 1.0], (N, 1))
    Y = X + rng.normal(scale=SIGMA_X, size=(N, 1)) * normal
    W = F + rng.normal(scale=SIGMA_V, size=F.shape)
    return {
        "Y": Y, "field": W, "P": X, "truth": F, "labels": np.zeros(N, int),
        "d": INTRINSIC_DIMENSION, "field_name": field_name, "field_scale": field_scale,
    }


def evaluation_targets(data: dict, Xhat: np.ndarray) -> dict:
    """Distance to the true disk (clip to unit radius on z=0, same style as
    flat_rotation_annulus's own evaluation_targets branch) and the true
    field evaluated at that projected location (location-anchored velocity
    target), using the SAME field_scale as truth generation for consistent
    units."""
    position = np.asarray(Xhat, float).copy()
    position[:, 2] = 0.0
    radius = np.linalg.norm(position[:, :2], axis=1)
    clipped = np.clip(radius, 0.0, 1.0)
    scale = np.where(radius > EPS, clipped / np.maximum(radius, EPS), 1.0)
    position[:, :2] *= scale[:, None]
    projection_rmse = float(np.sqrt(np.mean(np.sum((Xhat - position) ** 2, axis=1))))

    raw = raw_field(data["field_name"], position[:, 0], position[:, 1])
    location_velocity = raw / max(data["field_scale"], EPS)
    return {"position": position, "velocity": location_velocity, "distance_to_manifold": projection_rmse}


def state_metrics(data: dict, Xhat: np.ndarray, Vhat: np.ndarray, tau: float) -> dict:
    target = evaluation_targets(data, Xhat)
    valid = np.ones(len(Xhat), dtype=bool)
    return {
        "clean_point_rmse": float(np.sqrt(np.mean(np.sum((Xhat - data["P"]) ** 2, axis=1)))),
        "distance_to_manifold": target["distance_to_manifold"],
        "velocity_rmse_id": vector_rmse(Vhat, data["truth"], valid),
        "velocity_angle_mae_id": angle_mae(Vhat, data["truth"], valid),
        "velocity_rmse_loc": vector_rmse(Vhat, target["velocity"], valid),
        "joint_euler_state_rmse": joint_error(Xhat, Vhat, data["P"], data["truth"], tau),
    }


def relative_state_metrics(absolute: dict, baseline: dict) -> dict:
    return {f"{key}_rel": float(absolute[key] / baseline[key]) for key in absolute}


def fit_method(method: str, data: dict, config: dict, k: int, seed: int):
    X, V, d = data["Y"], data["field"], data["d"]
    if method == "ambient_noisy":
        return X.copy(), V.copy()
    if method == "cosine_kernel":
        direction, _ = cosine_kernel_projection(X, V, shared_knn_graph(X, k))
        Vhat, _ = restore_noisy_speed(direction, V)
        return X.copy(), Vhat
    if method == "graphvelo":
        Vhat, _ = graphvelo_velocity_standardized(X, V)
        return X.copy(), Vhat
    if method == "joint_low_rank":
        Xhat, Vhat, _ = joint_low_rank_state(X, V)
        return Xhat, Vhat
    if method == "local_pca":
        Xhat, Vhat, _ = local_pca_state(X, V, d, k)
        return Xhat, Vhat
    if method == "position_only_manfit":
        cfg = config["position_only_manfit"]
        Xhat = position_only_trajectory(X, V, d, k, cfg["T"], cfg["eta_g"])[-1][1]
        Vhat, _ = downstream_velocity(Xhat, V, d, k)
        return Xhat, Vhat
    if method == "manfitvelo":
        vmf_cfg = dict(config["velocity_manifold_fitter"])
        vmf_cfg["k"] = k
        result = fit_vmf_variant(X, V, d, vmf_cfg, seed)
        return result["X"], result["V"]
    raise KeyError(method)


def run(config_source_scenario: str = "circle") -> tuple[pd.DataFrame, dict]:
    frozen = load_frozen_config()
    vmf_config = frozen["velocity_manifold_fitter"][config_source_scenario]
    position_config = {"position_only_manfit": frozen["position_only_manfit"][config_source_scenario]}
    k = neighbor_count(N, INTRINSIC_DIMENSION)

    rows = []
    for field_name in FIELDS:
        for seed in FINAL_SEEDS:
            data = disk_data(field_name, seed)
            graph = shared_knn_graph(data["Y"], k)
            tau, _, _ = observed_tau(data["Y"], data["field"], graph)
            baseline = state_metrics(data, data["Y"], data["field"], tau)
            for method in METHODS:
                Xhat, Vhat = fit_method(method, data, {"velocity_manifold_fitter": vmf_config, **position_config}, k, seed)
                nan_inf_count = int(np.sum(~np.isfinite(Xhat)) + np.sum(~np.isfinite(Vhat)))
                if nan_inf_count:
                    Xhat = np.nan_to_num(Xhat, nan=0.0, posinf=0.0, neginf=0.0)
                    Vhat = np.nan_to_num(Vhat, nan=0.0, posinf=0.0, neginf=0.0)
                absolute = state_metrics(data, Xhat, Vhat, tau)
                relative = relative_state_metrics(absolute, baseline)
                rows.append(
                    {
                        "field": field_name, "seed": seed, "method": method, "method_label": LABELS[method],
                        "k": k, "nan_inf_count": nan_inf_count, **absolute,
                        **relative,
                    }
                )
    frame = pd.DataFrame(rows)
    provenance = {
        "n": N, "d": INTRINSIC_DIMENSION, "k": k, "sigma_x": SIGMA_X, "sigma_v": SIGMA_V,
        "shared_config_source_scenario": config_source_scenario, "vmf_config": vmf_config,
        "position_only_config": position_config["position_only_manfit"], "final_seeds": list(FINAL_SEEDS),
        "fields": list(FIELDS.keys()), "deferred": ["double_well (explicitly optional in current_plan.md)"],
    }
    return frame, provenance


HEADLINE_METRICS = ("clean_point_rmse_rel", "distance_to_manifold_rel", "velocity_rmse_loc_rel", "joint_euler_state_rmse_rel")


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in frame.groupby(["field", "method", "method_label"], sort=False):
        row = dict(zip(["field", "method", "method_label"], keys))
        for metric in HEADLINE_METRICS:
            row[f"{metric}_median"] = float(g[metric].median())
            row[f"{metric}_q25"] = float(g[metric].quantile(0.25))
            row[f"{metric}_q75"] = float(g[metric].quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows)


# Two views, mirroring the canonical benchmark report's own split (build_experiment_report.py
# SS5.1/SS5.2) rather than only ever showing the M4/M5/M6 ablation trio -- added 2026-08-12 after
# noticing all 7 METHODS were computed and stored in summary_metrics.csv but only 3 were ever
# plotted/tabled, which under-used data already paid for and made the report look incomplete.
PRIMARY_METHOD_ORDER = ("ambient_noisy", "graphvelo", "cosine_kernel", "joint_low_rank", "manfitvelo")
ABLATION_METHOD_ORDER = ("local_pca", "position_only_manfit", "manfitvelo")

_METHOD_COLORS = {
    "ambient_noisy": "#7a7a7a", "graphvelo": "#c2410c", "cosine_kernel": "#0e7490",
    "joint_low_rank": "#9d174d", "local_pca": "#7a7a7a",
    "position_only_manfit": "#b8860b", "manfitvelo": "#1f6f5c",
}


def plot_methods(summary: pd.DataFrame, output: Path, methods: tuple[str, ...], filename: str) -> Path:
    fields = list(FIELDS.keys())
    fig, axes = plt.subplots(1, len(fields), figsize=(4.4 * len(fields), 4), constrained_layout=True)
    for ax, field_name in zip(axes, fields):
        sub = summary[summary.field == field_name]
        x = np.arange(len(methods))
        heights = [float(sub[sub.method == m]["clean_point_rmse_rel_median"].iloc[0]) for m in methods]
        ax.bar(x, heights, color=[_METHOD_COLORS[m] for m in methods])
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in methods], fontsize=7, rotation=25, ha="right")
        ax.set_title(field_name, fontsize=10)
        ax.set_ylabel("clean_point_rmse_rel")
    path = output / "figures" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


class AuditParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []
        self.external = False

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            for key, value in attrs:
                if key == "src":
                    self.images.append(value)
                    if not value.startswith("data:image/png;base64,"):
                        self.external = True


def build_report(output: Path, summary: pd.DataFrame, provenance: dict) -> dict:
    style = (
        "body{margin:0;background:#f4f6f8;color:#17212b;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}"
        "main{max-width:1300px;margin:auto;padding:28px 20px 60px}.card{background:#fff;border:1px solid #d8dee7;"
        "border-radius:9px;padding:20px;margin:18px 0;overflow:auto}img{max-width:100%;height:auto}p{line-height:1.55}"
        "table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #d8dee7;padding:5px}th{background:#edf1f5}"
    )
    primary_fig = plot_methods(summary, output, PRIMARY_METHOD_ORDER, "v1_primary_clean_point_rmse_by_field.png")
    ablation_fig = plot_methods(summary, output, ABLATION_METHOD_ORDER, "v1_ablation_clean_point_rmse_by_field.png")
    metric_cols = ["clean_point_rmse_rel_median", "distance_to_manifold_rel_median", "velocity_rmse_loc_rel_median", "joint_euler_state_rmse_rel_median"]
    primary_table = summary[summary.method.isin(PRIMARY_METHOD_ORDER)][["field", "method_label", *metric_cols]].to_html(
        index=False, border=0, float_format=lambda x: f"{x:.4g}"
    )
    ablation_table = summary[summary.method.isin(ABLATION_METHOD_ORDER)][["field", "method_label", *metric_cols]].to_html(
        index=False, border=0, float_format=lambda x: f"{x:.4g}"
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>V1 field family</title><style>{style}</style></head><body><main>"
        "<h1>P3 Experiment V1: same manifold, different vector fields</h1>"
        f"<section class='card'><p>Flat unit disk (z=0 plane), n={provenance['n']}, "
        f"sigma_X={provenance['sigma_x']}, sigma_V={provenance['sigma_v']}, k={provenance['k']} "
        "(neighbor_count(n,d), same frozen rule as the canonical protocol). Shared VMF/Position-only "
        f"hyperparameters reused verbatim from the canonical protocol (identical across all 9 "
        f"scenarios; sourced here from '{provenance['shared_config_source_scenario']}'). 5 fields: "
        "source, sink, saddle, rotation, nonlinear -- each rescaled to its own median speed = 1 "
        "before noise, preserving within-field speed structure while keeping noise-to-signal "
        "comparable across fields. double_well is explicitly optional in current_plan.md and not run this "
        "round. All 7 methods (M0-M6) are computed for every field; shown here in the same two views "
        "as the canonical benchmark report -- primary (external baselines) and ablation "
        "(M4-to-M5-to-M6 pipeline capability) -- rather than only the ablation trio.</p></section>"
        "<section class='card'><h2>Primary comparison (M0, M1 GraphVelo, M2 Cosine Kernel, "
        f"M3 Joint Low-Rank, M6 ManfitVelo)</h2><img src='{image_uri(primary_fig)}'>{primary_table}</section>"
        "<section class='card'><h2>Ablation (M4 Local PCA, M5 Position-only MANFIT, M6 ManfitVelo)</h2>"
        f"<img src='{image_uri(ablation_fig)}'>{ablation_table}</section>"
        "</main></body></html>"
    )
    path = output / "v1_report.html"
    path.write_text(html, encoding="utf-8")
    parser = AuditParser()
    parser.feed(html)
    return {
        "self_contained_html": len(parser.images) == 2 and all(u.startswith("data:image/png;base64,") for u in parser.images) and not parser.external,
        "embedded_figure_count": len(parser.images),
        "expected_figure_count": 2,
    }


def main() -> None:
    output = ROOT / "results" / "v1_field_family"
    output.mkdir(parents=True, exist_ok=True)
    frame, provenance = run()
    summary = summarize(frame)
    frame.to_csv(output / "seed_metrics.csv", index=False)
    summary.to_csv(output / "summary_metrics.csv", index=False)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    audit = build_report(output, summary, provenance)
    expected_rows = len(FIELDS) * len(FINAL_SEEDS) * len(METHODS)
    # final_seeds_used_for_selection is correctly False by design (this is a
    # reporting-only run, nothing here selects anything) -- an informational
    # field, not a pass/fail check, so it's kept out of the all() below
    # (matching-value-means-pass would be backwards for this one field).
    pass_fail_checks = {
        "expected_seed_rows": bool(len(frame) == expected_rows),
        "no_duplicate_rows": bool(not frame.duplicated(["field", "seed", "method"]).any()),
        **audit,
    }
    checks = {**pass_fail_checks, "final_seeds_used_for_selection": False}
    checks["all_checks_pass"] = bool(all(pass_fail_checks.values()))
    (output / "sanity_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(output / "v1_report.html")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
