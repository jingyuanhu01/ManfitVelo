"""P4 Experiment S1: same manifold, different scalar landscapes.

current_plan.md P4 / Experiment S1 (the scalar analog of P3's V1, "same manifold,
different vector fields" -- same flat unit disk embedding/noise convention as
`run_v1_field_family.py`, for direct cross-experiment comparability). Fixed
flat 2D disk (unit radius, z=0 plane in R^3), n and noise held constant;
only the scalar landscape f(x,y) changes across four types (current_plan.md's own
list):

    single_basin        f = x^2 + y^2
    double_well         f = (x^2-a^2)^2 + c*y^2,  a=0.5, c=1.0
    saddle              f = x^2 - y^2
    nonlinear_multimodal  a log-sum-exp soft two-well landscape (genuinely
                           nonlinear gradient, unlike the three polynomial
                           landscapes above -- current_plan.md explicitly asks for
                           a "mixture/log-sum-exp" construction here):
                           f = -tau*log(exp(-d1^2/tau) + exp(-d2^2/tau)),
                           d1^2=(x-0.5)^2+y^2, d2^2=(x+0.5)^2+y^2, tau=0.15

Each landscape's gradient is rescaled to a median norm of 1 over its own
sampled points before noise is added (same convention as V1's field
rescaling), and f itself is rescaled by the SAME constant (linear in f, so
this keeps grad(f) consistent with the rescaled f with no extra derivation).

Four pipelines, exactly current_plan.md's own list:

    raw_local_regression   estimate_gradient_from_neighbors on the raw noisy
                            observations, no manifold fitting at all (no
                            position output).
    geometry_only           Local PCA denoises position (M4-style, current_plan.md's
                            "geometry-only manifold fitting"), then gradient is
                            estimated post-hoc from the denoised positions --
                            isolates whether pure geometric denoising alone
                            (no scalar-aware coupling) already helps, the way
                            `downstream_velocity` does for M4 in the vector
                            pipeline.
    joint_scalar_aware       fit_scalar_gradient_manfit at the scalar branch's
                            now-frozen protocol (lambda_v=SCALAR_FROZEN_
                            LAMBDA_V=0.0, see parameter_rules.md SS3b --
                            "rank" scaling is a no-op at lambda_v=0, kept for
                            documentation fidelity). Note lambda_v=0 does NOT
                            disable the velocity-aware neighbor reranking
                            mechanism (same caveat as run_lambda_sensitivity.py
                            makes for the vector-field lambda_v=0 case) -- only
                            the tangent-covariance blend is off.
    oracle_gradient_joint     Same frozen config, but the estimated gradient is
                            replaced by the exact ground-truth gradient at
                            every outer iteration (fit_scalar_gradient_manfit's
                            oracle_gradient=). Isolates joint-fitting error
                            from local-regression error, same logic as P4.1's
                            own oracle ablation.

Three metric layers per current_plan.md's "Scalar metrics" section:

    Geometry     distance_to_manifold(Xhat), ||Xhat - P||  (clean_point_rmse)
    Scalar       RMSE(f_hat, f) -- f_hat is a simple local-neighbor-average
                 scalar denoiser at the FINAL fitted positions (see
                 local_scalar_smooth() below): there is no existing scalar-
                 value reconstruction anywhere in this pipeline (P4.1 never
                 needed one, only gradient/position metrics), so this is a
                 new, deliberately simple baseline -- documented here rather
                 than silently invented. raw_local_regression has no fitted
                 position, so its f_hat is just the raw noisy observation
                 (no smoothing at all, matching its "no processing" name).
    Gradient     ||grad_hat - grad_true|| (RMSE) and gradient angle MAE.

k(n,d): plain neighbor_count(N, INTRINSIC_DIMENSION) Stage-1 rule, no
curvature-aware refinement needed (flat domain, same reasoning as V1).
15 final seeds, reporting only -- nothing here selects any hyperparameter,
so there is no final-seed-leakage concern (frozen config was already chosen
via TUNING_SEEDS in run_scalar_lambda_v_selection.py).

UPDATE (2026-08-12, same day): theta/kappa are now reused directly from the
vector-field M6's own frozen shared values (0.02, 0.0) by direct user
decision (parameter_rules.md SS3c) -- NOT its T/eta_g (3, 0.7), which was
tried first and found to overshoot badly, since fit_scalar_gradient_manfit's
outer_iterations=4 loop means 4x the position-update budget at that eta_g
compared to M6's single fit() call (see SS3c for the numbers). inner_T/eta_g
stay at fit_scalar_gradient_manfit's own defaults (2, 0.35).

KNOWN LIMITATION, flagged rather than silently used: fit_scalar_gradient_
manfit's own inner_T/eta_g/outer_iterations/gradient_n_neighbors are still
just the function's built-in defaults -- these have never gone through the
tier-3 grid-search selection process the vector-field's own shared
hyperparameters did (parameter_rules.md SS3). lambda_v/scaling-mode (SS3b)
and theta/kappa (SS3c, this update) are the two exceptions. Left as a known
gap, not fixed here.

    python simulation/run_s1_scalar_landscape_family.py
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
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pca_denoisers import local_pca_denoise  # noqa: E402
from scripts.scalar_potential_manfit import estimate_gradient_from_neighbors, fit_scalar_gradient_manfit  # noqa: E402
from simulation.benchmark_core import angle_mae, neighbor_count, vector_rmse  # noqa: E402

FINAL_SEEDS = tuple(range(43000, 43015))
N = 480
INTRINSIC_DIMENSION = 2
SIGMA_X = 0.05  # position noise, matches V1's convention exactly
SIGMA_S = 0.08  # scalar-observation noise, matches scalar_s_curve/scalar_saddle's field_noise
EPS = 1e-12

SCALAR_FROZEN_LAMBDA_V = 0.0  # parameter_rules.md SS3b (2026-08-12 tuning-seed selection)
SCALAR_FROZEN_SCALING = "rank"  # no-op at lambda_v=0.0, kept for documentation fidelity
# theta/kappa reused from the vector-field M6's own frozen values (parameter_rules.md SS3c);
# inner_T/eta_g deliberately NOT copied (overshoots -- see SS3c/run_scalar_lambda_v_selection.py).
SCALAR_FROZEN_SHARED = dict(theta=0.02, kappa=0.0)

PIPELINES = ("raw_local_regression", "geometry_only", "joint_scalar_aware", "oracle_gradient_joint")
LABELS = {
    "raw_local_regression": "Raw local regression",
    "geometry_only": "Geometry-only (Local PCA) + gradient",
    "joint_scalar_aware": "Joint scalar-aware (frozen)",
    "oracle_gradient_joint": "Oracle-gradient joint",
}


def landscape_single_basin(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = x**2 + y**2
    grad = np.c_[2 * x, 2 * y]
    return f, grad


def landscape_double_well(x: np.ndarray, y: np.ndarray, a: float = 0.5, c: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    f = (x**2 - a**2) ** 2 + c * y**2
    grad = np.c_[4 * x * (x**2 - a**2), 2 * c * y]
    return f, grad


def landscape_saddle(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = x**2 - y**2
    grad = np.c_[2 * x, -2 * y]
    return f, grad


def landscape_nonlinear_multimodal(x: np.ndarray, y: np.ndarray, tau: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    d1_sq = (x - 0.5) ** 2 + y**2
    d2_sq = (x + 0.5) ** 2 + y**2
    g1 = np.exp(-d1_sq / tau)
    g2 = np.exp(-d2_sq / tau)
    denom = np.maximum(g1 + g2, EPS)
    f = -tau * np.log(denom)
    dfdx = 2.0 * (g1 * (x - 0.5) + g2 * (x + 0.5)) / denom
    dfdy = 2.0 * y  # both terms share the same y-dependence, see module docstring derivation
    return f, np.c_[dfdx, dfdy]


LANDSCAPES = {
    "single_basin": landscape_single_basin,
    "double_well": landscape_double_well,
    "saddle": landscape_saddle,
    "nonlinear_multimodal": landscape_nonlinear_multimodal,
}


def raw_landscape(name: str, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return LANDSCAPES[name](x, y)


def disk_data(landscape_name: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(0, 1, N))
    theta = rng.uniform(0, 2 * np.pi, N)
    x, y = r * np.cos(theta), r * np.sin(theta)
    X = np.c_[x, y, np.zeros(N)]

    f_raw, grad_raw_2d = raw_landscape(landscape_name, x, y)
    grad_raw = np.c_[grad_raw_2d, np.zeros(N)]  # ambient R^3, matching Y/V1's convention
    speed = np.linalg.norm(grad_raw, axis=1)
    scale = float(np.median(speed[speed > EPS])) if np.any(speed > EPS) else 1.0
    F = f_raw / max(scale, EPS)
    G = grad_raw / max(scale, EPS)

    normal = np.tile([0.0, 0.0, 1.0], (N, 1))
    Y = X + rng.normal(scale=SIGMA_X, size=(N, 1)) * normal
    scalar_obs = F + rng.normal(scale=SIGMA_S, size=N)
    return {
        "Y": Y, "scalar": scalar_obs, "P": X, "scalar_clean": F, "truth": G,
        "d": INTRINSIC_DIMENSION, "landscape_name": landscape_name, "scale": scale,
    }


def local_scalar_smooth(Xhat: np.ndarray, scalar_obs: np.ndarray, k: int) -> np.ndarray:
    """Simple local-neighbor-average scalar denoiser at the given positions.

    Deliberately simple, uniform-weighted kNN average -- there is no existing
    scalar-value reconstruction anywhere in this pipeline to reuse (unlike
    `downstream_velocity` for vectors), see module docstring. Not part of any
    fitting pipeline; purely a post-hoc reporting metric.
    """
    k = min(int(k), len(Xhat) - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(Xhat)
    _, indices = nbrs.kneighbors(Xhat)
    return scalar_obs[indices[:, 1:]].mean(axis=1)


def distance_to_disk(Xhat: np.ndarray) -> float:
    """Distance to the true unit disk (z=0, radius<=1) -- clip-to-disk style,
    same convention as V1's evaluation_targets/flat_rotation_annulus."""
    position = np.asarray(Xhat, float).copy()
    position[:, 2] = 0.0
    radius = np.linalg.norm(position[:, :2], axis=1)
    clipped = np.clip(radius, 0.0, 1.0)
    scale = np.where(radius > EPS, clipped / np.maximum(radius, EPS), 1.0)
    position[:, :2] *= scale[:, None]
    return float(np.sqrt(np.mean(np.sum((Xhat - position) ** 2, axis=1))))


def run_pipeline(pipeline: str, data: dict, k: int, seed: int) -> dict:
    Y, scalar_obs, P, truth = data["Y"], data["scalar"], data["P"], data["truth"]

    if pipeline == "raw_local_regression":
        grad_hat = estimate_gradient_from_neighbors(Y, scalar_obs, n_neighbors=k, ridge=5e-2)
        return {"position": None, "gradient": grad_hat, "scalar_hat": scalar_obs.copy()}

    if pipeline == "geometry_only":
        Xhat = local_pca_denoise(Y, INTRINSIC_DIMENSION, n_neighbors=k, return_info=False)
        grad_hat = estimate_gradient_from_neighbors(Xhat, scalar_obs, n_neighbors=k, ridge=5e-2)
        scalar_hat = local_scalar_smooth(Xhat, scalar_obs, k)
        return {"position": Xhat, "gradient": grad_hat, "scalar_hat": scalar_hat}

    kwargs = dict(
        k=k, lambda_v=SCALAR_FROZEN_LAMBDA_V, lambda_v_confidence_scaling=SCALAR_FROZEN_SCALING,
        random_state=seed, **SCALAR_FROZEN_SHARED,
    )
    if pipeline == "oracle_gradient_joint":
        kwargs["oracle_gradient"] = truth
    elif pipeline != "joint_scalar_aware":
        raise KeyError(pipeline)
    result = fit_scalar_gradient_manfit(Y, scalar_obs, **kwargs)
    scalar_hat = local_scalar_smooth(result["position"], scalar_obs, k)
    return {"position": result["position"], "gradient": result["gradient"], "scalar_hat": scalar_hat}


def evaluate(pipeline: str, data: dict, k: int, seed: int) -> dict:
    out = run_pipeline(pipeline, data, k, seed)
    valid = np.ones(N, dtype=bool)
    metrics = {
        "clean_point_rmse": float("nan") if out["position"] is None else float(
            np.sqrt(np.mean(np.sum((out["position"] - data["P"]) ** 2, axis=1)))
        ),
        "distance_to_manifold": float("nan") if out["position"] is None else distance_to_disk(out["position"]),
        "scalar_rmse": float(np.sqrt(np.mean((out["scalar_hat"] - data["scalar_clean"]) ** 2))),
        "gradient_rmse": vector_rmse(out["gradient"], data["truth"], valid),
        "gradient_angle_mae": angle_mae(out["gradient"], data["truth"], valid),
    }
    nan_inf_count = int(np.sum(~np.isfinite(out["gradient"])))
    if out["position"] is not None:
        nan_inf_count += int(np.sum(~np.isfinite(out["position"])))
    metrics["nan_inf_count"] = nan_inf_count
    return metrics


def run() -> tuple[pd.DataFrame, dict]:
    k = neighbor_count(N, INTRINSIC_DIMENSION)
    rows = []
    for landscape_name in LANDSCAPES:
        for seed in FINAL_SEEDS:
            data = disk_data(landscape_name, seed)
            for pipeline in PIPELINES:
                metrics = evaluate(pipeline, data, k, seed)
                rows.append(
                    {
                        "landscape": landscape_name, "seed": seed, "pipeline": pipeline,
                        "pipeline_label": LABELS[pipeline], "k": k, **metrics,
                    }
                )
    frame = pd.DataFrame(rows)
    provenance = {
        "n": N, "d": INTRINSIC_DIMENSION, "k": k, "sigma_x": SIGMA_X, "sigma_s": SIGMA_S,
        "scalar_frozen_lambda_v": SCALAR_FROZEN_LAMBDA_V, "scalar_frozen_scaling": SCALAR_FROZEN_SCALING,
        "scalar_frozen_shared": SCALAR_FROZEN_SHARED,
        "final_seeds": list(FINAL_SEEDS), "landscapes": list(LANDSCAPES.keys()),
        "known_limitation": "fit_scalar_gradient_manfit's inner_T/eta_g/outer_iterations/"
        "gradient_n_neighbors are still function defaults, never tier-3 selected -- theta/kappa are "
        "the exception, reused from the vector-field's frozen values (parameter_rules.md SS3c)",
    }
    return frame, provenance


HEADLINE_METRICS = ("clean_point_rmse", "distance_to_manifold", "scalar_rmse", "gradient_rmse", "gradient_angle_mae")


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in frame.groupby(["landscape", "pipeline", "pipeline_label"], sort=False):
        row = dict(zip(["landscape", "pipeline", "pipeline_label"], keys))
        for metric in HEADLINE_METRICS:
            row[f"{metric}_median"] = float(g[metric].median())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_landscapes(summary: pd.DataFrame, output: Path) -> Path:
    landscapes = list(LANDSCAPES.keys())
    fig, axes = plt.subplots(1, len(landscapes), figsize=(4.4 * len(landscapes), 4), constrained_layout=True)
    colors = {
        "raw_local_regression": "#7a7a7a", "geometry_only": "#b8860b",
        "joint_scalar_aware": "#1f6f5c", "oracle_gradient_joint": "#2f5fa8",
    }
    for ax, landscape_name in zip(axes, landscapes):
        sub = summary[summary.landscape == landscape_name]
        x = np.arange(len(PIPELINES))
        heights = [float(sub[sub.pipeline == p]["gradient_rmse_median"].iloc[0]) for p in PIPELINES]
        ax.bar(x, heights, color=[colors[p] for p in PIPELINES])
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[p] for p in PIPELINES], fontsize=6, rotation=25, ha="right")
        ax.set_title(landscape_name, fontsize=10)
        ax.set_ylabel("gradient_rmse (median)")
    path = output / "figures" / "s1_gradient_rmse_by_landscape.png"
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
    fig_path = plot_landscapes(summary, output)
    table = summary[
        ["landscape", "pipeline_label", "clean_point_rmse_median", "distance_to_manifold_median",
         "scalar_rmse_median", "gradient_rmse_median", "gradient_angle_mae_median"]
    ].to_html(index=False, border=0, float_format=lambda x: f"{x:.4g}")
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>S1 scalar landscape family</title><style>{style}</style></head><body><main>"
        "<h1>P4 Experiment S1: same manifold, different scalar landscapes</h1>"
        f"<section class='card'><p>Flat unit disk (z=0 plane, same embedding as V1), n={provenance['n']}, "
        f"sigma_X={provenance['sigma_x']}, sigma_S={provenance['sigma_s']}, k={provenance['k']} "
        f"(neighbor_count(n,d)). Scalar branch frozen protocol: lambda_v={provenance['scalar_frozen_lambda_v']}, "
        f"scaling={provenance['scalar_frozen_scaling']!r} (parameter_rules.md SS3b). 4 landscapes: "
        "single_basin, double_well, saddle, nonlinear_multimodal (log-sum-exp). "
        f"<b>Known limitation</b>: {provenance['known_limitation']}.</p></section>"
        f"<section class='card'><img src='{image_uri(fig_path)}'></section>"
        f"<section class='card'><h2>Median results</h2>{table}</section>"
        "</main></body></html>"
    )
    path = output / "s1_report.html"
    path.write_text(html, encoding="utf-8")
    parser = AuditParser()
    parser.feed(html)
    return {
        "self_contained_html": len(parser.images) == 1 and parser.images[0].startswith("data:image/png;base64,") and not parser.external,
        "embedded_figure_count": len(parser.images),
        "expected_figure_count": 1,
    }


def main() -> None:
    output = ROOT / "results" / "s1_scalar_landscape_family"
    output.mkdir(parents=True, exist_ok=True)
    frame, provenance = run()
    summary = summarize(frame)
    frame.to_csv(output / "seed_metrics.csv", index=False)
    summary.to_csv(output / "summary_metrics.csv", index=False)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    audit = build_report(output, summary, provenance)
    expected_rows = len(LANDSCAPES) * len(FINAL_SEEDS) * len(PIPELINES)
    pass_fail_checks = {
        "expected_seed_rows": bool(len(frame) == expected_rows),
        "no_duplicate_rows": bool(not frame.duplicated(["landscape", "seed", "pipeline"]).any()),
        "no_nan_inf": bool((frame.nan_inf_count == 0).all()),
        **audit,
    }
    checks = {**pass_fail_checks, "final_seeds_used_for_selection": False}
    checks["all_checks_pass"] = bool(all(pass_fail_checks.values()))
    (output / "sanity_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(output / "s1_report.html")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
