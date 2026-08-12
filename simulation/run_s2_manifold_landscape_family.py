"""P4 Experiment S2: same scalar landscape, different manifolds.

current_plan.md P4 / Experiment S2 (the scalar analog of P3's V2, "same intrinsic
dynamics, different manifolds" -- exactly parallel: S1 held geometry fixed
and varied the field/landscape, S2 holds the landscape fixed and varies
geometry). Reuses `run_v2_manifold_family.py`'s four embeddings verbatim
(flat_plane, sphere_patch, swiss_roll, saddle_surface -- same phi(u,v),
domains, unit normals, curvature-aware k selection) rather than
reimplementing them, so the two experiments are directly comparable.

One shared scalar landscape f(u,v), defined on each manifold's own (u,v)
chart via an affine remap to a common reference square [-1,1] x [-1,1]
(u_tilde, v_tilde) -- necessary because the four manifolds' *native* (u,v)
ranges don't share a common scale to begin with (longitude/colatitude for
the sphere, arc-length-like t/y for the swiss roll, plain xy for
flat_plane/saddle_surface -- V2 faced the same issue for its own dynamics
and resolved it structurally, not by literal coordinate-range matching;
same idea here). f itself is `run_s1_scalar_landscape_family`'s
`landscape_nonlinear_multimodal` (imported, not reimplemented) -- picked as
the single representative landscape because it's the most genuinely
nonlinear of S1's four, so this experiment isolates "does the SAME
nonlinear scalar landscape become harder or easier to recover as embedding
curvature changes" as cleanly as possible.

Ground-truth intrinsic gradient: because the landscape is defined via the
chart (u, v) rather than being read off directly from ambient coordinates,
recovering its ambient (tangent-space) gradient needs the local pullback
metric g_ij = <dphi/du_i, dphi/du_j>, not just the naive chain rule -- same
math already used by the existing `scalar_saddle` scenario in
`scripts/benchmark_scenarios.py::scalar_data` (the g11/g22/
g12 + 2x2 metric inverse block), reused here for consistency rather than
re-derived: grad_ambient = g^{-1} (df/du, df/dv) contracted against
(dphi/du, dphi/dv). This matters concretely for saddle_surface, whose
(u, v) parametrization has a nonzero g12 cross term (unlike the other three,
which happen to be orthogonal charts) -- an isotropic per-axis rescale
would silently be wrong there.

Same four pipelines and three metric layers as S1 (raw_local_regression,
geometry_only, joint_scalar_aware at the frozen scalar-branch protocol
lambda_v=0.0, oracle_gradient_joint; geometry/scalar/gradient metrics,
reusing S1's `local_scalar_smooth`). k(n,d): full two-stage curvature-aware
rule via `run_v2_manifold_family.curvature_aware_k_for_manifold` (NOT the
plain Stage-1 ceiling -- V2's own log.md entry documents catastrophic
failure on sphere_patch/swiss_roll without this refinement; S1 didn't need
it because its domain was flat throughout, but three of these four
manifolds are curved). n=480, sigma_X=0.05 (matches V2), sigma_S=0.08
(matches S1). 15 final seeds, reporting only.

    python simulation/run_s2_manifold_landscape_family.py
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
from simulation.benchmark_core import angle_mae, vector_rmse  # noqa: E402
from simulation.run_s1_scalar_landscape_family import (  # noqa: E402
    SCALAR_FROZEN_LAMBDA_V,
    SCALAR_FROZEN_SCALING,
    SCALAR_FROZEN_SHARED,
    LABELS,
    PIPELINES,
    local_scalar_smooth,
    landscape_nonlinear_multimodal,
)
from simulation.run_v2_manifold_family import (  # noqa: E402
    DOMAINS,
    MANIFOLDS,
    curvature_aware_k_for_manifold,
    phi,
    dphi_du,
    sample_uv,
    unit_normal,
)

FINAL_SEEDS = tuple(range(43000, 43015))
N = 480
INTRINSIC_DIMENSION = 2
SIGMA_X = 0.05  # matches V2
SIGMA_S = 0.08  # matches S1
EPS = 1e-12
REFERENCE_SEED = 90210  # matches V2's own reference seed convention, independent of FINAL_SEEDS
MESH_GRID_SIDE = 220  # matches V2's dense-mesh resolution for swiss_roll/saddle_surface projection

SADDLE_A = 0.45  # matches run_v2_manifold_family.SADDLE_A, needed here to build dphi/dv


def dphi_dv(name: str, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """d(phi)/dv -- V2 never needed this (its dynamics only pushed forward
    d(phi)/du), but a general scalar-landscape gradient needs the full
    Jacobian to build the pullback metric. Kept alongside V2's own
    dphi_du/phi rather than inside that module, since it's S2-specific."""
    if name == "flat_plane":
        return np.tile([0.0, 1.0, 0.0], (len(u), 1))
    if name == "sphere_patch":
        return np.c_[np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), -np.sin(v)]
    if name == "swiss_roll":
        return np.tile([0.0, 1.0, 0.0], (len(u), 1))
    if name == "saddle_surface":
        return np.c_[np.zeros_like(u), np.ones_like(u), -2 * SADDLE_A * v]
    raise KeyError(name)


def _normalized_uv(name: str, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Affine remap of this manifold's own (u,v) domain to the shared
    reference square [-1,1]x[-1,1] -- see module docstring. Returns the
    remapped coordinates plus the two constant Jacobian factors
    (d u_tilde/du, d v_tilde/dv) needed for the chain rule."""
    (u_lo, u_hi), (v_lo, v_hi) = DOMAINS[name]["u"], DOMAINS[name]["v"]
    du_tilde = 2.0 / (u_hi - u_lo)
    dv_tilde = 2.0 / (v_hi - v_lo)
    u_tilde = du_tilde * (u - u_lo) - 1.0
    v_tilde = dv_tilde * (v - v_lo) - 1.0
    return u_tilde, v_tilde, du_tilde, dv_tilde


def landscape_and_ambient_gradient(name: str, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """f(u,v) (via the shared normalized landscape) and its ambient
    (tangent-space) gradient at phi(u,v), using the pullback-metric
    construction described in the module docstring."""
    u_tilde, v_tilde, du_tilde, dv_tilde = _normalized_uv(name, u, v)
    f_tilde, grad_tilde = landscape_nonlinear_multimodal(u_tilde, v_tilde)
    df_du = grad_tilde[:, 0] * du_tilde
    df_dv = grad_tilde[:, 1] * dv_tilde

    Ju = dphi_du(name, u, v)
    Jv = dphi_dv(name, u, v)
    g11 = np.sum(Ju * Ju, axis=1)
    g22 = np.sum(Jv * Jv, axis=1)
    g12 = np.sum(Ju * Jv, axis=1)
    det = np.maximum(g11 * g22 - g12 * g12, EPS)
    coef1 = (g22 * df_du - g12 * df_dv) / det
    coef2 = (-g12 * df_du + g11 * df_dv) / det
    grad_ambient = coef1[:, None] * Ju + coef2[:, None] * Jv
    return f_tilde, grad_ambient


_MANIFOLD_SEED_TAG = {"flat_plane": 1, "sphere_patch": 2, "swiss_roll": 3, "saddle_surface": 4}


def _gradient_scale(name: str) -> float:
    # Same fixed-reference-seed convention as V2's own SPEED_SCALE, for the
    # same reason: a global, final-seed-independent rescale so sigma_S means
    # a comparable relative noise level across four differently-curved
    # manifolds while preserving each manifold's own real gradient-magnitude
    # variation. Must not use Python's hash() -- see V2's own note.
    rng = np.random.default_rng(np.random.SeedSequence([REFERENCE_SEED, _MANIFOLD_SEED_TAG[name]]))
    u, v = sample_uv(name, rng, 5000)
    _, grad = landscape_and_ambient_gradient(name, u, v)
    speed = np.linalg.norm(grad, axis=1)
    return float(np.median(speed[speed > EPS]))


GRADIENT_SCALE = {name: _gradient_scale(name) for name in MANIFOLDS}


def manifold_data(name: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    u, v = sample_uv(name, rng, N)
    X3 = phi(name, u, v)
    f_raw, grad_raw = landscape_and_ambient_gradient(name, u, v)
    scale = max(GRADIENT_SCALE[name], EPS)
    F = f_raw / scale
    G = grad_raw / scale
    N3 = unit_normal(name, u, v, X3)

    Y = X3 + rng.normal(scale=SIGMA_X, size=(N, 1)) * N3
    scalar_obs = F + rng.normal(scale=SIGMA_S, size=N)
    return {
        "Y": Y, "scalar": scalar_obs, "P": X3, "scalar_clean": F, "truth": G,
        "d": INTRINSIC_DIMENSION, "manifold": name,
    }


def _dense_position_mesh(name: str) -> np.ndarray:
    (u_lo, u_hi), (v_lo, v_hi) = DOMAINS[name]["u"], DOMAINS[name]["v"]
    uu, vv = np.meshgrid(np.linspace(u_lo, u_hi, MESH_GRID_SIDE), np.linspace(v_lo, v_hi, MESH_GRID_SIDE), indexing="ij")
    return phi(name, uu.ravel(), vv.ravel())


_MESH_CACHE: dict[str, tuple[np.ndarray, NearestNeighbors]] = {}


def distance_to_manifold(name: str, Xhat: np.ndarray) -> float:
    """Same style as V2's own evaluation_targets: closed-form projection for
    flat_plane/sphere_patch, dense-mesh nearest-neighbor for swiss_roll/
    saddle_surface (position-only -- S2 has no vector-field target to look
    up alongside it, unlike V2)."""
    if name == "flat_plane":
        position = np.asarray(Xhat, float).copy()
        position[:, 2] = 0.0
        (u_lo, u_hi), (v_lo, v_hi) = DOMAINS[name]["u"], DOMAINS[name]["v"]
        position[:, 0] = np.clip(position[:, 0], u_lo, u_hi)
        position[:, 1] = np.clip(position[:, 1], v_lo, v_hi)
    elif name == "sphere_patch":
        radius = np.linalg.norm(Xhat, axis=1)
        position = Xhat / np.maximum(radius[:, None], EPS)
    else:
        if name not in _MESH_CACHE:
            mesh_X = _dense_position_mesh(name)
            _MESH_CACHE[name] = (mesh_X, NearestNeighbors(n_neighbors=1).fit(mesh_X))
        mesh_X, index = _MESH_CACHE[name]
        nn = index.kneighbors(Xhat, return_distance=False)[:, 0]
        position = mesh_X[nn]
    return float(np.sqrt(np.mean(np.sum((Xhat - position) ** 2, axis=1))))


def run_pipeline(pipeline: str, data: dict, k: int, seed: int) -> dict:
    Y, scalar_obs, truth = data["Y"], data["scalar"], data["truth"]

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
    name = data["manifold"]
    metrics = {
        "clean_point_rmse": float("nan") if out["position"] is None else float(
            np.sqrt(np.mean(np.sum((out["position"] - data["P"]) ** 2, axis=1)))
        ),
        "distance_to_manifold": float("nan") if out["position"] is None else distance_to_manifold(name, out["position"]),
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
    manifold_k = {}
    for name in MANIFOLDS:
        manifold_k[name], _ = curvature_aware_k_for_manifold(name)

    rows = []
    for name in MANIFOLDS:
        k = manifold_k[name]
        for seed in FINAL_SEEDS:
            data = manifold_data(name, seed)
            for pipeline in PIPELINES:
                metrics = evaluate(pipeline, data, k, seed)
                rows.append(
                    {
                        "manifold": name, "seed": seed, "pipeline": pipeline,
                        "pipeline_label": LABELS[pipeline], "k": k, **metrics,
                    }
                )
    frame = pd.DataFrame(rows)
    provenance = {
        "n": N, "d": INTRINSIC_DIMENSION, "sigma_x": SIGMA_X, "sigma_s": SIGMA_S,
        "scalar_frozen_lambda_v": SCALAR_FROZEN_LAMBDA_V, "scalar_frozen_scaling": SCALAR_FROZEN_SCALING,
        "scalar_frozen_shared": SCALAR_FROZEN_SHARED,
        "final_seeds": list(FINAL_SEEDS), "manifolds": MANIFOLDS, "manifold_k": manifold_k,
        "k_rule": "curvature_aware_neighbor_count on TUNING_SEEDS, reused from run_v2_manifold_family",
        "landscape": "run_s1_scalar_landscape_family.landscape_nonlinear_multimodal, on each manifold's "
        "own (u,v) chart affinely remapped to [-1,1]x[-1,1]",
        "gradient_scale": GRADIENT_SCALE, "reference_seed": REFERENCE_SEED,
        "known_limitation": "fit_scalar_gradient_manfit's inner_T/eta_g/outer_iterations/"
        "gradient_n_neighbors are still function defaults, never tier-3 selected -- theta/kappa are "
        "the exception, reused from the vector-field's frozen values (see S1's docstring)",
    }
    return frame, provenance


HEADLINE_METRICS = ("clean_point_rmse", "distance_to_manifold", "scalar_rmse", "gradient_rmse", "gradient_angle_mae")


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in frame.groupby(["manifold", "pipeline", "pipeline_label"], sort=False):
        row = dict(zip(["manifold", "pipeline", "pipeline_label"], keys))
        for metric in HEADLINE_METRICS:
            row[f"{metric}_median"] = float(g[metric].median())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_manifolds(summary: pd.DataFrame, output: Path) -> Path:
    fig, axes = plt.subplots(1, len(MANIFOLDS), figsize=(4.4 * len(MANIFOLDS), 4), constrained_layout=True)
    colors = {
        "raw_local_regression": "#7a7a7a", "geometry_only": "#b8860b",
        "joint_scalar_aware": "#1f6f5c", "oracle_gradient_joint": "#2f5fa8",
    }
    for ax, name in zip(axes, MANIFOLDS):
        sub = summary[summary.manifold == name]
        x = np.arange(len(PIPELINES))
        heights = [float(sub[sub.pipeline == p]["gradient_rmse_median"].iloc[0]) for p in PIPELINES]
        ax.bar(x, heights, color=[colors[p] for p in PIPELINES])
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[p] for p in PIPELINES], fontsize=6, rotation=25, ha="right")
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("gradient_rmse (median)")
    path = output / "figures" / "s2_gradient_rmse_by_manifold.png"
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
    fig_path = plot_manifolds(summary, output)
    table = summary[
        ["manifold", "pipeline_label", "clean_point_rmse_median", "distance_to_manifold_median",
         "scalar_rmse_median", "gradient_rmse_median", "gradient_angle_mae_median"]
    ].to_html(index=False, border=0, float_format=lambda x: f"{x:.4g}")
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>S2 manifold landscape family</title><style>{style}</style></head><body><main>"
        "<h1>P4 Experiment S2: same scalar landscape, different manifolds</h1>"
        f"<section class='card'><p>One shared nonlinear scalar landscape (S1's log-sum-exp two-well), "
        f"transported to four embeddings (flat_plane, sphere_patch, swiss_roll, saddle_surface -- reused "
        f"verbatim from V2). n={provenance['n']}, sigma_X={provenance['sigma_x']}, sigma_S={provenance['sigma_s']}. "
        f"Scalar branch frozen protocol: lambda_v={provenance['scalar_frozen_lambda_v']}, "
        f"scaling={provenance['scalar_frozen_scaling']!r}. k recomputed per manifold via the full "
        f"curvature-aware rule ({provenance['manifold_k']}). "
        f"<b>Known limitation</b>: {provenance['known_limitation']}.</p></section>"
        f"<section class='card'><img src='{image_uri(fig_path)}'></section>"
        f"<section class='card'><h2>Median results</h2>{table}</section>"
        "</main></body></html>"
    )
    path = output / "s2_report.html"
    path.write_text(html, encoding="utf-8")
    parser = AuditParser()
    parser.feed(html)
    return {
        "self_contained_html": len(parser.images) == 1 and parser.images[0].startswith("data:image/png;base64,") and not parser.external,
        "embedded_figure_count": len(parser.images),
        "expected_figure_count": 1,
    }


def main() -> None:
    output = ROOT / "results" / "s2_manifold_landscape_family"
    output.mkdir(parents=True, exist_ok=True)
    frame, provenance = run()
    summary = summarize(frame)
    frame.to_csv(output / "seed_metrics.csv", index=False)
    summary.to_csv(output / "summary_metrics.csv", index=False)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    audit = build_report(output, summary, provenance)
    expected_rows = len(MANIFOLDS) * len(FINAL_SEEDS) * len(PIPELINES)
    pass_fail_checks = {
        "expected_seed_rows": bool(len(frame) == expected_rows),
        "no_duplicate_rows": bool(not frame.duplicated(["manifold", "seed", "pipeline"]).any()),
        "no_nan_inf": bool((frame.nan_inf_count == 0).all()),
        **audit,
    }
    checks = {**pass_fail_checks, "final_seeds_used_for_selection": False}
    checks["all_checks_pass"] = bool(all(pass_fail_checks.values()))
    (output / "sanity_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(output / "s2_report.html")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
