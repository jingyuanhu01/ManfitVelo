"""P3 Experiment V2: same intrinsic dynamics, different manifolds.

current_plan.md P3/V2. One shared latent dynamics, u_dot=1, v_dot=0 (constant unit
"translation" in the first latent coordinate; the optional rotational
variant, u_dot=0 in polar coordinates, is skipped this round per the plan's
own "视计算量决定" -- compute-budget-permitting -- language), pushed forward
through four different embeddings phi(u, v) -> R^3:

    X = phi(u, v)
    V = D(phi)(u, v) @ (1, 0)^T     -- the Jacobian's first column, i.e.
                                        d(phi)/du, since v_dot=0

so every embedding observes literally the same intrinsic dynamics; only the
extrinsic geometry (curvature, embedding complexity) differs. This is what
makes it a controlled experiment rather than "four more scenarios."

Embeddings:
  flat_plane     phi(u,v) = (u, v, 0), u,v in [-1,1] -- flat, trivial Jacobian
  sphere_patch   phi(u,v) = (sin v cos u, sin v sin u, cos v), u in [0,2pi)
                 (longitude), v in [pi/3, 2pi/3] (colatitude, +-30 deg from
                 the equator, kept away from the poles where the Jacobian
                 degenerates). d(phi)/du = (-sin v sin u, sin v cos u, 0),
                 which in ambient coordinates is exactly (-y, x, 0) -- the
                 same rotation-around-z form as flat_rotation_annulus's own
                 field, just restricted to the sphere's surface.
  swiss_roll     SAME phi(u,v) as the canonical swiss_roll scenario (u=t in
                 [1.5pi,3.5pi], v=y in [-1,1], one winding -- see
                 scripts/benchmark_scenarios.py::vector_data)
  saddle_surface SAME phi(u,v) as the canonical saddle_surface scenario
                 (u,v in [-1,1], a=0.45)

Deliberate departure from the canonical swiss_roll/saddle_surface velocity
convention: those scenarios normalize velocity to *unit speed at every
point* (v_hat = raw / |raw|), which would erase exactly the "how does
embedding curvature change the induced speed" signal this experiment exists
to show. V2 instead uses the RAW (unnormalized) Jacobian pushforward,
rescaled by one GLOBAL constant per manifold (median speed over a large
fixed reference sample, seed 90210, independent of any FINAL_SEEDS draw) --
same convention as run_v1_field_family.py's per-field rescaling, and the
only way to keep sigma_V=0.10 meaning a comparable relative noise level
across four very differently-curved manifolds while preserving each
manifold's own real speed variation.

Question: with the *dynamics* held fixed, how does embedding curvature
change recovery?

Frozen shared hyperparameters reused verbatim from the canonical protocol
(same as run_v1_field_family.py); k(n,d) freshly computed per manifold via
neighbor_count(n,d). n=480, sigma_X=0.05, sigma_V=0.10 for every manifold
(matches the canonical swiss_roll/saddle_surface settings exactly). 15 final
seeds, reporting only.

    python simulation/run_v2_manifold_family.py
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
from simulation.benchmark_core import (  # noqa: E402
    angle_mae,
    curvature_aware_neighbor_count,
    curvature_probe_k_grid,
    joint_error,
    load_frozen_config,
    local_pca_normal_residual,
    observed_tau,
    vector_rmse,
)
from simulation.run_manfitvelo_benchmark import TUNING_SEEDS  # noqa: E402

FINAL_SEEDS = tuple(range(43000, 43015))
N = 480
INTRINSIC_DIMENSION = 2
SIGMA_X = 0.05
SIGMA_V = 0.10
EPS = 1e-12
REFERENCE_SEED = 90210  # fixed, independent of FINAL_SEEDS -- only used to size the global speed rescale
MESH_GRID_SIDE = 220  # dense (u,v) grid resolution for swiss_roll/saddle_surface NN evaluation

METHODS = ("ambient_noisy", "cosine_kernel", "graphvelo", "joint_low_rank", "local_pca", "position_only_manfit", "manfitvelo")
LABELS = {
    "ambient_noisy": "Ambient noisy input", "cosine_kernel": "Cosine kernel", "graphvelo": "GraphVelo",
    "joint_low_rank": "Joint Low-Rank (M3)", "local_pca": "Local PCA",
    "position_only_manfit": "Position-only MANFIT", "manfitvelo": "ManfitVelo",
}

SWISS_ROLL_TMAX = 3.5 * np.pi
SADDLE_A = 0.45


def phi(name: str, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    if name == "flat_plane":
        return np.c_[u, v, np.zeros_like(u)]
    if name == "sphere_patch":
        return np.c_[np.sin(v) * np.cos(u), np.sin(v) * np.sin(u), np.cos(v)]
    if name == "swiss_roll":
        return np.c_[u * np.cos(u) / SWISS_ROLL_TMAX, v, u * np.sin(u) / SWISS_ROLL_TMAX]
    if name == "saddle_surface":
        return np.c_[u, v, SADDLE_A * (u * u - v * v)]
    raise KeyError(name)


def dphi_du(name: str, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """d(phi)/du -- the pushforward of the shared latent dynamics (u_dot,
    v_dot) = (1, 0). Raw, unnormalized (see module docstring)."""
    if name == "flat_plane":
        return np.tile([1.0, 0.0, 0.0], (len(u), 1))
    if name == "sphere_patch":
        return np.c_[-np.sin(v) * np.sin(u), np.sin(v) * np.cos(u), np.zeros_like(u)]
    if name == "swiss_roll":
        return np.c_[
            (np.cos(u) - u * np.sin(u)) / SWISS_ROLL_TMAX,
            np.zeros_like(u),
            (np.sin(u) + u * np.cos(u)) / SWISS_ROLL_TMAX,
        ]
    if name == "saddle_surface":
        return np.c_[np.ones_like(u), np.zeros_like(u), 2 * SADDLE_A * u]
    raise KeyError(name)


def unit_normal(name: str, u: np.ndarray, v: np.ndarray, X3: np.ndarray) -> np.ndarray:
    if name == "flat_plane":
        return np.tile([0.0, 0.0, 1.0], (len(u), 1))
    if name == "sphere_patch":
        return X3.copy()  # outward normal of a unit sphere is the point itself
    if name == "swiss_roll":
        # d(phi)/dv = (0,1,0) exactly for this embedding, so a 90-degree
        # rotation of d(phi)/du within the XZ-plane is already orthogonal to
        # both tangent directions -- same trick as the canonical scenario.
        d = dphi_du(name, u, v)
        raw = np.c_[-d[:, 2], np.zeros_like(u), d[:, 0]]
        return raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), EPS)
    if name == "saddle_surface":
        dv = np.c_[np.zeros_like(u), np.ones_like(u), -2 * SADDLE_A * v]
        du = dphi_du(name, u, v)
        raw = np.cross(du, dv)
        return raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), EPS)
    raise KeyError(name)


DOMAINS = {
    "flat_plane": {"u": (-1.0, 1.0), "v": (-1.0, 1.0)},
    "sphere_patch": {"u": (0.0, 2 * np.pi), "v": (np.pi / 3, 2 * np.pi / 3)},
    "swiss_roll": {"u": (1.5 * np.pi, SWISS_ROLL_TMAX), "v": (-1.0, 1.0)},
    "saddle_surface": {"u": (-1.0, 1.0), "v": (-1.0, 1.0)},
}
MANIFOLDS = list(DOMAINS.keys())


def sample_uv(name: str, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    (u_lo, u_hi), (v_lo, v_hi) = DOMAINS[name]["u"], DOMAINS[name]["v"]
    return rng.uniform(u_lo, u_hi, n), rng.uniform(v_lo, v_hi, n)


_MANIFOLD_SEED_TAG = {"flat_plane": 1, "sphere_patch": 2, "swiss_roll": 3, "saddle_surface": 4}


def _speed_scale(name: str) -> float:
    # NOTE: must not use Python's built-in hash() for the per-manifold seed
    # tag -- str hashing is randomized per-process (PYTHONHASHSEED) unless
    # explicitly disabled, which would make SPEED_SCALE (and therefore every
    # downstream noisy draw) silently irreproducible across runs.
    rng = np.random.default_rng(np.random.SeedSequence([REFERENCE_SEED, _MANIFOLD_SEED_TAG[name]]))
    u, v = sample_uv(name, rng, 5000)
    raw = dphi_du(name, u, v)
    speed = np.linalg.norm(raw, axis=1)
    return float(np.median(speed[speed > EPS]))


SPEED_SCALE = {name: _speed_scale(name) for name in MANIFOLDS}


def manifold_data(name: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    u, v = sample_uv(name, rng, N)
    X3 = phi(name, u, v)
    raw_V3 = dphi_du(name, u, v)
    V3_clean = raw_V3 / max(SPEED_SCALE[name], EPS)
    N3 = unit_normal(name, u, v, X3)

    Y = X3 + rng.normal(scale=SIGMA_X, size=(N, 1)) * N3
    W = V3_clean + rng.normal(scale=SIGMA_V, size=V3_clean.shape)
    return {"Y": Y, "field": W, "P": X3, "truth": V3_clean, "labels": np.zeros(N, int), "d": INTRINSIC_DIMENSION, "manifold": name}


def _dense_mesh(name: str) -> tuple[np.ndarray, np.ndarray]:
    (u_lo, u_hi), (v_lo, v_hi) = DOMAINS[name]["u"], DOMAINS[name]["v"]
    uu, vv = np.meshgrid(np.linspace(u_lo, u_hi, MESH_GRID_SIDE), np.linspace(v_lo, v_hi, MESH_GRID_SIDE), indexing="ij")
    uu, vv = uu.ravel(), vv.ravel()
    X3 = phi(name, uu, vv)
    V3 = dphi_du(name, uu, vv) / max(SPEED_SCALE[name], EPS)
    return X3, V3


_MESH_CACHE: dict[str, tuple[np.ndarray, np.ndarray, NearestNeighbors]] = {}


def dense_mesh_index(name: str) -> tuple[np.ndarray, np.ndarray, NearestNeighbors]:
    if name not in _MESH_CACHE:
        X3, V3 = _dense_mesh(name)
        _MESH_CACHE[name] = (X3, V3, NearestNeighbors(n_neighbors=1).fit(X3))
    return _MESH_CACHE[name]


def evaluation_targets(data: dict, Xhat: np.ndarray) -> dict:
    name = data["manifold"]
    if name == "flat_plane":
        position = np.asarray(Xhat, float).copy()
        position[:, 2] = 0.0
        (u_lo, u_hi), (v_lo, v_hi) = DOMAINS[name]["u"], DOMAINS[name]["v"]
        position[:, 0] = np.clip(position[:, 0], u_lo, u_hi)
        position[:, 1] = np.clip(position[:, 1], v_lo, v_hi)
        velocity = np.tile([1.0, 0.0, 0.0], (len(Xhat), 1)) / max(SPEED_SCALE[name], EPS)
    elif name == "sphere_patch":
        radius = np.linalg.norm(Xhat, axis=1)
        position = Xhat / np.maximum(radius[:, None], EPS)
        velocity = np.c_[-position[:, 1], position[:, 0], np.zeros(len(Xhat))] / max(SPEED_SCALE[name], EPS)
    else:
        mesh_X, mesh_V, index = dense_mesh_index(name)
        nn = index.kneighbors(Xhat, return_distance=False)[:, 0]
        position, velocity = mesh_X[nn], mesh_V[nn]
    distance = float(np.sqrt(np.mean(np.sum((Xhat - position) ** 2, axis=1))))
    return {"position": position, "velocity": velocity, "distance_to_manifold": distance}


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


def curvature_aware_k_for_manifold(name: str) -> tuple[int, dict]:
    """Same two-stage k(n,d) procedure used everywhere else in this pipeline
    (Stage-1 ceiling, then curvature-aware refinement on TUNING_SEEDS draws)
    -- NOT optional here. An earlier version of this script used the raw
    Stage-1 ceiling alone (37 for n=480,d=2), which on curved manifolds
    (sphere_patch, swiss_roll -- exactly the geometries this refinement was
    built for) reproduces the known Euclidean-kNN-bridges-across-curvature
    failure mode documented in log.md Round 2/3: every geometry-fitting
    method looked catastrophically worse than noisy input. Caught by
    comparing against the canonical protocol's own frozen k for swiss_roll
    (16) and half_sphere_tangent (21), both far below 37."""
    k_grid = curvature_probe_k_grid(N, INTRINSIC_DIMENSION)
    curves = [local_pca_normal_residual(manifold_data(name, seed)["Y"], INTRINSIC_DIMENSION, k_grid) for seed in TUNING_SEEDS]
    return curvature_aware_neighbor_count(k_grid, curves)


def run(config_source_scenario: str = "circle") -> tuple[pd.DataFrame, dict]:
    frozen = load_frozen_config()
    vmf_config = frozen["velocity_manifold_fitter"][config_source_scenario]
    position_config = {"position_only_manfit": frozen["position_only_manfit"][config_source_scenario]}

    manifold_k = {}
    for name in MANIFOLDS:
        manifold_k[name], _ = curvature_aware_k_for_manifold(name)

    rows = []
    for name in MANIFOLDS:
        k = manifold_k[name]
        for seed in FINAL_SEEDS:
            data = manifold_data(name, seed)
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
                        "manifold": name, "seed": seed, "method": method, "method_label": LABELS[method],
                        "k": k, "nan_inf_count": nan_inf_count, **absolute, **relative,
                    }
                )
    frame = pd.DataFrame(rows)
    provenance = {
        "n": N, "d": INTRINSIC_DIMENSION, "sigma_x": SIGMA_X, "sigma_v": SIGMA_V,
        "shared_config_source_scenario": config_source_scenario, "vmf_config": vmf_config,
        "position_only_config": position_config["position_only_manfit"], "final_seeds": list(FINAL_SEEDS),
        "manifolds": MANIFOLDS, "domains": DOMAINS, "speed_scale": SPEED_SCALE, "reference_seed": REFERENCE_SEED,
        "manifold_k": manifold_k, "k_rule": "curvature_aware_neighbor_count on TUNING_SEEDS (same two-stage rule as the canonical protocol)",
        "deferred": ["rotational latent dynamics variant (r_dot=0, theta_dot=1) -- explicitly compute-budget-optional in current_plan.md"],
    }
    return frame, provenance


HEADLINE_METRICS = ("clean_point_rmse_rel", "distance_to_manifold_rel", "velocity_rmse_loc_rel", "joint_euler_state_rmse_rel")


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in frame.groupby(["manifold", "method", "method_label"], sort=False):
        row = dict(zip(["manifold", "method", "method_label"], keys))
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
    fig, axes = plt.subplots(1, len(MANIFOLDS), figsize=(4.4 * len(MANIFOLDS), 4), constrained_layout=True)
    for ax, name in zip(axes, MANIFOLDS):
        sub = summary[summary.manifold == name]
        x = np.arange(len(methods))
        heights = [float(sub[sub.method == m]["clean_point_rmse_rel_median"].iloc[0]) for m in methods]
        ax.bar(x, heights, color=[_METHOD_COLORS[m] for m in methods])
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in methods], fontsize=7, rotation=25, ha="right")
        ax.set_title(name, fontsize=10)
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
    primary_fig = plot_methods(summary, output, PRIMARY_METHOD_ORDER, "v2_primary_clean_point_rmse_by_manifold.png")
    ablation_fig = plot_methods(summary, output, ABLATION_METHOD_ORDER, "v2_ablation_clean_point_rmse_by_manifold.png")
    metric_cols = ["clean_point_rmse_rel_median", "distance_to_manifold_rel_median", "velocity_rmse_loc_rel_median", "joint_euler_state_rmse_rel_median"]
    primary_table = summary[summary.method.isin(PRIMARY_METHOD_ORDER)][["manifold", "method_label", *metric_cols]].to_html(
        index=False, border=0, float_format=lambda x: f"{x:.4g}"
    )
    ablation_table = summary[summary.method.isin(ABLATION_METHOD_ORDER)][["manifold", "method_label", *metric_cols]].to_html(
        index=False, border=0, float_format=lambda x: f"{x:.4g}"
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>V2 manifold family</title><style>{style}</style></head><body><main>"
        "<h1>P3 Experiment V2: same intrinsic dynamics, different manifolds</h1>"
        f"<section class='card'><p>Shared latent dynamics u_dot=1, v_dot=0, pushed forward through "
        f"four embeddings (flat_plane, sphere_patch, swiss_roll, saddle_surface) via V=D(phi)(u,v)@(1,0). "
        f"n={provenance['n']}, sigma_X={provenance['sigma_x']}, sigma_V={provenance['sigma_v']} for every "
        f"manifold. Shared VMF/Position-only hyperparameters reused verbatim from the canonical protocol "
        f"(sourced from '{provenance['shared_config_source_scenario']}'); k(n,d) recomputed fresh per "
        "manifold via the full two-stage curvature-aware rule (same as the canonical protocol; "
        f"per-manifold k: {provenance['manifold_k']}). Velocity uses the raw (non-unit-normalized) "
        "Jacobian pushforward, globally rescaled per manifold to median speed 1 (fixed reference seed, "
        "independent of the final seeds) -- deliberately NOT per-point unit-normalized like the "
        "canonical swiss_roll/saddle_surface scenarios, since that would erase the very "
        "embedding-induced speed variation this experiment exists to observe. All 7 methods (M0-M6) "
        "are computed for every manifold; shown here in the same two views as the canonical benchmark "
        "report -- primary (external baselines) and ablation (M4-to-M5-to-M6 pipeline capability) -- "
        "rather than only the ablation trio.</p></section>"
        "<section class='card'><h2>Primary comparison (M0, M1 GraphVelo, M2 Cosine Kernel, "
        f"M3 Joint Low-Rank, M6 ManfitVelo)</h2><img src='{image_uri(primary_fig)}'>{primary_table}</section>"
        "<section class='card'><h2>Ablation (M4 Local PCA, M5 Position-only MANFIT, M6 ManfitVelo)</h2>"
        f"<img src='{image_uri(ablation_fig)}'>{ablation_table}</section>"
        "</main></body></html>"
    )
    path = output / "v2_report.html"
    path.write_text(html, encoding="utf-8")
    parser = AuditParser()
    parser.feed(html)
    return {
        "self_contained_html": len(parser.images) == 2 and all(u.startswith("data:image/png;base64,") for u in parser.images) and not parser.external,
        "embedded_figure_count": len(parser.images),
        "expected_figure_count": 2,
    }


def main() -> None:
    output = ROOT / "results" / "v2_manifold_family"
    output.mkdir(parents=True, exist_ok=True)
    frame, provenance = run()
    summary = summarize(frame)
    frame.to_csv(output / "seed_metrics.csv", index=False)
    summary.to_csv(output / "summary_metrics.csv", index=False)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    audit = build_report(output, summary, provenance)
    expected_rows = len(MANIFOLDS) * len(FINAL_SEEDS) * len(METHODS)
    pass_fail_checks = {
        "expected_seed_rows": bool(len(frame) == expected_rows),
        "no_duplicate_rows": bool(not frame.duplicated(["manifold", "seed", "method"]).any()),
        **audit,
    }
    checks = {**pass_fail_checks, "final_seeds_used_for_selection": False}
    checks["all_checks_pass"] = bool(all(pass_fail_checks.values()))
    (output / "sanity_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(output / "v2_report.html")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
