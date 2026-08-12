"""Ambient-dimension scalability for Circle (d=1) and Saddle Surface (d=2),
embedded in R^D via a deterministic orthonormal D x 3 matrix.

current_plan.md P1.2, scoped down per the user's 2026-08-12 split of P1.2 into two
separate goals: (1) does the isotropic-Gaussian-noise + ambient-D mechanism
work at all -- already answered by run_sphere_scalability.py (S^2, positive
curvature, d=2); (2) does ambient dimension itself hurt manifold recovery,
independent of noise mode -- not yet answered for any d=1 case, and not
answered for a negative/mixed-curvature d=2 case either. This module answers
(2) only, reusing the EXISTING normal_only position-noise convention already
used by the 9 canonical scenarios (a single scalar draw along the manifold's
own analytic normal direction, embedded into R^D via the same orthonormal
matrix as the clean geometry -- not isotropic Gaussian, which is out of scope
here per the split above). Velocity noise stays isotropic across all D
ambient coordinates, same "fixed_coordinate" per-coordinate-variance
convention as run_sphere_scalability.py (so total velocity noise grows as
sqrt(D), same as there), since velocity noise in this codebase has always
been ambient-isotropic (see scripts/benchmark_scenarios.py::
add_noise) -- noise_mode only ever described *position* noise.

Circle and Saddle Surface are literally two of the 9 canonical scenario
names, so their frozen (T, eta_g, theta, kappa, theta_schedule, lambda_v, k)
are reused directly from results/manfitvelo_benchmark/selected_hyperparameters
.json -- no separate tuning stage, same as every other supplement scan.

Scope, stated explicitly rather than silently: this module reports
clean_point_rmse, distance_to_manifold, velocity_rmse_id,
velocity_angle_mae_id, short_step_euler_rmse (all identity-anchored -- no
location-projection machinery, unlike the canonical benchmark's
velocity_rmse_loc), and tangent_projector_error (mechanism-level, using the
analytic normal/projector embedded through the same orthonormal matrix as
the geometry). It does NOT reproduce run_sphere_scalability.py's
knn_recall / local_covariance_eigengap / median_knn_radius geometry
diagnostics or angle-valid-fraction speed-threshold masking -- secondary
diagnostics not needed to answer "does ambient D hurt, does velocity help",
and skipped to keep this module's scope tractable.

    python simulation/run_manifold_dimension_scalability.py
    python simulation/run_manifold_dimension_scalability.py --report-only
"""

from __future__ import annotations

import argparse
import base64
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.graphvelo_official_adapter import (  # noqa: E402
    GRAPHVELO_CONFIG,
    GRAPHVELO_PROVENANCE,
    GRAPHVELO_STANDARDIZATION,
    graphvelo_velocity_standardized,
)
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
    array_hash,
    dense_truth_support,
    joint_error,
    load_frozen_config,
    observed_tau,
)

FINAL_SEEDS = tuple(range(43000, 43015))
DIMENSIONS = (3, 5, 10, 20, 50)
METHODS = ("ambient_noisy", "cosine_kernel", "graphvelo", "joint_low_rank", "local_pca", "position_only_manfit", "manfitvelo")
LABELS = {
    "ambient_noisy": "Ambient noisy input", "cosine_kernel": "Cosine kernel", "graphvelo": "GraphVelo",
    "joint_low_rank": "Joint Low-Rank (M3)", "local_pca": "Local PCA",
    "position_only_manfit": "Position-only MANFIT", "manfitvelo": "ManfitVelo",
}
EPS = 1e-12


def circle_generate_3d(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same analytic construction as scripts/benchmark_scenarios.py::vector_data("circle")."""
    t = rng.uniform(0, 2 * np.pi, n)
    X3 = np.c_[np.cos(t), np.sin(t), np.zeros(n)]
    F3 = np.c_[-np.sin(t), np.cos(t), np.zeros(n)]
    N3 = X3.copy()
    return X3, F3, N3


def circle_distance(coords3: np.ndarray) -> np.ndarray:
    radius = np.linalg.norm(coords3[:, :2], axis=1)
    return np.sqrt((radius - 1.0) ** 2 + coords3[:, 2] ** 2)


def saddle_generate_3d(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same analytic construction as vector_data("saddle_surface"); a=0.45 matches the canonical scenario."""
    a = 0.45
    u = rng.uniform(-1, 1, n)
    v = rng.uniform(-1, 1, n)
    X3 = np.c_[u, v, a * (u * u - v * v)]
    Ju = np.c_[np.ones(n), np.zeros(n), 2 * a * u]
    F3 = Ju / np.maximum(np.linalg.norm(Ju, axis=1, keepdims=True), EPS)
    Jv = np.c_[np.zeros(n), np.ones(n), -2 * a * v]
    N3 = np.cross(Ju, Jv)
    N3 = N3 / np.maximum(np.linalg.norm(N3, axis=1, keepdims=True), EPS)
    return X3, F3, N3


_SADDLE_SUPPORT_INDEX: NearestNeighbors | None = None


def saddle_distance(coords3: np.ndarray) -> np.ndarray:
    global _SADDLE_SUPPORT_INDEX
    if _SADDLE_SUPPORT_INDEX is None:
        support, _, _ = dense_truth_support("saddle_surface")
        _SADDLE_SUPPORT_INDEX = NearestNeighbors(n_neighbors=1).fit(support)
    return _SADDLE_SUPPORT_INDEX.kneighbors(coords3, return_distance=True)[0][:, 0]


def _tangent_projector_1d(X3: np.ndarray, F3: np.ndarray, N3: np.ndarray) -> np.ndarray:
    """Rank-1 tangent projector for a curve (codimension 2 in R^3 -- a single
    normal direction does not span the full orthogonal complement, unlike the
    codimension-1 surface case below). Matches the generic d=1 fallback in
    run_manfitvelo_benchmark.py::true_projector (tangent outer tangent)."""
    return np.einsum("ni,nj->nij", F3, F3)


def _tangent_projector_2d_codim1(X3: np.ndarray, F3: np.ndarray, N3: np.ndarray) -> np.ndarray:
    """Rank-2 tangent projector for a codimension-1 surface in R^3: the full
    orthogonal complement of the single analytic normal. Matches
    run_manfitvelo_benchmark.py::true_projector's swiss_roll/saddle_surface
    branch."""
    return np.eye(3)[None] - np.einsum("ni,nj->nij", N3, N3)


MANIFOLDS = {
    "circle": {
        "d": 1, "n": 360, "canonical_sigma_x": 0.05, "canonical_sigma_v": 0.10,
        "generate": circle_generate_3d, "distance": circle_distance,
        "tangent_projector_3d": _tangent_projector_1d,
    },
    "saddle_surface": {
        "d": 2, "n": 480, "canonical_sigma_x": 0.05, "canonical_sigma_v": 0.10,
        "generate": saddle_generate_3d, "distance": saddle_distance,
        "tangent_projector_3d": _tangent_projector_2d_codim1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/manifold_dimension_scalability")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def orthogonal_embedding(seed: int, ambient_dimension: int) -> np.ndarray:
    """Deterministic orthonormal D x 3 embedding matrix -- same construction as run_sphere_scalability.py."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, ambient_dimension, 9137]))
    raw = rng.normal(size=(ambient_dimension, 3))
    q, r = np.linalg.qr(raw, mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


def manifold_data(manifold: str, seed: int, ambient_dimension: int) -> dict:
    cfg = MANIFOLDS[manifold]
    n = cfg["n"]
    latent_rng = np.random.default_rng(np.random.SeedSequence([seed, 771]))
    X3, F3, N3 = cfg["generate"](latent_rng, n)
    Q = orthogonal_embedding(seed, ambient_dimension)
    X = X3 @ Q.T
    V_clean = F3 @ Q.T
    N = N3 @ Q.T

    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, ambient_dimension, 331]))
    # noise_mode = normal_only: a single scalar draw per point along the
    # manifold's own analytic normal, embedded via the same Q as the clean
    # geometry -- magnitude is D-independent by construction (a scalar times
    # a unit vector), unlike isotropic Gaussian noise (out of scope here,
    # already covered by run_sphere_scalability.py's isotropic_gaussian mode).
    eps_x = noise_rng.normal(scale=cfg["canonical_sigma_x"], size=(n, 1))
    Y = X + eps_x * N

    # Velocity noise: isotropic across all D ambient coordinates, fixed
    # per-coordinate std (matches run_sphere_scalability.py's
    # "fixed_coordinate" regime -- total velocity noise magnitude grows as
    # sqrt(D)). Velocity noise has always been ambient-isotropic in this
    # codebase (see add_noise in scripts/benchmark_scenarios
    # .py); noise_mode only ever described *position* noise.
    zv = noise_rng.normal(size=(n, ambient_dimension))
    W = V_clean + cfg["canonical_sigma_v"] * zv

    P3 = cfg["tangent_projector_3d"](X3, F3, N3)
    true_projector = np.einsum("da,nab,eb->nde", Q, P3, Q)
    return {
        "Y": Y, "field": W, "P": X, "truth": V_clean, "Q": Q,
        "true_projector": true_projector, "d": cfg["d"], "n": n,
        "labels": np.zeros(n, int),
    }


def diagnostic_projectors(X: np.ndarray, d: int, k: int) -> np.ndarray:
    from scripts.pca_denoisers import local_pca_denoise

    _, info = local_pca_denoise(X, d, n_neighbors=k, return_info=True)
    return info["projectors"]


def fit_method(method: str, manifold: str, data: dict, config: dict, k: int, seed: int):
    X, V, d = data["Y"], data["field"], data["d"]
    if method == "ambient_noisy":
        Xhat, Vhat, info = X.copy(), V.copy(), {}
    elif method == "cosine_kernel":
        direction, info = cosine_kernel_projection(X, V, shared_knn_graph(X, k))
        Vhat, speed = restore_noisy_speed(direction, V)
        info.update(speed)
        Xhat = X.copy()
    elif method == "graphvelo":
        Xhat = X.copy()
        Vhat, info = graphvelo_velocity_standardized(X, V)
    elif method == "joint_low_rank":
        Xhat, Vhat, info = joint_low_rank_state(X, V)
    elif method == "local_pca":
        Xhat, Vhat, info = local_pca_state(X, V, d, k)
    elif method == "position_only_manfit":
        cfg = config["position_only_manfit"][manifold]
        Xhat = position_only_trajectory(X, V, d, cfg["k"], cfg["T"], cfg["eta_g"])[-1][1]
        Vhat, info = downstream_velocity(Xhat, V, d, k)
    elif method == "manfitvelo":
        result = fit_vmf_variant(X, V, d, config["velocity_manifold_fitter"][manifold], seed)
        Xhat, Vhat, info = result["X"], result["V"], {"projectors": result["P"]}
    else:
        raise KeyError(method)
    return np.asarray(Xhat), np.asarray(Vhat), info


def evaluate(manifold: str, Xhat: np.ndarray, Vhat: np.ndarray, info: dict, data: dict, tau: float, k: int) -> dict:
    cfg = MANIFOLDS[manifold]
    coords3 = Xhat @ data["Q"]
    distance = float(np.sqrt(np.mean(cfg["distance"](coords3) ** 2)))
    estimate_speed = np.linalg.norm(Vhat, axis=1)
    clean_speed = np.linalg.norm(data["truth"], axis=1)
    valid = (estimate_speed > 1e-8) & (clean_speed > 1e-8)
    cosine = np.sum(Vhat[valid] * data["truth"][valid], axis=1) / (estimate_speed[valid] * clean_speed[valid])
    if "projectors" in info:
        estimated_projectors = info["projectors"]
    else:
        estimated_projectors = diagnostic_projectors(Xhat, data["d"], k)
    projector_error = float(np.sqrt(np.mean(np.sum((estimated_projectors - data["true_projector"]) ** 2, axis=(1, 2)))))
    return {
        "clean_point_rmse": float(np.sqrt(np.mean(np.sum((Xhat - data["P"]) ** 2, axis=1)))),
        "distance_to_manifold": distance,
        "velocity_rmse_id": float(np.sqrt(np.mean(np.sum((Vhat - data["truth"]) ** 2, axis=1)))),
        "velocity_angle_mae_id": float(np.degrees(np.mean(np.arccos(np.clip(cosine, -1, 1))))) if np.any(valid) else float("nan"),
        "angle_valid_fraction": float(np.mean(valid)),
        "short_step_euler_rmse": joint_error(Xhat, Vhat, data["P"], data["truth"], tau),
        "tangent_projector_error": projector_error,
    }


def warm_up(config: dict) -> None:
    for manifold in MANIFOLDS:
        data = manifold_data(manifold, 41000, 3)
        k = int(config["shared_graph_k"][manifold])
        for method in METHODS:
            fit_method(method, manifold, data, config, k, 41000)


def run(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    warm_up(config)
    rows, runtime = [], []
    for manifold in MANIFOLDS:
        k = int(config["shared_graph_k"][manifold])
        for D in DIMENSIONS:
            for seed in FINAL_SEEDS:
                data = manifold_data(manifold, seed, D)
                graph = shared_knn_graph(data["Y"], k)
                tau, _, _ = observed_tau(data["Y"], data["field"], graph)
                sample_hash = array_hash(data["Y"], data["field"], data["P"], data["truth"])
                for method in METHODS:
                    start = time.perf_counter()
                    Xhat, Vhat, info = fit_method(method, manifold, data, config, k, seed)
                    elapsed = time.perf_counter() - start
                    nan_inf_count = int(np.sum(~np.isfinite(Xhat)) + np.sum(~np.isfinite(Vhat)))
                    if nan_inf_count:
                        Xhat = np.nan_to_num(Xhat, nan=0.0, posinf=0.0, neginf=0.0)
                        Vhat = np.nan_to_num(Vhat, nan=0.0, posinf=0.0, neginf=0.0)
                    rows.append(
                        {
                            "manifold": manifold, "ambient_dimension": D, "seed": seed, "method": method,
                            "method_label": LABELS[method], "n": data["n"], "intrinsic_dimension": data["d"],
                            "k": k, "tau": tau, "sample_hash": sample_hash, "nan_inf_count": nan_inf_count,
                            **evaluate(manifold, Xhat, Vhat, info, data, tau, k),
                        }
                    )
                    runtime.append(
                        {"manifold": manifold, "ambient_dimension": D, "seed": seed, "method": method, "runtime_seconds": elapsed}
                    )
    return pd.DataFrame(rows), pd.DataFrame(runtime)


METRICS = (
    "clean_point_rmse", "distance_to_manifold", "velocity_rmse_id", "velocity_angle_mae_id",
    "short_step_euler_rmse", "tangent_projector_error", "angle_valid_fraction",
)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in frame.groupby(["manifold", "ambient_dimension", "method", "method_label"], sort=False):
        row = dict(zip(["manifold", "ambient_dimension", "method", "method_label"], keys))
        for metric in METRICS:
            row[f"{metric}_median"] = float(g[metric].median())
            row[f"{metric}_q25"] = float(g[metric].quantile(0.25))
            row[f"{metric}_q75"] = float(g[metric].quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_group(summary: pd.DataFrame, manifold: str, output: Path, name: str, metrics: tuple[str, ...]) -> Path:
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    sub = summary[summary.manifold == manifold]
    for ax, metric in zip(axes, metrics):
        for method, g in sub.groupby("method", sort=False):
            g = g.sort_values("ambient_dimension")
            ax.plot(g.ambient_dimension, g[f"{metric}_median"], "-", marker="o", label=LABELS[method])
        ax.set(xlabel="Ambient dimension D", ylabel=metric)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=6, ncol=2)
    path = output / f"{manifold}_{name}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


class AuditParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []
        self.external = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "img":
            self.images.append(values.get("src", ""))
        if tag in {"link", "script"} and (values.get("href") or values.get("src")):
            self.external.append(values.get("href") or values.get("src"))


def build_report(output: Path, summary: pd.DataFrame) -> dict:
    style = (
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f4f6f8;"
        "color:#17212b;margin:0}main{max-width:1350px;margin:auto;padding:28px}.card{background:white;"
        "border:1px solid #d8dee7;border-radius:9px;padding:18px;margin:16px 0;overflow:auto}"
        "img{max-width:100%}table{border-collapse:collapse;width:100%;font-size:11px}"
        "th,td{border:1px solid #d8dee7;padding:5px}th{background:#edf1f5}p{line-height:1.5}"
    )
    sections = []
    for manifold in MANIFOLDS:
        error_path = plot_group(summary, manifold, output, "errors_vs_dimension", ("clean_point_rmse", "velocity_rmse_id", "short_step_euler_rmse"))
        diag_path = plot_group(summary, manifold, output, "diagnostics_vs_dimension", ("distance_to_manifold", "tangent_projector_error"))
        table = summary[summary.manifold == manifold][
            ["ambient_dimension", "method_label", "clean_point_rmse_median", "velocity_rmse_id_median", "short_step_euler_rmse_median", "tangent_projector_error_median"]
        ].to_html(index=False, border=0, float_format=lambda x: f"{x:.4g}")
        sections.append(
            f"<section class='card'><h2>{manifold}</h2><img src='{image_uri(error_path)}'>"
            f"<img src='{image_uri(diag_path)}'>{table}</section>"
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Circle/Saddle ambient-D scalability</title><style>{style}</style></head><body><main>"
        "<h1>Circle (d=1) / Saddle Surface (d=2) &rarr; R^D scalability, normal_only noise</h1>"
        "<section class='card'><p>current_plan.md P1.2 (scoped 2026-08-12): this module isolates "
        "ambient-dimension sensitivity from noise mode -- position noise stays normal_only "
        "(D-independent magnitude, embedded along the analytic normal), same convention as "
        "the 9 canonical scenarios; isotropic Gaussian position noise + ambient-D is covered "
        "separately by run_sphere_scalability.py. Velocity noise is ambient-isotropic with a "
        "fixed per-coordinate std (total velocity noise grows as sqrt(D)), same convention as "
        "run_sphere_scalability.py. Frozen (T, eta_g, theta, kappa, theta_schedule, lambda_v, k) "
        "reused directly from the canonical circle/saddle_surface scenarios -- no separate tuning."
        "</p></section>" + "".join(sections) + "</main></body></html>"
    )
    path = output / "scalability_report.html"
    path.write_text(html, encoding="utf-8")
    parser = AuditParser()
    parser.feed(html)
    return {
        "self_contained_html": len(parser.images) == 2 * len(MANIFOLDS) and all(v.startswith("data:image/png;base64,") for v in parser.images) and not parser.external,
        "embedded_figure_count": len(parser.images),
        "expected_figure_count": 2 * len(MANIFOLDS),
    }


def environment() -> dict:
    return {
        "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
        "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "graphvelo": GRAPHVELO_PROVENANCE, "graphvelo_config": GRAPHVELO_CONFIG, "graphvelo_standardization": GRAPHVELO_STANDARDIZATION,
    }


def validate(frame: pd.DataFrame, audit: dict) -> dict:
    expected = len(MANIFOLDS) * len(DIMENSIONS) * len(FINAL_SEEDS) * len(METHODS)
    checks = {
        "expected_seed_rows": bool(len(frame) == expected),
        "no_duplicate_rows": bool(not frame.duplicated(["manifold", "ambient_dimension", "seed", "method"]).any()),
        "noise_mode": "normal_only",
        **audit,
    }
    checks["all_checks_pass"] = bool(all(v for k, v in checks.items() if isinstance(v, bool)))
    if not checks["all_checks_pass"]:
        raise AssertionError(checks)
    return checks


def main() -> None:
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        frame = pd.read_csv(output / "seed_metrics.csv")
        summary = pd.read_csv(output / "summary_metrics.csv")
    else:
        config = load_frozen_config()
        frame, runtime = run(config)
        summary = summarize(frame)
        frame.to_csv(output / "seed_metrics.csv", index=False)
        summary.to_csv(output / "summary_metrics.csv", index=False)
        runtime.to_csv(output / "runtime.csv", index=False)
        (output / "config.json").write_text(
            json.dumps(
                {
                    "dimensions": DIMENSIONS, "manifolds": {name: {k: v for k, v in cfg.items() if k not in ("generate", "distance", "tangent_projector_3d")} for name, cfg in MANIFOLDS.items()},
                    "noise_mode": "normal_only", "velocity_noise_regime": "fixed_coordinate (sqrt(D) total growth)",
                    "seeds": FINAL_SEEDS, "algorithms": config,
                },
                indent=2,
            )
            + "\n"
        )
        (output / "environment_provenance.json").write_text(json.dumps(environment(), indent=2) + "\n")
    audit = build_report(output, summary)
    checks = validate(frame, audit)
    (output / "sanity_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(output / "scalability_report.html")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
