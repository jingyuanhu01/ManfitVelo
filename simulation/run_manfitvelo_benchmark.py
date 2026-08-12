"""Main reach-audited ManfitVelo benchmark with GraphVelo diagnostics.

Geometry selection and all model selection use only seeds 42000--42002.
Final seeds 43000--43014 are evaluated once after configurations are frozen.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

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

from scripts.simulation_baselines import (  # noqa: E402
    JOINT_LOW_RANK_VARIANCE_THRESHOLD,
    cosine_kernel_projection,
    joint_low_rank_state,
    shared_knn_graph,
    restore_noisy_speed,
)
from scripts.graphvelo_official_adapter import (  # noqa: E402
    GRAPHVELO_CONFIG,
    GRAPHVELO_PROVENANCE,
    GRAPHVELO_STANDARDIZATION,
    graphvelo_velocity,
    graphvelo_velocity_standardized,
    official_neighbors,
)
from scripts.pca_denoisers import local_pca_denoise  # noqa: E402
from scripts.benchmark_scenarios import (  # noqa: E402
    HAIRPIN_DEFAULT_SEPARATION,
    HAIRPIN_LEGACY_SEPARATION,
    SETS,
    Set,
    add_noise,
    fit_vmf_variant,
    hairpin,
    position_only_trajectory,
    vector_data,
)
from simulation.benchmark_core import (  # noqa: E402
    NEIGHBOR_COUNT_CLIP,
    NEIGHBOR_SCALING_CONSTANT,
    PILOT_SEED,
    SCENARIO_LABELS,
    angle_mae,
    array_hash,
    curvature_aware_neighbor_count,
    curvature_probe_k_grid,
    evaluation_targets,
    joint_error,
    local_pca_normal_residual,
    neighbor_count,
    observed_tau,
    vector_rmse,
)


EPS = 1e-12
TUNING_SEEDS = (42000, 42001, 42002)
FINAL_SEEDS = tuple(range(43000, 43015))
SCENARIOS = (
    "circle",
    "s_curve",
    "curved_hairpin",
    "flat_rotation_annulus",
    "half_sphere_tangent",
    "y_branch",
    "near_intersection",
    "swiss_roll",
    "saddle_surface",
)
BASE_METHODS = (
    "ambient_noisy",
    "cosine_kernel",
    "graphvelo",
    "joint_low_rank",
    "local_pca",
    "position_only_manfit",
)
BASE_LABELS = {
    "truth": "Clean truth",
    "ambient_noisy": "Ambient noisy input",
    "cosine_kernel": "Cosine kernel",
    "graphvelo": "GraphVelo (standardized)",
    "joint_low_rank": "Joint Low-Rank (M3)",
    "local_pca": "Local PCA",
    "position_only_manfit": "Position-only MANFIT",
    "manfitvelo": "ManfitVelo",
}
ABSOLUTE_METRICS = (
    "clean_point_rmse",
    "distance_to_manifold",
    "velocity_rmse_id",
    "velocity_angle_mae_id",
    "velocity_rmse_loc",
    "joint_euler_state_rmse",
)
PRIMARY_METRICS = tuple(f"{metric}_rel" for metric in ABSOLUTE_METRICS)
METRIC_LABELS = {
    "clean_point_rmse_rel": "Clean-point RMSE",
    "distance_to_manifold_rel": "Distance to manifold",
    "velocity_rmse_id_rel": "Velocity RMSE (identity)",
    "velocity_angle_mae_id_rel": "Velocity angle MAE (identity)",
    "velocity_rmse_loc_rel": "Velocity RMSE (location)",
    "joint_euler_state_rmse_rel": "Short-step Euler forecast RMSE rel",
}
HAIRPIN_GEOMETRY_GRID = (0.13, 0.18, 0.20, 0.22, 0.24, 0.28)
HAIRPIN_K_GRID = (4, 6, 8, 10, 12, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/manfitvelo_benchmark",
    )
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def cross_branch_mask(
    scenario: str, neighbors: np.ndarray, labels: np.ndarray, clean: np.ndarray
) -> np.ndarray:
    source = labels[:, None]
    target = labels[neighbors]
    if scenario == "curved_hairpin":
        return ((source == 0) & (target == 2)) | ((source == 2) & (target == 0))
    if scenario == "near_intersection":
        return source != target
    if scenario == "y_branch":
        away = np.linalg.norm(clean[:, :2], axis=1) >= 0.05
        return (source != target) & away[:, None] & away[neighbors]
    return np.zeros(neighbors.shape, dtype=bool)


def hairpin_reach_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Select geometry by reach/neighborhood rules, without fitting a method."""
    summary_rows, point_rows = [], []
    arm_length, curvature = 1.2, 0.15
    arm_curvature_radius = 1.0 / (curvature * (np.pi / (2 * arm_length)) ** 2)
    for separation in HAIRPIN_GEOMETRY_GRID:
        bend_radius = separation / 2.0
        rho_curv = min(bend_radius, arm_curvature_radius)
        d_nonlocal = separation
        rho_sep = d_nonlocal / 2.0
        reach = min(rho_curv, rho_sep)
        for k in HAIRPIN_K_GRID:
            seed_rows = []
            for seed in TUNING_SEEDS:
                clean, truth, normal, _, labels = hairpin(480, separation, seed)
                noisy, _ = add_noise(
                    clean,
                    truth,
                    normal,
                    Set(480, SETS["curved_hairpin"].px, SETS["curved_hairpin"].field_noise, 1),
                    np.random.default_rng(seed),
                )
                distance, raw = NearestNeighbors(n_neighbors=k + 1).fit(noisy).kneighbors(noisy)
                neighbors = raw[:, 1:]
                radii = distance[:, -1]
                edge_mask = cross_branch_mask("curved_hairpin", neighbors, labels, clean)
                per_point = edge_mask.mean(axis=1)
                seed_rows.append(
                    {
                        "median_knn_radius_over_reach": float(np.median(radii) / reach),
                        "q90_knn_radius_over_reach": float(np.quantile(radii, 0.9) / reach),
                        "cross_arm_edge_fraction": float(edge_mask.mean()),
                        "point_cross_arm_fraction_median": float(np.median(per_point)),
                        "point_cross_arm_fraction_q75": float(np.quantile(per_point, 0.75)),
                        "point_cross_arm_fraction_q90": float(np.quantile(per_point, 0.9)),
                        "point_cross_arm_fraction_q95": float(np.quantile(per_point, 0.95)),
                        "point_cross_arm_fraction_max": float(np.max(per_point)),
                        "points_with_any_cross_arm_neighbor": float(np.mean(per_point > 0)),
                    }
                )
                point_rows.extend(
                    {
                        "separation": separation,
                        "bend_radius": bend_radius,
                        "k": k,
                        "seed": seed,
                        "point_index": i,
                        "arm_label": int(labels[i]),
                        "cross_arm_neighbor_fraction": float(value),
                    }
                    for i, value in enumerate(per_point)
                )
            med = pd.DataFrame(seed_rows).median().to_dict()
            row = {
                "separation": separation,
                "bend_radius": bend_radius,
                "arm_length": arm_length,
                "rho_curvature": rho_curv,
                "d_nonlocal": d_nonlocal,
                "rho_separation": rho_sep,
                "reach_proxy": reach,
                "position_noise_sigma": SETS["curved_hairpin"].px,
                "reach_over_sigma": reach / SETS["curved_hairpin"].px,
                "k": k,
                **med,
            }
            row["meets_reach_condition"] = row["reach_over_sigma"] >= 4.0
            row["meets_radius_condition"] = row["q90_knn_radius_over_reach"] < 0.5
            row["meets_cross_arm_condition"] = row["cross_arm_edge_fraction"] < 0.05
            row["meets_all_conditions"] = bool(
                row["meets_reach_condition"]
                and row["meets_radius_condition"]
                and row["meets_cross_arm_condition"]
            )
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    feasible = summary[summary.meets_all_conditions].copy()
    if len(feasible):
        feasible["geometry_change"] = abs(feasible.separation - HAIRPIN_LEGACY_SEPARATION)
        choice = feasible.sort_values(["geometry_change", "k"]).iloc[0]
        selection_basis = "smallest geometry change satisfying all prespecified conditions"
    else:
        summary["violation"] = (
            np.maximum(0, 4 - summary.reach_over_sigma)
            + np.maximum(0, summary.q90_knn_radius_over_reach - 0.5)
            + np.maximum(0, summary.cross_arm_edge_fraction - 0.05)
        )
        choice = summary.sort_values(["violation", "separation", "k"]).iloc[0]
        selection_basis = "closest prespecified diagnostic compromise"
    selected = {
        "separation": float(choice.separation),
        "bend_radius": float(choice.bend_radius),
        "k": int(choice.k),
        "selection_basis": selection_basis,
        "selection_uses_method_results": False,
        "selection_uses_final_seeds": False,
    }
    summary["selected"] = (
        np.isclose(summary.separation, selected["separation"]) & summary.k.eq(selected["k"])
    )
    if not np.isclose(selected["separation"], HAIRPIN_DEFAULT_SEPARATION):
        raise AssertionError("generator default does not match reach-selected hairpin")
    return summary, pd.DataFrame(point_rows), selected


def downstream_velocity(X: np.ndarray, W: np.ndarray, d: int, k: int) -> np.ndarray:
    _, velocity, _ = local_pca_denoise(
        X, d, n_neighbors=k, vectors=W, return_info=True
    )
    return velocity


def state_metrics(
    scenario: str, Xhat: np.ndarray, Vhat: np.ndarray, data: dict, tau: float
) -> dict[str, float]:
    target = evaluation_targets(scenario, Xhat, data)
    all_valid = np.ones(len(Xhat), dtype=bool)
    estimate_norm = np.linalg.norm(Vhat, axis=1)
    target_norm = np.linalg.norm(data["truth"], axis=1)
    angle_valid = (estimate_norm > 1e-8) & (target_norm > 1e-8)
    return {
        "clean_point_rmse": float(np.sqrt(np.mean(np.sum((Xhat - data["P"]) ** 2, axis=1)))),
        "distance_to_manifold": float(target["distance_to_manifold"]),
        "velocity_rmse_id": vector_rmse(Vhat, data["truth"], all_valid),
        "velocity_angle_mae_id": angle_mae(Vhat, data["truth"], all_valid),
        "velocity_rmse_loc": vector_rmse(Vhat, target["location_velocity"], target["location_valid"]),
        "joint_euler_state_rmse": joint_error(Xhat, Vhat, data["P"], data["truth"], tau),
        "velocity_angle_valid_fraction_id": float(np.mean(angle_valid)),
    }


def relative_state_metrics(metrics: dict, baseline: dict) -> dict[str, float]:
    return {f"{key}_rel": float(metrics[key] / baseline[key]) for key in ABSOLUTE_METRICS}


def tuning_score(relative: dict[str, float]) -> float:
    keys = (
        "clean_point_rmse_rel",
        "velocity_rmse_id_rel",
        "velocity_angle_mae_id_rel",
        "joint_euler_state_rmse_rel",
    )
    return float(np.mean(np.log([max(relative[key], EPS) for key in keys])))


def curvature_aware_scenario_k() -> tuple[dict[str, int], pd.DataFrame]:
    """Curvature-aware refinement of neighbor_count(n, d) for every scenario.

    Ground-truth-free: uses only the noisy TUNING_SEEDS draws (development
    seeds), never final seeds. See benchmark_core.curvature_aware_neighbor_count
    for the method; see simulation/log.md for the validation against
    clean_point_rmse_rel. Returns the frozen per-scenario k plus a long-form
    diagnostics frame (k grid / residual curve / log-log slope) for audit.
    """
    scenario_k: dict[str, int] = {}
    rows = []
    for scenario in SCENARIOS:
        setting = SETS[scenario]
        k_grid = curvature_probe_k_grid(setting.n, setting.d)
        curves = [
            local_pca_normal_residual(vector_data(scenario, seed)["Y"], setting.d, k_grid)
            for seed in TUNING_SEEDS
        ]
        safe_k, diag = curvature_aware_neighbor_count(k_grid, curves)
        scenario_k[scenario] = safe_k
        rows.append(
            {
                "scenario": scenario,
                "n": setting.n,
                "d": setting.d,
                "formula_ceiling_k": neighbor_count(setting.n, setting.d),
                "curvature_aware_k": safe_k,
                "k_grid": json.dumps(diag["k_grid"]),
                "residual_curve": json.dumps(diag["residual_curve"]),
                "loglog_slope": json.dumps(diag["loglog_slope"]),
                "turn_index": diag["turn_index"],
            }
        )
    return scenario_k, pd.DataFrame(rows)


def shared_vmf_grid() -> list[dict]:
    """Grid for the single VMF hyperparameter set frozen across every scenario.

    k is deliberately excluded: it is supplied per scenario by the
    neighbor_count(n, d) rule (benchmark_core.py), not searched here.
    velocity_covariance_mode / velocity_trace_normalization were already
    constant across the legacy per-scenario configs, so they stay fixed
    rather than being re-searched. lambda_v=1.0 (updated 2026-08-11, see
    simulation/log.md Round 5 and run_lambda_sensitivity.py): selected on
    TUNING SEEDS ONLY via pooled tuning_score across all 9 scenarios, subject
    to a safeguard that no scenario may score worse than its own lambda_v=0
    baseline (candidate 2.0 had the best naive pooled average but violated
    this safeguard on Swiss Roll, which regresses past lambda_v~0.5-1.0).
    Not re-searched inside this grid to avoid conflating the two selection
    procedures; see run_lambda_sensitivity.py for the dedicated sweep.
    """
    grid = []
    for T in (3, 5, 8):
        for eta_g in (0.35, 0.5, 0.7):
            for kappa in (0.0, 1.0, 2.0):
                for theta in (0.02, 0.05, 0.1):
                    for theta_schedule in (None, "ramp"):
                        config = {
                            "T": T,
                            "eta_g": eta_g,
                            "kappa": kappa,
                            "theta": theta,
                            "bandwidth_mode": "variable",
                            "lambda_v": 1.0,
                            "velocity_covariance_mode": "uncentered",
                            "velocity_trace_normalization": "match_position_trace",
                        }
                        if theta_schedule == "ramp":
                            config["theta_schedule"] = "ramp"
                        grid.append(config)
    return grid


def tune_shared_vmf(scenario_k: dict[str, int]) -> tuple[dict, pd.DataFrame]:
    """Once-for-all VMF search on development seeds, pooled across all scenarios.

    Each candidate is scored by its mean tuning_score over every
    (scenario, tuning seed) pair (7 scenarios x 3 seeds, equal weight); the
    winning config is then frozen for all final scenarios/seeds. This
    replaces the previous scenario-specific (Curved-Hairpin-only) grid.
    """
    rows = []
    for index, base_config in enumerate(shared_vmf_grid()):
        for scenario in SCENARIOS:
            config = dict(base_config, k=scenario_k[scenario])
            for seed in TUNING_SEEDS:
                data = vector_data(scenario, seed)
                noisy_neighbors = shared_knn_graph(data["Y"], config["k"])
                tau, _, _ = observed_tau(data["Y"], data["field"], noisy_neighbors)
                baseline = state_metrics(scenario, data["Y"], data["field"], data, tau)
                result = fit_vmf_variant(data["Y"], data["field"], data["d"], config, seed)
                absolute = state_metrics(scenario, result["X"], result["V"], data, tau)
                relative = relative_state_metrics(absolute, baseline)
                rows.append(
                    {
                        "tuning_stage": "shared_vmf_grid",
                        "candidate_index": index,
                        "scenario": scenario,
                        "seed": seed,
                        "config_json": json.dumps(base_config, sort_keys=True),
                        **relative,
                        "tuning_score": tuning_score(relative),
                    }
                )
    frame = pd.DataFrame(rows)
    choice = frame.groupby(["candidate_index", "config_json"]).tuning_score.mean().idxmin()
    return json.loads(choice[1]), frame


def shared_position_only_grid() -> list[dict]:
    """Grid for the single Position-only MANFIT (M5) (T, eta_g) pair."""
    return [
        {"T": T, "eta_g": eta_g}
        for T in (3, 5, 8)
        for eta_g in (0.35, 0.5, 0.7)
    ]


def tune_shared_position_only(scenario_k: dict[str, int]) -> tuple[dict, pd.DataFrame]:
    """Once-for-all (T, eta_g) search for Position-only MANFIT on dev seeds."""
    rows = []
    for index, base_config in enumerate(shared_position_only_grid()):
        for scenario in SCENARIOS:
            k = scenario_k[scenario]
            for seed in TUNING_SEEDS:
                data = vector_data(scenario, seed)
                noisy_neighbors = shared_knn_graph(data["Y"], k)
                tau, _, _ = observed_tau(data["Y"], data["field"], noisy_neighbors)
                baseline = state_metrics(scenario, data["Y"], data["field"], data, tau)
                trajectory = position_only_trajectory(
                    data["Y"], data["field"], data["d"], k, base_config["T"], base_config["eta_g"]
                )
                _, Xhat, Vhat = trajectory[-1]
                absolute = state_metrics(scenario, Xhat, Vhat, data, tau)
                relative = relative_state_metrics(absolute, baseline)
                rows.append(
                    {
                        "tuning_stage": "shared_position_only_grid",
                        "candidate_index": index,
                        "scenario": scenario,
                        "seed": seed,
                        "config_json": json.dumps(base_config, sort_keys=True),
                        **relative,
                        "tuning_score": tuning_score(relative),
                    }
                )
    frame = pd.DataFrame(rows)
    choice = frame.groupby(["candidate_index", "config_json"]).tuning_score.mean().idxmin()
    return json.loads(choice[1]), frame


def true_projector(scenario: str, data: dict) -> np.ndarray:
    n, ambient = data["P"].shape
    if scenario == "half_sphere_tangent":
        return np.eye(ambient)[None] - np.einsum("ni,nj->nij", data["P"], data["P"])
    if scenario == "flat_rotation_annulus":
        projector = np.diag([1.0, 1.0, 0.0])
        return np.repeat(projector[None], n, axis=0)
    if scenario in ("swiss_roll", "saddle_surface"):
        # Genuinely curved 2D surfaces (unlike the flat annulus): the true
        # tangent plane is the full 2D orthogonal complement of the analytic
        # unit normal (see vector_data's "true_normal"), not the 1D
        # velocity-aligned subspace the generic fallback below assumes.
        normal = data["true_normal"]
        return np.eye(ambient)[None] - np.einsum("ni,nj->nij", normal, normal)
    tangent = data["truth"] / np.maximum(np.linalg.norm(data["truth"], axis=1, keepdims=True), EPS)
    return np.einsum("ni,nj->nij", tangent, tangent)


def mechanism_metrics(
    scenario: str,
    Vhat: np.ndarray,
    data: dict,
    neighbors: np.ndarray,
) -> dict[str, float]:
    difference = Vhat - data["truth"]
    projector = true_projector(scenario, data)
    tangent_error = np.einsum("nij,nj->ni", projector, difference)
    normal_error = difference - tangent_error
    cross = cross_branch_mask(scenario, neighbors, data["labels"], data["P"])
    return {
        "identity_velocity_rmse": float(np.sqrt(np.mean(np.sum(difference**2, axis=1)))),
        "velocity_speed_rmse": float(np.sqrt(np.mean((np.linalg.norm(Vhat, axis=1) - np.linalg.norm(data["truth"], axis=1)) ** 2))),
        "velocity_direction_angle_error": angle_mae(Vhat, data["truth"], np.ones(len(Vhat), bool)),
        "velocity_direction_angle_valid_fraction": float(np.mean((np.linalg.norm(Vhat, axis=1) > 1e-8) & (np.linalg.norm(data["truth"], axis=1) > 1e-8))),
        "tangential_component_rmse": float(np.sqrt(np.mean(np.sum(tangent_error**2, axis=1)))),
        "normal_component_rmse": float(np.sqrt(np.mean(np.sum(normal_error**2, axis=1)))),
        "graph_cross_branch_edge_fraction": float(cross.mean()),
    }


def graphvelo_scale_metrics(Vhat: np.ndarray, data: dict) -> dict[str, float | bool]:
    """Scale audit; oracle rescaling is diagnostic and never enters ranking."""
    truth = np.asarray(data["truth"], float)
    noisy = np.asarray(data["field"], float)
    estimate = np.asarray(Vhat, float)
    denominator = float(np.sum(estimate**2))
    oracle_scale = max(0.0, float(np.sum(estimate * truth)) / max(denominator, EPS))
    oracle = oracle_scale * estimate
    return {
        "output_input_median_norm_ratio": float(
            np.median(np.linalg.norm(estimate, axis=1))
            / max(float(np.median(np.linalg.norm(noisy, axis=1))), EPS)
        ),
        "identity_velocity_rmse": float(np.sqrt(np.mean(np.sum((estimate - truth) ** 2, axis=1)))),
        "angle_error_degrees": angle_mae(estimate, truth, np.ones(len(estimate), bool)),
        "speed_rmse": float(np.sqrt(np.mean((np.linalg.norm(estimate, axis=1) - np.linalg.norm(truth, axis=1)) ** 2))),
        "oracle_nonnegative_global_scale": oracle_scale,
        "oracle_rescaled_velocity_rmse": float(np.sqrt(np.mean(np.sum((oracle - truth) ** 2, axis=1)))),
        "oracle_uses_clean_truth": True,
        "oracle_enters_primary_ranking": False,
    }


def fit_final_states(
    scenario: str,
    seed: int,
    data: dict,
    selected: dict,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict], list[dict]]:
    Y, W, d = data["Y"], data["field"], int(data["d"])
    base_k = int(selected["shared_graph_k"][scenario])
    cosine_graph = shared_knn_graph(Y, base_k)
    noisy_graph = official_neighbors(Y, GRAPHVELO_CONFIG["n_neighbors"])
    states = {"ambient_noisy": (Y.copy(), W.copy())}
    direction, _ = cosine_kernel_projection(Y, W, cosine_graph)
    cosine, _ = restore_noisy_speed(direction, W)
    states["cosine_kernel"] = (Y.copy(), cosine)
    graphvelo_raw, raw_info = graphvelo_velocity(Y, W)
    graphvelo, standardized_info = graphvelo_velocity_standardized(Y, W)
    states["graphvelo"] = (Y.copy(), graphvelo)
    joint_X, joint_V, _ = joint_low_rank_state(Y, W)
    states["joint_low_rank"] = (joint_X, joint_V)
    local_X, _ = local_pca_denoise(Y, d, n_neighbors=base_k, return_info=True)
    states["local_pca"] = (local_X, downstream_velocity(local_X, W, d, base_k))
    pos = selected["position_only_manfit"][scenario]
    pos_X = position_only_trajectory(Y, W, d, pos["k"], pos["T"], pos["eta_g"])[-1][1]
    states["position_only_manfit"] = (pos_X, downstream_velocity(pos_X, W, d, base_k))
    vmf = fit_vmf_variant(
        Y, W, d, selected["velocity_manifold_fitter"][scenario], seed
    )
    mechanism = [
        {
            "scenario": scenario,
            "seed": seed,
            "method": "graphvelo",
            **mechanism_metrics(scenario, graphvelo, data, noisy_graph),
        },
        {
            "scenario": scenario,
            "seed": seed,
            "method": "graphvelo_raw",
            **mechanism_metrics(scenario, graphvelo_raw, data, noisy_graph),
        },
        {
            "scenario": scenario,
            "seed": seed,
            "method": "manfitvelo",
            **mechanism_metrics(scenario, vmf["V"], data, vmf["neighbors"]),
        },
    ]
    scale_audit = [
        {
            "scenario": scenario,
            "seed": seed,
            "variant": "standardized_primary",
            "position_scale": standardized_info["position_scale"],
            "velocity_scale": standardized_info["velocity_scale"],
            "normalization_uses_clean_truth": False,
            "selected_by_performance": False,
            **graphvelo_scale_metrics(graphvelo, data),
        },
        {
            "scenario": scenario,
            "seed": seed,
            "variant": "raw_official_sensitivity",
            "position_scale": 1.0,
            "velocity_scale": 1.0,
            "normalization_uses_clean_truth": False,
            "selected_by_performance": False,
            **graphvelo_scale_metrics(graphvelo_raw, data),
        },
    ]
    states["manfitvelo"] = (vmf["X"], vmf["V"])
    return states, mechanism, scale_audit


def run_final(selected: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, mechanism_rows, tau_rows, scale_rows = [], [], [], []
    final_id = "manfitvelo"
    methods = BASE_METHODS + (final_id,)
    for scenario in SCENARIOS:
        for seed in FINAL_SEEDS:
            data = vector_data(scenario, seed)
            tau_k = int(selected["shared_graph_k"][scenario])
            tau_graph = shared_knn_graph(data["Y"], tau_k)
            tau, median_knn, median_speed = observed_tau(data["Y"], data["field"], tau_graph)
            baseline = state_metrics(scenario, data["Y"], data["field"], data, tau)
            sample_hash = array_hash(data["Y"], data["field"], data["P"], data["truth"])
            states, mechanisms, scale_audit = fit_final_states(scenario, seed, data, selected)
            mechanism_rows.extend(mechanisms)
            scale_rows.extend(scale_audit)
            tau_rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "tau": tau,
                    "k": tau_k,
                    "median_observed_knn_edge_distance": median_knn,
                    "median_observed_velocity_norm": median_speed,
                }
            )
            for method in methods:
                Xhat, Vhat = states[method]
                # Lightweight numerical-failure audit (near-zero cost): count
                # non-finite outputs per (scenario, seed, method). Not a full
                # failure-rate protocol -- deferred until stress-test sweeps
                # (Scan B/C) actually exercise high-noise regimes where this
                # would matter; see simulation/log.md.
                nan_inf_count = int(np.sum(~np.isfinite(Xhat)) + np.sum(~np.isfinite(Vhat)))
                if nan_inf_count:
                    Xhat = np.nan_to_num(Xhat, nan=0.0, posinf=0.0, neginf=0.0)
                    Vhat = np.nan_to_num(Vhat, nan=0.0, posinf=0.0, neginf=0.0)
                absolute = state_metrics(scenario, Xhat, Vhat, data, tau)
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "method": method,
                        "method_label": BASE_LABELS[method],
                        "evaluation_sample_hash": sample_hash,
                        "tau": tau,
                        "nan_inf_count": nan_inf_count,
                        **absolute,
                        **relative_state_metrics(absolute, baseline),
                    }
                )
    frame = pd.DataFrame(rows)
    mechanism = pd.DataFrame(mechanism_rows)
    paired_rows = []
    metrics = [
        "identity_velocity_rmse",
        "velocity_speed_rmse",
        "velocity_direction_angle_error",
        "tangential_component_rmse",
        "normal_component_rmse",
        "graph_cross_branch_edge_fraction",
    ]
    for scenario, group in mechanism.groupby("scenario"):
        final_method = final_id
        for metric in metrics:
            pivot = group[group.method.isin(["graphvelo", final_method])].pivot(
                index="seed", columns="method", values=metric
            )
            difference = pivot[final_method] - pivot["graphvelo"]
            for seed, value in difference.items():
                paired_rows.append(
                    {
                        "record_type": "paired_difference",
                        "scenario": scenario,
                        "seed": seed,
                        "method": f"{final_method}_minus_graphvelo",
                        "metric": metric,
                        "value": value,
                        "median_paired_difference": float(difference.median()),
                        "manfitvelo_win_fraction": float((difference < 0).mean()),
                    }
                )
    mechanism_long = mechanism.melt(
        id_vars=["scenario", "seed", "method"], var_name="metric", value_name="value"
    )
    mechanism_long["record_type"] = "method_metric"
    return (
        frame,
        pd.concat([mechanism_long, pd.DataFrame(paired_rows)], ignore_index=True),
        pd.DataFrame(tau_rows),
        pd.DataFrame(scale_rows),
    )


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, method, label), group in frame.groupby(
        ["scenario", "method", "method_label"], sort=False
    ):
        row = {"scenario": scenario, "method": method, "method_label": label}
        for metric in PRIMARY_METRICS:
            row[f"{metric}_median"] = float(group[metric].median())
            row[f"{metric}_q25"] = float(group[metric].quantile(0.25))
            row[f"{metric}_q75"] = float(group[metric].quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def representative_figure(
    output: Path, scenario: str, selected: dict
) -> tuple[Path, str]:
    data = vector_data(scenario, PILOT_SEED)
    states, _, _ = fit_final_states(scenario, PILOT_SEED, data, selected)
    final_id = "manfitvelo"
    methods = BASE_METHODS + (final_id,)
    panels = {"truth": (data["P"], data["truth"]), **states}
    panel_order = ("truth",) + methods
    indices = np.linspace(0, len(data["Y"]) - 1, 36, dtype=int)
    all_positions = np.vstack([panels[method][0] for method in panel_order])
    limits = []
    for dim in range(3):
        low, high = np.quantile(all_positions[:, dim], [0.002, 0.998])
        margin = 0.07 * max(high - low, 0.1)
        limits.append((low - margin, high + margin))
    median_speed = float(np.median(np.linalg.norm(data["field"][indices], axis=1)))
    target_length = 0.075 * max(limits[0][1] - limits[0][0], limits[1][1] - limits[1][0])
    is_3d = scenario in {"flat_rotation_annulus", "half_sphere_tangent", "swiss_roll", "saddle_surface"}
    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    axes = [fig.add_subplot(2, 4, i + 1, projection="3d" if is_3d else None) for i in range(8)]
    for ax, method in zip(axes, panel_order):
        Xhat, Vhat = panels[method]
        if is_3d:
            ax.scatter(Xhat[:, 0], Xhat[:, 1], Xhat[:, 2], c=data["labels"], s=5, alpha=0.45)
            ax.quiver(
                Xhat[indices, 0], Xhat[indices, 1], Xhat[indices, 2],
                Vhat[indices, 0], Vhat[indices, 1], Vhat[indices, 2],
                length=target_length / max(median_speed, EPS), normalize=False, linewidth=0.7,
            )
            ax.set_zlim(*limits[2]); ax.set_zticks([]); ax.view_init(24, -58)
        else:
            ax.scatter(Xhat[:, 0], Xhat[:, 1], c=data["labels"], s=5, alpha=0.45)
            ax.quiver(
                Xhat[indices, 0], Xhat[indices, 1], Vhat[indices, 0], Vhat[indices, 1],
                angles="xy", scale_units="xy", scale=max(median_speed, EPS) / target_length,
                width=0.004,
            )
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(BASE_LABELS[method], fontsize=9)
    axes[-1].axis("off")
    fig.suptitle(f"{SCENARIO_LABELS[scenario]} — fixed pilot seed {PILOT_SEED}")
    path = output / "figures" / f"state_{scenario}.png"
    fig.savefig(path, dpi=170, facecolor="white"); plt.close(fig)
    return path, hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def metric_table(summary: pd.DataFrame, scenario: str, methods: tuple[str, ...]) -> str:
    subset = summary[summary.scenario.eq(scenario)].set_index("method").reindex(methods)
    rounded_winner = {
        metric: subset[f"{metric}_median"].round(3).idxmin() for metric in PRIMARY_METRICS
    }
    group_headers = (
        ("Geometry", PRIMARY_METRICS[:2]),
        ("Velocity at original identity", PRIMARY_METRICS[2:4]),
        ("Location / joint consistency", PRIMARY_METRICS[4:]),
    )
    first = "<tr><th rowspan='2'>Method</th>" + "".join(
        f"<th colspan='{len(metrics)}'>{title}</th>" for title, metrics in group_headers
    ) + "</tr>"
    second = "<tr>" + "".join(
        f"<th>{escape(METRIC_LABELS[metric])}</th>" for _, metrics in group_headers for metric in metrics
    ) + "</tr>"
    body = []
    for method in methods:
        cells = []
        for metric in PRIMARY_METRICS:
            median = subset.loc[method, f"{metric}_median"]
            q25 = subset.loc[method, f"{metric}_q25"]
            q75 = subset.loc[method, f"{metric}_q75"]
            cls = "best" if method == rounded_winner[metric] else ""
            cells.append(f"<td class='{cls}'>{median:.3f} <small>[{q25:.3f}, {q75:.3f}]</small></td>")
        body.append(f"<tr><th>{escape(BASE_LABELS[method])}</th>{''.join(cells)}</tr>")
    return f"<table><thead>{first}{second}</thead><tbody>{''.join(body)}</tbody></table>"


def scenario_text(
    summary: pd.DataFrame, mechanism: pd.DataFrame, scenario: str, final_id: str
) -> str:
    table = summary[summary.scenario.eq(scenario)].set_index("method")
    vmf = table.loc[final_id]
    graph = table.loc["graphvelo"]
    statements = [
        f"Geometry: ManfitVelo clean-point and manifold ratios are {vmf.clean_point_rmse_rel_median:.2f} and {vmf.distance_to_manifold_rel_median:.2f}.",
        f"Identity velocity: RMSE and angle ratios are {vmf.velocity_rmse_id_rel_median:.2f} and {vmf.velocity_angle_mae_id_rel_median:.2f}.",
        f"Location/joint: ratios are {vmf.velocity_rmse_loc_rel_median:.2f} and {vmf.joint_euler_state_rmse_rel_median:.2f}.",
    ]
    if graph.velocity_rmse_id_rel_median < vmf.velocity_rmse_id_rel_median:
        method_rows = mechanism[
            (mechanism.record_type == "method_metric")
            & mechanism.scenario.eq(scenario)
            & mechanism.method.isin(["graphvelo", final_id])
        ]
        pivot = method_rows.pivot_table(index="method", columns="metric", values="value", aggfunc="median")
        graph_angle = pivot.loc["graphvelo", "velocity_direction_angle_error"]
        vmf_angle = pivot.loc[final_id, "velocity_direction_angle_error"]
        graph_speed = pivot.loc["graphvelo", "velocity_speed_rmse"]
        vmf_speed = pivot.loc[final_id, "velocity_speed_rmse"]
        angle_better = graph_angle < 0.99 * vmf_angle
        speed_better = graph_speed < 0.99 * vmf_speed
        if not angle_better and speed_better:
            reason = "lower speed/magnitude error rather than better direction"
        elif angle_better and speed_better:
            reason = "both direction reconstruction and speed smoothing"
        elif angle_better:
            reason = "direction reconstruction"
        else:
            reason = "the combined tangential/normal error budget; neither speed nor angle alone explains it"
        statements.append(f"GraphVelo has lower identity RMSE here; its diagnostic advantage is attributable mainly to {reason}.")
    if scenario == "curved_hairpin":
        rows = mechanism[
            (mechanism.record_type == "method_metric")
            & mechanism.scenario.eq(scenario)
            & mechanism.metric.eq("graph_cross_branch_edge_fraction")
        ]
        cross = rows.groupby("method").value.median()
        statements.append(
            f"With the reach-audited hairpin, median cross-arm fractions are {cross.get('graphvelo', np.nan):.3f} for GraphVelo and {cross.get(final_id, np.nan):.3f} for the selected ManfitVelo pipeline."
        )
    if scenario == "flat_rotation_annulus" and graph.velocity_rmse_id_rel_median < vmf.velocity_rmse_id_rel_median:
        statements.append("This d=2 result is consistent with GraphVelo smoothing velocity noise inside the true tangent plane, which tangent projection alone cannot remove.")
    return " ".join(statements)


class AuditParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(); self.images = []; self.external = []
    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "img": self.images.append(values.get("src", ""))
        if tag in {"link", "script"}:
            value = values.get("href") or values.get("src") or ""
            if value and not value.startswith("data:"): self.external.append(value)


def experiment_parameter_table(selected: dict) -> str:
    geometry = {
        "circle": "t~Unif(0,2π); X=(cos t,sin t,0); V=(-sin t,cos t,0)",
        "s_curve": "t~Unif(-1.4,1.4); X=(sin(1.6t),t,0); normalized derivative field",
        "curved_hairpin": f"arm length 1.2; curvature 0.15; separation {HAIRPIN_DEFAULT_SEPARATION:.2f}; bend radius {HAIRPIN_DEFAULT_SEPARATION/2:.2f}",
        "flat_rotation_annulus": "r²~Unif(0.35²,1), t~Unif(0,2π); V=(-y,x,0)",
        "half_sphere_tangent": "z~Unif(0,1), azimuth~Unif(0,2π); b=(0.7,-0.4,0.6) projected to tangent; clean speed>0.18",
        "y_branch": "three length-1 branches with directions (0,-1),(-0.8,0.8),(0.8,0.8); outward unit flow",
        "near_intersection": "x~Unif(-1.1,1.1); separation 0.13; curvature 0.3; opposing branch flow",
        "swiss_roll": "t~Unif(1.5π,3.5π) [one winding], y~Unif(-1,1); X=(t cos t, y, t sin t)/3.5π; flow along increasing t",
        "saddle_surface": "u,v~Unif(-1,1); X=(u,v,0.45(u²-v²)); flow along +u",
    }
    rows = []
    for scenario in SCENARIOS:
        setting = SETS[scenario]
        pos = selected["position_only_manfit"][scenario]
        vmf = selected["velocity_manifold_fitter"][scenario]
        rows.append({
            "Scenario": SCENARIO_LABELS[scenario],
            "n": setting.n,
            "d / D": f"{setting.d} / 3",
            "σ_X / σ_V": f"{setting.px:g} / {setting.field_noise:g}",
            "Geometry / field": geometry[scenario],
            "Cosine & downstream k": selected["shared_graph_k"][scenario],
            "Local PCA k": selected["local_pca"][scenario]["k"],
            "Position-only (k, T, η_g)": f"({pos['k']}, {pos['T']}, {pos['eta_g']:g})",
            "ManfitVelo (k, T, η_g, θ, κ, λ_v)": (
                f"({vmf['k']}, {vmf['T']}, {vmf['eta_g']:g}, {vmf['theta']:g}, "
                f"{vmf.get('kappa', 0):g}, {vmf.get('lambda_v', 0):g})"
            ),
        })
    main = pd.DataFrame(rows).to_html(index=False, border=0, escape=True)
    full_rows = []
    for scenario in SCENARIOS:
        full_rows.extend([
            {"Scenario": SCENARIO_LABELS[scenario], "Pipeline": "Local PCA", "Complete frozen config": json.dumps(selected["local_pca"][scenario], sort_keys=True)},
            {"Scenario": SCENARIO_LABELS[scenario], "Pipeline": "Downstream velocity", "Complete frozen config": json.dumps(selected.get("downstream_velocity", {}).get(scenario, {"k": selected["shared_graph_k"][scenario], "rank": SETS[scenario].d}), sort_keys=True)},
            {"Scenario": SCENARIO_LABELS[scenario], "Pipeline": "Position-only MANFIT", "Complete frozen config": json.dumps(selected["position_only_manfit"][scenario], sort_keys=True)},
            {"Scenario": SCENARIO_LABELS[scenario], "Pipeline": "ManfitVelo", "Complete frozen config": json.dumps(selected["velocity_manifold_fitter"][scenario], sort_keys=True)},
        ])
    full = pd.DataFrame(full_rows).to_html(index=False, border=0, escape=True)
    return main + "<details><summary>Show every frozen per-scenario algorithm parameter</summary>" + full + "</details>"


def metric_definition_html() -> str:
    rows = [
        ("clean_point_rmse_rel", "RMSE(X̂, X_clean) / RMSE(X_noisy, X_clean)", "Recovery of the original cell positions; preserves cell identity."),
        ("distance_to_manifold_rel", "RMSE(dist(X̂, M_true)) / RMSE(dist(X_noisy, M_true))", "Geometric support recovery, irrespective of sliding along the manifold."),
        ("velocity_rmse_id_rel", "RMSE(V̂_i, V_true(X_clean,i)) / noisy counterpart", "Velocity recovery for the original generating cell identity."),
        ("velocity_angle_mae_id_rel", "mean arccos(⟨V̂_i,V_i⟩/(‖V̂_i‖‖V_i‖)) / noisy counterpart", "Directional error at original identity; valid nonzero-vector fraction is recorded."),
        ("velocity_rmse_loc_rel", "RMSE(V̂_i, V_true(Π_M(X̂_i))) / noisy counterpart", "Compatibility with the vector field at the reconstructed location; branch-aware where needed and does not penalize tangential sliding."),
        ("joint_euler_state_rmse_rel", "RMSE[(X̂+τV̂)−(X_clean+τV_true)] / noisy counterpart", "Short-step joint state/forecast recovery; not a pure velocity metric."),
    ]
    table = pd.DataFrame(rows, columns=["Metric", "Mathematical definition", "Brief meaning"]).to_html(index=False, border=0, escape=True)
    return table + "<p><code>τ = 0.5 × median observed kNN distance / median observed velocity norm</code>. τ is determined once from noisy observations for each scenario/seed and shared by all methods. All relative metrics are lower-is-better and Ambient noisy input is exactly 1.</p>"


def graphvelo_scale_audit_table(scale_audit: pd.DataFrame) -> str:
    columns = (
        "output_input_median_norm_ratio", "angle_error_degrees", "speed_rmse",
        "identity_velocity_rmse", "oracle_nonnegative_global_scale",
        "oracle_rescaled_velocity_rmse",
    )
    rows = []
    for (scenario, variant), group in scale_audit.groupby(["scenario", "variant"], sort=False):
        row = {"Scenario": SCENARIO_LABELS[scenario], "Variant": variant}
        for column in columns:
            row[column] = f"{group[column].median():.3f} [{group[column].quantile(.25):.3f}, {group[column].quantile(.75):.3f}]"
        rows.append(row)
    return pd.DataFrame(rows).to_html(index=False, border=0, escape=True)


def build_report(
    output: Path,
    summary: pd.DataFrame,
    mechanism: pd.DataFrame,
    scale_audit: pd.DataFrame,
    selected: dict,
    hairpin_selected: dict,
    *,
    regenerate_figures: bool = True,
) -> tuple[dict, dict]:
    final_id = "manfitvelo"
    methods = BASE_METHODS + (final_id,)
    figures, arrow_hash = {}, {}
    for scenario in SCENARIOS:
        if regenerate_figures:
            figures[scenario], arrow_hash[scenario] = representative_figure(output, scenario, selected)
        else:
            figures[scenario] = output / "figures" / f"state_{scenario}.png"
            if not figures[scenario].exists():
                raise FileNotFoundError(f"report-only requires {figures[scenario]}")
            prior = output / "sanity_checks.json"
            hashes = json.loads(prior.read_text()).get("arrow_subsample_hashes", {}) if prior.exists() else {}
            arrow_hash[scenario] = hashes.get(scenario, "preserved-existing-figure")
    style = """
body{margin:0;background:#f4f6f8;color:#17212b;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}main{max-width:1320px;margin:auto;padding:28px 20px 60px}.card{background:#fff;border:1px solid #d8dee7;border-radius:9px;padding:20px;margin:18px 0;overflow:auto}table{border-collapse:collapse;width:100%;font-size:11px}th,td{border:1px solid #d8dee7;padding:6px;text-align:center}th{background:#edf1f5}tbody th{text-align:left;white-space:nowrap}td.best{background:#cdeed4;font-weight:650}small{color:#56616f}img{max-width:100%;height:auto}p{line-height:1.55}.note{border-left:4px solid #3978b8;background:#eef6ff;padding:10px}code{background:#edf1f5;padding:1px 4px}
"""
    pipeline_table = pd.DataFrame(
        [
            ("Ambient noisy input", "X_noisy", "V_noisy"),
            ("Cosine kernel", "X_noisy", "Cosine direction + noisy speed"),
            ("GraphVelo (standardized)", "X_noisy", "Official TSP in fixed noisy-data units; mapped back to velocity units"),
            ("Joint Low-Rank (M3)", "Block-normalized joint [X,V] SVD reconstruction", "Same joint SVD reconstruction, unscaled to velocity units"),
            ("Local PCA", "Local affine reconstruction", "Rebuilt local-tangent projection"),
            ("Position-only MANFIT", "Position-only fit", "Rebuilt local-tangent projection"),
            (BASE_LABELS[final_id], "Velocity-aware geometry fit", "Final fitted-tangent projection"),
        ], columns=["Method", "Position output", "Velocity output"]
    ).to_html(index=False, border=0)
    parameter_table = experiment_parameter_table(selected)
    scale_table = graphvelo_scale_audit_table(scale_audit)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Reach-audited ManfitVelo benchmark</title><style>{style}</style></head><body><main>",
        "<h1>Reach-audited ManfitVelo state-recovery benchmark</h1>",
        "<section class='card'><p>The target is recovery of the clean manifold state <code>(X,V)</code> from ambient noisy observations. Only prespecified default noise is shown. Entries are median [IQR] over final seeds 43000–43014, normalized by the matching ambient-noisy error.</p>",
        "<p>Identity anchoring preserves the generating cell; branch-aware location anchoring evaluates compatibility at the reconstructed location but does not penalize tangential sliding. Joint Euler error uses one noisy-observation-derived tau per scenario/seed.</p>",
        f"{pipeline_table}<p class='note'>Local PCA is the complete position–velocity pipeline: it reconstructs positions, rebuilds neighborhoods and tangents, and then reconstructs velocity. Primary GraphVelo applies the unchanged official analytical-manifold objective after one globally fixed truth-free unit conversion. Joint Low-Rank (M3) has no fitted hyperparameter beyond the fixed {JOINT_LOW_RANK_VARIANCE_THRESHOLD:.2f} cumulative-explained-variance threshold, which is never selected from ground truth or performance; ManfitVelo used only seeds 42000–42002 for selection. Final seeds were never used for selection.</p></section>",
        f"<section class='card'><h2>Experiment parameters</h2><p>All rows use the default noise only. Final seeds are 43000–43014; tuning seeds are 42000–42002; visualization seed is {PILOT_SEED}. GraphVelo is fixed globally at <code>n_neighbors=15, a=1, b=0, r=1, loss_func=linear</code>. Joint Low-Rank (M3) truncates its joint SVD at {JOINT_LOW_RANK_VARIANCE_THRESHOLD:.2f} cumulative explained variance, chosen fresh per sample from the observed singular-value spectrum only. No parameter below is selected from final performance.</p>{parameter_table}</section>",
        f"<section class='card'><h2>Metric definitions</h2>{metric_definition_html()}</section>",
        f"<section class='card'><h2>GraphVelo scale-equivariance audit</h2><p>The fixed-ridge TSP objective <code>a‖Σ_j φ_ij(x_j−x_i)−v_i‖² + r‖φ_i‖²</code> is not invariant to independent changes of position and velocity units. The main GraphVelo row therefore uses <code>X*=(X_noisy−mean(X_noisy))/s_X</code> and <code>V*=V_noisy/s_V</code>, where <code>s_X</code> is the median positive displacement norm in the official 15-NN graph and <code>s_V</code> is the median noisy velocity norm. It runs the unchanged official configuration and maps back with <code>V̂=s_V V̂*</code>. This rule is fixed, truth-free, and never performance-selected.</p><p>The raw-scale official result is retained below as provenance/sensitivity. Oracle scaling is computed only after fitting as <code>c*=max(0, ⟨V̂,V_true⟩/‖V̂‖²)</code>, followed by <code>RMSE(c*V̂,V_true)</code>. It uses clean truth, is diagnostic only, and never enters the seven-method table, highlighting, or interpretation of the primary ranking.</p>{scale_table}</section>",
    ]
    for scenario in SCENARIOS:
        parts.extend(
            [
                f"<section class='card'><h2>{escape(SCENARIO_LABELS[scenario])}</h2>",
                f"<img src='{image_uri(figures[scenario])}' alt='Unified state visualization for {escape(SCENARIO_LABELS[scenario])}'>",
                metric_table(summary, scenario, methods),
                f"<p>{escape(scenario_text(summary, mechanism, scenario, final_id))}</p></section>",
            ]
        )
    parts.extend(
        [
            "</main></body></html>",
        ]
    )
    html = "".join(parts)
    report = output / "final_report.html"; report.write_text(html, encoding="utf-8")
    parser = AuditParser(); parser.feed(html)
    audit = {
        "self_contained_html": all(value.startswith("data:image/png;base64,") for value in parser.images) and not parser.external,
        "embedded_figure_count": len(parser.images),
        "expected_figure_count": len(SCENARIOS),
        "only_default_noise": all(term not in html for term in ("zero-position-noise", "double-position-noise", "position noise: zero")),
        "joint_low_rank_present": "Joint Low-Rank" in html,
        "old_local_pca_names_absent": all(term not in html for term in ("Local PCA–XV", "local_pca_xv", "Local PCA position–velocity denoiser")),
        "no_cross_scenario_heatmap": "heatmap" not in html.lower(),
        "metric_definitions_present": "Mathematical definition" in html,
        "experiment_parameters_present": "Experiment parameters" in html,
        "graphvelo_scale_audit_present": "GraphVelo scale-equivariance audit" in html,
        "hairpin_reach_section_absent": "<h2>Hairpin reach audit</h2>" not in html,
        "conservative_synthesis_absent": "<h2>Conservative synthesis</h2>" not in html,
    }
    audit["html_checks_pass"] = bool(all(audit.values()))
    return arrow_hash, audit


def validate(frame: pd.DataFrame, scale_audit: pd.DataFrame, html_audit: dict) -> dict:
    final_id = "manfitvelo"
    methods = set(BASE_METHODS + (final_id,))
    noisy = frame[frame.method.eq("ambient_noisy")]
    checks = {
        "ambient_noisy_relative_metrics_equal_one": bool(np.allclose(noisy[list(PRIMARY_METRICS)], 1.0, atol=1e-12)),
        "all_final_methods_share_seeds": bool(frame.groupby(["scenario", "method"]).seed.nunique().eq(len(FINAL_SEEDS)).all()),
        "all_final_methods_share_samples": bool(frame.groupby(["scenario", "seed"]).evaluation_sample_hash.nunique().eq(1).all()),
        "exact_final_method_set": set(frame.method) == methods,
        "final_seeds_used_for_selection": False,
        "graphvelo_uses_oracle_information": False,
        "graphvelo_primary_is_truth_free_standardized": bool(
            set(scale_audit.variant) == {"standardized_primary", "raw_official_sensitivity"}
            and not scale_audit.normalization_uses_clean_truth.any()
            and not scale_audit.selected_by_performance.any()
            and not scale_audit.oracle_enters_primary_ranking.any()
        ),
        "hairpin_selection_uses_method_results": False,
        "hairpin_selection_uses_final_seeds": False,
        **html_audit,
    }
    required_true = (
        "ambient_noisy_relative_metrics_equal_one",
        "all_final_methods_share_seeds",
        "all_final_methods_share_samples",
        "exact_final_method_set",
        "self_contained_html",
        "only_default_noise",
        "joint_low_rank_present",
        "old_local_pca_names_absent",
        "no_cross_scenario_heatmap",
        "graphvelo_primary_is_truth_free_standardized",
        "html_checks_pass",
    )
    checks["all_checks_pass"] = bool(
        all(checks[key] is True for key in required_true)
        and not checks["final_seeds_used_for_selection"]
        and not checks["graphvelo_uses_oracle_information"]
        and not checks["hairpin_selection_uses_method_results"]
        and not checks["hairpin_selection_uses_final_seeds"]
    )
    if not checks["all_checks_pass"]:
        raise AssertionError(checks)
    return checks


def write_readme(output: Path, selected: dict) -> None:
    text = f"""# Reach-audited ManfitVelo benchmark

The old curved hairpin used separation `0.13` and bend radius `0.065`, giving a
reach proxy only 2.6 times the position-noise standard deviation. At the old
`k=20`, local-neighborhood radius and cross-arm mixing were too large relative
to reach. The new default uses separation `{HAIRPIN_DEFAULT_SEPARATION:.2f}` and
bend radius `{HAIRPIN_DEFAULT_SEPARATION/2:.2f}`. It was selected from geometry
and noisy-neighborhood diagnostics on tuning seeds, without method rankings.

`Local PCA` here is the complete pipeline: local affine position
reconstruction, rebuilt neighborhoods/tangent projectors at the reconstructed
positions, and downstream velocity projection.

GraphVelo uses the fixed official analytical-manifold path with 15 neighbors,
cosine kernel, density correction, and `a=1, b=0, r=1, loss_func=linear`; it is
not tuned. The primary row applies a fixed truth-free unit conversion using the
median noisy 15-NN distance and median noisy velocity norm. The exact raw-scale
call is retained in `graphvelo_scale_audit.csv` as a sensitivity/provenance
result. Oracle global rescaling is diagnostic only. Joint Low-Rank (M3) block-
normalizes position and velocity by their own Frobenius norms, concatenates
them, truncates the joint SVD at a fixed {JOINT_LOW_RANK_VARIANCE_THRESHOLD:.2f}
cumulative-explained-variance threshold (chosen fresh per sample from the
observed spectrum only, never from ground truth), and inverts with an exact
affine unscale. ManfitVelo uses its final fitted tangent projectors for
velocity reconstruction.

Primary metrics remain identity/location anchored and short-step Euler state
error. The latter is a joint forecast error and is not a pure velocity metric.
`graphvelo_mechanism_diagnostics.csv` additionally reports speed, direction,
tangential/normal component errors, graph contamination, seed-paired
differences, and win fractions.

## Reproduce

```bash
python simulation/run_manfitvelo_benchmark.py
```

Tuning seeds are 42000–42002; final seeds are 43000–43014. Final seeds never
participate in geometry or model selection.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def environment_provenance() -> dict:
    packages = {}
    for name in ("numpy", "scipy", "pandas", "scikit-learn", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
        "graphvelo": GRAPHVELO_PROVENANCE,
        "graphvelo_config": GRAPHVELO_CONFIG,
        "graphvelo_standardization": GRAPHVELO_STANDARDIZATION,
    }


def main() -> None:
    args = parse_args(); output = args.output_dir
    (output / "figures").mkdir(parents=True, exist_ok=True)
    if args.report_only:
        frame = pd.read_csv(output / "final_seed_metrics.csv")
        summary = pd.read_csv(output / "summary_metrics.csv")
        mechanism = pd.read_csv(output / "graphvelo_mechanism_diagnostics.csv")
        scale_audit = pd.read_csv(output / "graphvelo_scale_audit.csv")
        selected = json.loads((output / "selected_hyperparameters.json").read_text())
        (output / "selected_hyperparameters.json").write_text(
            json.dumps(selected, indent=2) + "\n"
        )
        hairpin_selected = selected["hairpin_geometry"]
        arrow_hash, html_audit = build_report(
            output, summary, mechanism, scale_audit, selected, hairpin_selected,
            regenerate_figures=False,
        )
        checks = validate(frame, scale_audit, html_audit)
        checks["arrow_subsample_hashes"] = arrow_hash
        (output / "sanity_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        write_readme(output, selected)
        return

    reach, point_cross, hairpin_selected = hairpin_reach_diagnostics()
    reach.to_csv(output / "hairpin_reach_neighborhood_diagnostics.csv", index=False)
    point_cross.to_csv(output / "hairpin_point_cross_arm_distribution.csv", index=False)
    # hairpin_reach_diagnostics() is retained purely as a geometry-validity
    # audit (it asserts the frozen hairpin separation still satisfies the
    # reach/cross-arm conditions); its own "k" is intentionally not used
    # operationally any more -- see neighbor_count_rule below.

    # Shared k(n,d) neighborhood rule (Weekly Plan v1.1 section 4), refined by
    # a curvature-aware probe (added 2026-08-11, see benchmark_core.py and
    # simulation/log.md), applied to every scenario -- including Curved
    # Hairpin and Near Intersection -- with no scenario-specific exception
    # (decision recorded in simulation/log.md). The probe can only shrink k
    # relative to neighbor_count(n,d) (it searches up to that ceiling), never
    # grow it.
    scenario_k, curvature_diagnostics = curvature_aware_scenario_k()
    curvature_diagnostics.to_csv(output / "curvature_aware_k_diagnostics.csv", index=False)

    vmf_shared, vmf_tuning = tune_shared_vmf(scenario_k)
    position_shared, position_tuning = tune_shared_position_only(scenario_k)

    selected: dict = {
        "shared_graph_k": dict(scenario_k),
        "local_pca": {scenario: {"k": scenario_k[scenario]} for scenario in SCENARIOS},
        "position_only_manfit": {
            scenario: {**position_shared, "k": scenario_k[scenario]} for scenario in SCENARIOS
        },
        "velocity_manifold_fitter": {
            scenario: {**vmf_shared, "k": scenario_k[scenario]} for scenario in SCENARIOS
        },
        "neighbor_count_rule": {
            "base_formula": "k(n,d) = clip(ceil(C * n**(4/(d+4))), k_min, k_max)",
            "C": NEIGHBOR_SCALING_CONSTANT,
            "clip": list(NEIGHBOR_COUNT_CLIP),
            "curvature_refinement": (
                "sweep k up to the base-formula ceiling; measure population-mean "
                "normal-direction local-PCA residual (no ground truth) at each k; "
                "pick the k right after the log-log(residual) vs log(k) slope "
                "minimum -- see benchmark_core.curvature_aware_neighbor_count and "
                "simulation/log.md"
            ),
            "curvature_refinement_uses_final_seeds": False,
            "applies_to": ["cosine_kernel", "local_pca", "position_only_manfit", "velocity_manifold_fitter"],
            "scenario_specific_exception": False,
        },
    }
    selected["graphvelo"] = dict(GRAPHVELO_CONFIG)
    selected["graphvelo"]["primary_input_units"] = "truth-free standardized"
    selected["graphvelo"]["standardization"] = dict(GRAPHVELO_STANDARDIZATION)
    selected["graphvelo"]["raw_official_retained_as"] = "provenance/sensitivity diagnostic only"
    selected["graphvelo_provenance"] = dict(GRAPHVELO_PROVENANCE)
    selected["joint_low_rank"] = {
        "rule": (
            "M3 Joint Low-Rank Denoising: block-normalize [X,V] by their own Frobenius "
            "norms, truncate the joint SVD at cumulative explained variance >= threshold, "
            "exact affine unscale back to original units"
        ),
        "variance_threshold": JOINT_LOW_RANK_VARIANCE_THRESHOLD,
        "rank_selection": "per-sample, from the observed singular-value spectrum only",
        "rank_uses_ground_truth": False,
        "tuned": False,
        "replaces": "global_pca (Weekly Plan v1.1 section 2, M3)",
    }
    selected["hairpin_geometry"] = hairpin_selected
    selected["selection_uses_final_seeds"] = False
    selected["tuning_budget"] = {
        "shared_vmf_candidates": len(shared_vmf_grid()),
        "shared_position_only_candidates": len(shared_position_only_grid()),
        "graphvelo_candidates": 0,
        "tuning_seeds": list(TUNING_SEEDS),
        "tuning_scenarios": list(SCENARIOS),
        "final_seeds_used": False,
    }
    tuning = pd.concat([vmf_tuning, position_tuning], ignore_index=True)
    tuning.to_csv(output / "tuning_audit.csv", index=False)
    frame, mechanism, tau, scale_audit = run_final(selected)
    summary = summarize(frame)
    frame.to_csv(output / "final_seed_metrics.csv", index=False)
    summary.to_csv(output / "summary_metrics.csv", index=False)
    mechanism.to_csv(output / "graphvelo_mechanism_diagnostics.csv", index=False)
    scale_audit.to_csv(output / "graphvelo_scale_audit.csv", index=False)
    scale_summary = scale_audit.groupby(["scenario", "variant"], as_index=False).agg(
        output_input_median_norm_ratio_median=("output_input_median_norm_ratio", "median"),
        angle_error_degrees_median=("angle_error_degrees", "median"),
        speed_rmse_median=("speed_rmse", "median"),
        identity_velocity_rmse_median=("identity_velocity_rmse", "median"),
        oracle_nonnegative_global_scale_median=("oracle_nonnegative_global_scale", "median"),
        oracle_rescaled_velocity_rmse_median=("oracle_rescaled_velocity_rmse", "median"),
    )
    scale_summary.to_csv(output / "graphvelo_scale_audit_summary.csv", index=False)
    tau.to_csv(output / "tau_by_seed.csv", index=False)
    (output / "selected_hyperparameters.json").write_text(json.dumps(selected, indent=2) + "\n")
    (output / "environment_provenance.json").write_text(
        json.dumps(environment_provenance(), indent=2) + "\n"
    )
    arrow_hash, html_audit = build_report(
        output, summary, mechanism, scale_audit, selected, hairpin_selected
    )
    checks = validate(frame, scale_audit, html_audit); checks["arrow_subsample_hashes"] = arrow_hash
    (output / "sanity_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    write_readme(output, selected)
    print(output / "final_report.html")
    print(json.dumps({"hairpin": hairpin_selected, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
