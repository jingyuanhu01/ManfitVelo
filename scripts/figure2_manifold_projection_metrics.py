"""Compute Figure 2 metrics by projection onto the true manifold.

The current Figure 2 set contains four three-dimensional simulations: two
velocity-flow manifolds and two scalar-potential manifolds.

This evaluation does not require fitted cells to match their original paired
ground-truth samples. Instead, each noisy/fitted point is projected to the
nearest point on the analytic manifold, and metrics are computed there.

Outputs:
  - manifold_distance_rmse: distance from points to the true manifold.
  - manifold_distance_relative_rmse: normalized to the noisy baseline.
  - projected_velocity_rmse: RMSE between unit tangent-projected method
    velocities and the analytic unit velocity at the projected location.
  - projected_velocity_relative_rmse: normalized to the noisy baseline.
  - projected_velocity_cosine: mean cosine similarity for the same vectors.
  - tangent_residual_mean: mean norm of the component of unit velocity normal
    to the true tangent space.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


EPS = 1e-12
S_CURVE_WIDTH = 1.15
SWISS_ROLL_WIDTH = 2.3
SADDLE_RADIUS = 1.12

METHOD_DIRS = {
    "Noisy": "simulated_data",
    "ManfitVelo": "manfitvelo",
    "Position-only MANFIT": "position_only_manfit",
    "Local PCA": "local_pca",
}


@dataclass
class ProjectionResult:
    projected_x: np.ndarray
    true_v: np.ndarray
    projectors: np.ndarray
    distances: np.ndarray


@dataclass
class ManifoldGrid:
    points: np.ndarray
    true_v: np.ndarray
    projectors: np.ndarray
    nn: NearestNeighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=Path("outputs/figure2_pipeline"),
        help="Figure 2 pipeline output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for metric tables. Defaults to PIPELINE_DIR/metrics.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=220,
        help="Base dense-grid resolution for 3D manifold projection.",
    )
    return parser.parse_args()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + EPS)


def row_rmse(values: np.ndarray, truth: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if values.shape != truth.shape:
        raise ValueError(f"shape mismatch: {values.shape} versus {truth.shape}")
    return float(np.sqrt(np.mean(np.sum((values - truth) ** 2, axis=1))))


def mean_cosine(values: np.ndarray, truth: np.ndarray) -> float:
    values = normalize_rows(values)
    truth = normalize_rows(truth)
    cosines = np.sum(values * truth, axis=1)
    return float(np.mean(np.clip(cosines, -1.0, 1.0)))


def safe_relative(value: float, baseline: float) -> float:
    if not np.isfinite(baseline) or abs(baseline) <= EPS:
        return float("nan")
    return float(value / baseline)


def projector_from_basis(*vectors: np.ndarray) -> np.ndarray:
    basis = np.column_stack(vectors).astype(float)
    q, r = np.linalg.qr(basis)
    keep = np.abs(np.diag(r)) > EPS
    if not np.any(keep):
        return np.zeros((basis.shape[0], basis.shape[0]), dtype=float)
    q = q[:, keep]
    return q @ q.T


def project_velocity_to_tangent(vectors: np.ndarray, projectors: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj->ni", projectors, vectors)


def tangent_residual(vectors: np.ndarray, projectors: np.ndarray) -> np.ndarray:
    unit_v = normalize_rows(vectors)
    tangent_v = project_velocity_to_tangent(unit_v, projectors)
    return np.linalg.norm(unit_v - tangent_v, axis=1)


def s_curve_map(t: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.column_stack([np.sin(t), h, np.sign(t) * (np.cos(t) - 1.0)])


def s_curve_velocity(t: np.ndarray, h: np.ndarray) -> np.ndarray:
    dxdt = np.column_stack([np.cos(t), np.zeros_like(t), -np.sign(t) * np.sin(t)])
    dxdt[np.abs(t) < 1e-8, 2] = 0.0
    return dxdt + 0.25 * np.column_stack([np.zeros_like(h), np.sin(2.0 * h), np.zeros_like(h)])


def s_curve_projectors(t: np.ndarray) -> np.ndarray:
    projectors = np.zeros((t.size, 3, 3), dtype=float)
    dh = np.array([0.0, 1.0, 0.0])
    for i, ti in enumerate(t):
        dt = np.array([np.cos(ti), 0.0, -np.sign(ti) * np.sin(ti)])
        if abs(ti) < 1e-8:
            dt[2] = 0.0
        projectors[i] = projector_from_basis(dt, dh)
    return projectors


def swiss_roll_map(t: np.ndarray, width: np.ndarray) -> np.ndarray:
    return np.column_stack([t * np.cos(t) / 8.0, width, t * np.sin(t) / 8.0])


def swiss_roll_velocity(t: np.ndarray, width: np.ndarray) -> np.ndarray:
    dxdt = np.column_stack(
        [
            (np.cos(t) - t * np.sin(t)) / 8.0,
            np.zeros_like(t),
            (np.sin(t) + t * np.cos(t)) / 8.0,
        ]
    )
    return dxdt + 0.12 * np.column_stack([np.zeros_like(width), np.sin(width), np.zeros_like(width)])


def swiss_roll_projectors(t: np.ndarray) -> np.ndarray:
    projectors = np.zeros((t.size, 3, 3), dtype=float)
    dwidth = np.array([0.0, 1.0, 0.0])
    for i, ti in enumerate(t):
        dt = np.array(
            [
                (np.cos(ti) - ti * np.sin(ti)) / 8.0,
                0.0,
                (np.sin(ti) + ti * np.cos(ti)) / 8.0,
            ]
        )
        projectors[i] = projector_from_basis(dt, dwidth)
    return projectors


def half_sphere_projection(x: np.ndarray) -> ProjectionResult:
    x = np.asarray(x, dtype=float)
    xy_norm = np.linalg.norm(x[:, :2], axis=1)
    full_norm = np.linalg.norm(x, axis=1)
    projected = np.zeros_like(x)

    upper = x[:, 2] >= 0
    projected[upper] = x[upper] / (full_norm[upper, None] + EPS)

    boundary = ~upper
    projected[boundary, 0] = x[boundary, 0] / (xy_norm[boundary] + EPS)
    projected[boundary, 1] = x[boundary, 1] / (xy_norm[boundary] + EPS)
    projected[boundary, 2] = 0.0

    target = np.array([0.25, -0.1, 0.96], dtype=float)
    target = target / np.linalg.norm(target)
    normal = normalize_rows(projected)
    grad_ambient = -target[None, :]
    tangent_grad = grad_ambient - np.sum(grad_ambient * normal, axis=1, keepdims=True) * normal
    true_v = -normalize_rows(tangent_grad)

    projectors = np.zeros((x.shape[0], 3, 3), dtype=float)
    eye = np.eye(3)
    for i, nvec in enumerate(normal):
        projectors[i] = eye - np.outer(nvec, nvec)

    return ProjectionResult(
        projected_x=projected,
        true_v=true_v,
        projectors=projectors,
        distances=np.linalg.norm(x - projected, axis=1),
    )


def saddle_map(u: np.ndarray, w: np.ndarray) -> np.ndarray:
    y = 0.55 * (u**2 - w**2)
    return np.column_stack([u, y, w])


def saddle_source_velocity(u: np.ndarray, w: np.ndarray) -> np.ndarray:
    uv = np.column_stack([u, w])
    direction_uw = normalize_rows(uv)
    tangent_u = np.column_stack([np.ones_like(u), 1.10 * u, np.zeros_like(u)])
    tangent_w = np.column_stack([np.zeros_like(w), -1.10 * w, np.ones_like(w)])
    return direction_uw[:, [0]] * tangent_u + direction_uw[:, [1]] * tangent_w


def saddle_projectors(u: np.ndarray, w: np.ndarray) -> np.ndarray:
    projectors = np.zeros((u.size, 3, 3), dtype=float)
    for i, (ui, wi) in enumerate(zip(u, w)):
        tangent_u = np.array([1.0, 1.10 * ui, 0.0])
        tangent_w = np.array([0.0, -1.10 * wi, 1.0])
        projectors[i] = projector_from_basis(tangent_u, tangent_w)
    return projectors


def build_grid_projection(
    points: np.ndarray,
    true_v: np.ndarray,
    projectors: np.ndarray,
) -> ManifoldGrid:
    nn = NearestNeighbors(n_neighbors=1).fit(points)
    return ManifoldGrid(points=points, true_v=normalize_rows(true_v), projectors=projectors, nn=nn)


def project_with_grid(x: np.ndarray, grid: ManifoldGrid) -> ProjectionResult:
    distances, indices = grid.nn.kneighbors(x)
    idx = indices[:, 0]
    return ProjectionResult(
        projected_x=grid.points[idx],
        true_v=grid.true_v[idx],
        projectors=grid.projectors[idx],
        distances=distances[:, 0],
    )


def build_manifold_grids(grid_size: int) -> dict[str, ManifoldGrid]:
    grids: dict[str, ManifoldGrid] = {}

    t = np.linspace(-1.2 * np.pi, 1.2 * np.pi, grid_size)
    h = np.linspace(-S_CURVE_WIDTH, S_CURVE_WIDTH, max(grid_size // 2, 80))
    tt, hh = np.meshgrid(t, h, indexing="ij")
    t_flat, h_flat = tt.ravel(), hh.ravel()
    grids["s_curve_velocity_flow"] = build_grid_projection(
        s_curve_map(t_flat, h_flat),
        s_curve_velocity(t_flat, h_flat),
        s_curve_projectors(t_flat),
    )

    t = np.linspace(1.5 * np.pi, 4.5 * np.pi, grid_size)
    width = np.linspace(-SWISS_ROLL_WIDTH, SWISS_ROLL_WIDTH, max(grid_size // 2, 80))
    tt, ww = np.meshgrid(t, width, indexing="ij")
    t_flat, w_flat = tt.ravel(), ww.ravel()
    grids["swiss_roll_velocity_flow"] = build_grid_projection(
        swiss_roll_map(t_flat, w_flat),
        swiss_roll_velocity(t_flat, w_flat),
        swiss_roll_projectors(t_flat),
    )

    u_axis = np.linspace(-SADDLE_RADIUS, SADDLE_RADIUS, grid_size)
    w_axis = np.linspace(-SADDLE_RADIUS, SADDLE_RADIUS, grid_size)
    uu, ww = np.meshgrid(u_axis, w_axis, indexing="ij")
    keep = uu**2 + ww**2 <= SADDLE_RADIUS**2
    u_flat, w_flat = uu[keep], ww[keep]
    grids["saddle_surface_single_basin"] = build_grid_projection(
        saddle_map(u_flat, w_flat),
        saddle_source_velocity(u_flat, w_flat),
        saddle_projectors(u_flat, w_flat),
    )

    return grids


def project_to_true_manifold(name: str, x: np.ndarray, grids: dict[str, ManifoldGrid]) -> ProjectionResult:
    if name == "half_sphere_single_basin":
        return half_sphere_projection(x)
    if name in grids:
        return project_with_grid(x, grids[name])
    raise KeyError(f"no projection implementation for dataset {name!r}")


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing manifest: {path}")
    return json.loads(path.read_text())


def method_records_by_name(pipeline_dir: Path, method_dir: str) -> dict[str, dict]:
    manifest = load_manifest(pipeline_dir / method_dir / "manifest.json")
    return {record["name"]: record for record in manifest}


def load_npz(root: Path, record: dict) -> np.lib.npyio.NpzFile:
    return np.load(root / record["npz"])


def assert_matching_saved_inputs(method: str, dataset: str, method_data, simulated_data) -> None:
    checks = [("X_gt", "X_gt"), ("V_gt", "V_gt"), ("X_noisy", "X_noisy"), ("V_noisy", "V_noisy")]
    for method_key, simulated_key in checks:
        if method_key not in method_data.files:
            raise KeyError(f"{method} / {dataset} missing {method_key} in saved npz")
        left = method_data[method_key]
        right = simulated_data[simulated_key]
        if left.shape != right.shape or not np.allclose(left, right, rtol=1e-8, atol=1e-10):
            raise ValueError(
                f"{method} output for {dataset!r} is stale relative to simulated_data. "
                "Rerun notebooks 02-04 after rerunning notebook 01."
            )


def evaluate_state(name: str, x: np.ndarray, v: np.ndarray, grids: dict[str, ManifoldGrid]) -> dict[str, float]:
    projection = project_to_true_manifold(name, x, grids)
    v_unit = normalize_rows(v)
    v_tangent = normalize_rows(project_velocity_to_tangent(v_unit, projection.projectors))
    true_v = normalize_rows(projection.true_v)
    return {
        "manifold_distance_mean": float(np.mean(projection.distances)),
        "manifold_distance_rmse": float(np.sqrt(np.mean(projection.distances**2))),
        "projected_velocity_rmse": row_rmse(v_tangent, true_v),
        "projected_velocity_cosine": mean_cosine(v_tangent, true_v),
        "tangent_residual_mean": float(np.mean(tangent_residual(v, projection.projectors))),
    }


def compute_rows(pipeline_dir: Path, grid_size: int) -> tuple[pd.DataFrame, dict]:
    root = pipeline_dir.resolve().parents[1]
    grids = build_manifold_grids(grid_size)
    simulated_manifest = load_manifest(pipeline_dir / "simulated_data" / "manifest.json")
    method_manifests = {
        method: method_records_by_name(pipeline_dir, method_dir)
        for method, method_dir in METHOD_DIRS.items()
        if method != "Noisy"
    }

    rows = []
    method_parameters: dict[str, dict[str, object]] = {}

    for record in simulated_manifest:
        name = record["name"]
        sim_data = load_npz(root, record)
        noisy_metrics = evaluate_state(name, sim_data["X_noisy"], sim_data["V_noisy"], grids)

        rows.append(
            {
                "dataset_index": record["index"],
                "dataset": name,
                "family": record["family"],
                "kind": record["kind"],
                "method": "Noisy",
                **noisy_metrics,
                "manifold_distance_relative_rmse": 1.0 if noisy_metrics["manifold_distance_rmse"] > EPS else np.nan,
                "projected_velocity_relative_rmse": 1.0,
            }
        )

        for method, manifest_by_name in method_manifests.items():
            method_record = manifest_by_name[name]
            method_data = load_npz(root, method_record)
            assert_matching_saved_inputs(method, name, method_data, sim_data)
            metrics = evaluate_state(name, method_data["X_fit"], method_data["V_fit"], grids)
            rows.append(
                {
                    "dataset_index": record["index"],
                    "dataset": name,
                    "family": record["family"],
                    "kind": record["kind"],
                    "method": method,
                    **metrics,
                    "manifold_distance_relative_rmse": safe_relative(
                        metrics["manifold_distance_rmse"],
                        noisy_metrics["manifold_distance_rmse"],
                    ),
                    "projected_velocity_relative_rmse": safe_relative(
                        metrics["projected_velocity_rmse"],
                        noisy_metrics["projected_velocity_rmse"],
                    ),
                }
            )
            method_parameters.setdefault(method, {})
            method_parameters[method][name] = method_record.get("fit_kwargs", {})

    table = pd.DataFrame(rows)
    method_order = pd.CategoricalDtype(list(METHOD_DIRS.keys()), ordered=True)
    table["method"] = table["method"].astype(method_order)
    table = table.sort_values(["dataset_index", "method"]).reset_index(drop=True)
    ordered_columns = [
        "dataset_index",
        "dataset",
        "family",
        "kind",
        "method",
        "manifold_distance_rmse",
        "manifold_distance_relative_rmse",
        "projected_velocity_rmse",
        "projected_velocity_relative_rmse",
        "projected_velocity_cosine",
        "tangent_residual_mean",
        "manifold_distance_mean",
    ]
    return table[ordered_columns], method_parameters


def dataframe_to_markdown(table: pd.DataFrame) -> str:
    columns = list(table.columns)
    values = [[str(item) for item in row] for row in table.astype(object).to_numpy()]
    widths = []
    for col_idx, column in enumerate(columns):
        widths.append(max([len(column), *[len(row[col_idx]) for row in values]]))

    def format_row(items: list[str]) -> str:
        return "| " + " | ".join(item.ljust(width) for item, width in zip(items, widths)) + " |"

    return "\n".join(
        [
            format_row(columns),
            "| " + " | ".join("-" * width for width in widths) + " |",
            *[format_row(row) for row in values],
        ]
    )


def write_outputs(table: pd.DataFrame, method_parameters: dict, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "figure2_manifold_projection_metrics.csv"
    markdown_path = output_dir / "figure2_manifold_projection_metrics.md"
    params_path = output_dir / "figure2_manifold_projection_method_parameters.json"

    table.to_csv(csv_path, index=False)
    rounded = table.copy()
    metric_cols = [col for col in rounded.columns if col not in {"dataset", "family", "kind", "method"}]
    rounded[metric_cols] = rounded[metric_cols].round(4)
    markdown_path.write_text(dataframe_to_markdown(rounded) + "\n")
    params_path.write_text(json.dumps(method_parameters, indent=2) + "\n")
    return {"csv": csv_path, "markdown": markdown_path, "parameters": params_path}


def main() -> None:
    args = parse_args()
    pipeline_dir = args.pipeline_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else pipeline_dir / "metrics"
    table, method_parameters = compute_rows(pipeline_dir, args.grid_size)
    paths = write_outputs(table, method_parameters, output_dir)
    print(f"Wrote {len(table)} projection-metric rows")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
