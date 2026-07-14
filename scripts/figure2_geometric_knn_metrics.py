"""Compute Figure 2 local geometry and velocity-angle preservation metrics.

The current Figure 2 set contains four three-dimensional simulations: two
velocity-flow manifolds and two scalar-potential manifolds.

These metrics avoid direct analytic pointwise velocity comparison. Instead,
they ask whether the fitted/noisy result preserves local geometry and local
vector-field structure for the same simulated cells.

Metrics:
  - knn_overlap: mean overlap between each cell's kNN set in X_gt and X_method.
  - pairwise_distance_error_rel: mean relative error of local edge lengths on
    ground-truth kNN edges, normalized by the ground-truth edge length.
  - velocity_angle_mae: mean absolute error of pairwise velocity cosines on
    ground-truth kNN edges.
  - velocity_angle_corr: Pearson correlation of pairwise velocity cosines on
    ground-truth kNN edges.

The velocity-angle metric is deliberately local/topological: it compares
cos(v_i, v_j) over nearby cell pairs, rather than comparing v_i to an analytic
velocity at a projected location.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


EPS = 1e-12


METHOD_DIRS = {
    "Noisy": "simulated_data",
    "ManfitVelo": "manfitvelo",
    "Position-only MANFIT": "position_only_manfit",
    "Local PCA": "local_pca",
}


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
        "--n-neighbors",
        type=int,
        default=25,
        help="Number of neighbors for local geometry/velocity-angle edges.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing manifest: {path}")
    return json.loads(path.read_text())


def load_npz(root: Path, record: dict) -> np.lib.npyio.NpzFile:
    return np.load(root / record["npz"])


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + EPS)


def safe_relative(value: float, baseline: float) -> float:
    if not np.isfinite(baseline) or abs(baseline) <= EPS:
        return float("nan")
    return float(value / baseline)


def method_records_by_name(pipeline_dir: Path, method_dir: str) -> dict[str, dict]:
    manifest = load_manifest(pipeline_dir / method_dir / "manifest.json")
    return {record["name"]: record for record in manifest}


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


def knn_indices(x: np.ndarray, n_neighbors: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    k = min(max(int(n_neighbors), 1), x.shape[0] - 1)
    indices = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x, return_distance=False)
    return indices[:, 1:]


def knn_overlap(x_reference: np.ndarray, x_candidate: np.ndarray, n_neighbors: int) -> float:
    ref_idx = knn_indices(x_reference, n_neighbors)
    cand_idx = knn_indices(x_candidate, n_neighbors)
    overlaps = []
    for left, right in zip(ref_idx, cand_idx):
        overlaps.append(len(set(left).intersection(set(right))) / float(ref_idx.shape[1]))
    return float(np.mean(overlaps))


def undirected_edges(indices: np.ndarray) -> np.ndarray:
    edges = set()
    for i, neigh in enumerate(indices):
        for j in neigh:
            a, b = sorted((int(i), int(j)))
            if a != b:
                edges.add((a, b))
    return np.asarray(sorted(edges), dtype=int)


def edge_velocity_cosines(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    v = unit_rows(v)
    return np.sum(v[edges[:, 0]] * v[edges[:, 1]], axis=1)


def pairwise_distance_error_rel(x_reference: np.ndarray, x_candidate: np.ndarray, edges: np.ndarray) -> float:
    ref_dist = np.linalg.norm(x_reference[edges[:, 0]] - x_reference[edges[:, 1]], axis=1)
    cand_dist = np.linalg.norm(x_candidate[edges[:, 0]] - x_candidate[edges[:, 1]], axis=1)
    return float(np.mean(np.abs(cand_dist - ref_dist) / (ref_dist + EPS)))


def pearson_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if left.size < 2 or np.std(left) <= EPS or np.std(right) <= EPS:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def velocity_angle_metrics(v_reference: np.ndarray, v_candidate: np.ndarray, edges: np.ndarray) -> dict[str, float]:
    ref_cos = edge_velocity_cosines(v_reference, edges)
    cand_cos = edge_velocity_cosines(v_candidate, edges)
    return {
        "velocity_angle_mae": float(np.mean(np.abs(cand_cos - ref_cos))),
        "velocity_angle_rmse": float(np.sqrt(np.mean((cand_cos - ref_cos) ** 2))),
        "velocity_angle_corr": pearson_corr(ref_cos, cand_cos),
    }


def evaluate_state(
    x_gt: np.ndarray,
    v_gt: np.ndarray,
    x_candidate: np.ndarray,
    v_candidate: np.ndarray,
    n_neighbors: int,
) -> dict[str, float]:
    gt_neighbors = knn_indices(x_gt, n_neighbors)
    edges = undirected_edges(gt_neighbors)
    metrics = {
        "knn_overlap": knn_overlap(x_gt, x_candidate, n_neighbors),
        "pairwise_distance_error_rel": pairwise_distance_error_rel(x_gt, x_candidate, edges),
        "n_edges": int(edges.shape[0]),
    }
    metrics.update(velocity_angle_metrics(v_gt, v_candidate, edges))
    return metrics


def compute_rows(pipeline_dir: Path, n_neighbors: int) -> tuple[pd.DataFrame, dict]:
    root = pipeline_dir.resolve().parents[1]
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
        x_gt = sim_data["X_gt"]
        v_gt = sim_data["V_gt"]

        noisy_metrics = evaluate_state(x_gt, v_gt, sim_data["X_noisy"], sim_data["V_noisy"], n_neighbors)
        rows.append(
            {
                "dataset_index": record["index"],
                "dataset": name,
                "family": record["family"],
                "kind": record["kind"],
                "method": "Noisy",
                **noisy_metrics,
                "knn_overlap_relative": 1.0,
                "velocity_angle_mae_relative": 1.0,
            }
        )

        for method, manifest_by_name in method_manifests.items():
            method_record = manifest_by_name[name]
            method_data = load_npz(root, method_record)
            assert_matching_saved_inputs(method, name, method_data, sim_data)
            metrics = evaluate_state(x_gt, v_gt, method_data["X_fit"], method_data["V_fit"], n_neighbors)
            rows.append(
                {
                    "dataset_index": record["index"],
                    "dataset": name,
                    "family": record["family"],
                    "kind": record["kind"],
                    "method": method,
                    **metrics,
                    "knn_overlap_relative": safe_relative(metrics["knn_overlap"], noisy_metrics["knn_overlap"]),
                    "velocity_angle_mae_relative": safe_relative(
                        metrics["velocity_angle_mae"],
                        noisy_metrics["velocity_angle_mae"],
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
        "knn_overlap",
        "knn_overlap_relative",
        "pairwise_distance_error_rel",
        "velocity_angle_mae",
        "velocity_angle_mae_relative",
        "velocity_angle_rmse",
        "velocity_angle_corr",
        "n_edges",
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
    csv_path = output_dir / "figure2_geometric_knn_metrics.csv"
    markdown_path = output_dir / "figure2_geometric_knn_metrics.md"
    params_path = output_dir / "figure2_geometric_knn_method_parameters.json"

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
    table, method_parameters = compute_rows(pipeline_dir, args.n_neighbors)
    paths = write_outputs(table, method_parameters, output_dir)
    print(f"Wrote {len(table)} geometric-kNN metric rows")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
