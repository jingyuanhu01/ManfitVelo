"""Compute Figure 2 simulation reconstruction metrics.

The script reads the saved outputs from the Figure 2 notebook pipeline:

    outputs/figure2_pipeline/simulated_data/
    outputs/figure2_pipeline/manfitvelo/
    outputs/figure2_pipeline/position_only_manfit/
    outputs/figure2_pipeline/local_pca/

It writes a 16-row table: 4 three-dimensional simulations x
(noisy + 3 benchmark methods). The current Figure 2 set contains two
velocity-flow examples and two scalar-potential examples.
Velocity RMSE is computed after row-normalizing velocity vectors, matching the
directional interpretation used in the quiver plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


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
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing manifest: {path}")
    return json.loads(path.read_text())


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + EPS)


def rmse(values: np.ndarray, truth: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if values.shape != truth.shape:
        raise ValueError(f"shape mismatch: {values.shape} versus {truth.shape}")
    return float(np.sqrt(np.mean(np.sum((values - truth) ** 2, axis=1))))


def total_global_variance(values: np.ndarray) -> float:
    """Return the trace of the population covariance matrix."""
    values = np.asarray(values, dtype=float)
    return float(np.sum(np.var(values, axis=0, ddof=0)))


def assert_matching_saved_inputs(method: str, dataset: str, method_data, simulated_data) -> None:
    """Fail fast if method outputs were produced from stale simulated data."""
    checks = [
        ("X_gt", "X_gt"),
        ("V_gt", "V_gt"),
        ("X_noisy", "X_noisy"),
        ("V_noisy", "V_noisy"),
    ]
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


def safe_relative(value: float, baseline: float) -> float:
    if not np.isfinite(baseline) or abs(baseline) <= EPS:
        return float("nan")
    return float(value / baseline)


def method_records_by_name(pipeline_dir: Path, method_dir: str) -> dict[str, dict]:
    manifest = load_manifest(pipeline_dir / method_dir / "manifest.json")
    return {record["name"]: record for record in manifest}


def load_npz(root: Path, record: dict) -> np.lib.npyio.NpzFile:
    return np.load(root / record["npz"])


def compute_rows(pipeline_dir: Path) -> tuple[pd.DataFrame, dict]:
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
        v_gt = unit_rows(sim_data["V_gt"])
        x_noisy = sim_data["X_noisy"]
        v_noisy = unit_rows(sim_data["V_noisy"])

        noisy_position_rmse = rmse(x_noisy, x_gt)
        noisy_velocity_rmse = rmse(v_noisy, v_gt)
        clean_global_variance = total_global_variance(x_gt)
        noisy_global_variance = total_global_variance(x_noisy)

        rows.append(
            {
                "dataset_index": record["index"],
                "dataset": name,
                "family": record["family"],
                "kind": record["kind"],
                "method": "Noisy",
                "position_rmse": noisy_position_rmse,
                "position_relative_rmse": 1.0,
                "global_variance_ratio": safe_relative(
                    noisy_global_variance,
                    clean_global_variance,
                ),
                "velocity_rmse": noisy_velocity_rmse,
                "velocity_relative_rmse": 1.0,
            }
        )

        for method, manifest_by_name in method_manifests.items():
            if name not in manifest_by_name:
                raise KeyError(f"{method} manifest has no record for dataset {name!r}")
            method_record = manifest_by_name[name]
            method_data = load_npz(root, method_record)
            assert_matching_saved_inputs(method, name, method_data, sim_data)
            x_fit = method_data["X_fit"]
            v_fit = unit_rows(method_data["V_fit"])

            position_rmse = rmse(x_fit, x_gt)
            velocity_rmse = rmse(v_fit, v_gt)
            fitted_global_variance = total_global_variance(x_fit)
            rows.append(
                {
                    "dataset_index": record["index"],
                    "dataset": name,
                    "family": record["family"],
                    "kind": record["kind"],
                    "method": method,
                    "position_rmse": position_rmse,
                    "position_relative_rmse": safe_relative(position_rmse, noisy_position_rmse),
                    "global_variance_ratio": safe_relative(
                        fitted_global_variance,
                        clean_global_variance,
                    ),
                    "velocity_rmse": velocity_rmse,
                    "velocity_relative_rmse": safe_relative(velocity_rmse, noisy_velocity_rmse),
                }
            )

            method_parameters.setdefault(method, {})
            method_parameters[method][name] = method_record.get("fit_kwargs", {})

    table = pd.DataFrame(rows)
    method_order = pd.CategoricalDtype(list(METHOD_DIRS.keys()), ordered=True)
    table["method"] = table["method"].astype(method_order)
    table = table.sort_values(["dataset_index", "method"]).reset_index(drop=True)
    return table, method_parameters


def write_outputs(table: pd.DataFrame, method_parameters: dict, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "figure2_reconstruction_metrics.csv"
    markdown_path = output_dir / "figure2_reconstruction_metrics.md"
    params_path = output_dir / "figure2_method_parameters.json"

    table.to_csv(csv_path, index=False)
    rounded = table.copy()
    metric_cols = [
        "position_rmse",
        "position_relative_rmse",
        "global_variance_ratio",
        "velocity_rmse",
        "velocity_relative_rmse",
    ]
    rounded[metric_cols] = rounded[metric_cols].round(4)
    markdown_path.write_text(dataframe_to_markdown(rounded) + "\n")
    params_path.write_text(json.dumps(method_parameters, indent=2) + "\n")

    return {"csv": csv_path, "markdown": markdown_path, "parameters": params_path}


def dataframe_to_markdown(table: pd.DataFrame) -> str:
    """Return a GitHub-flavored Markdown table without optional dependencies."""
    columns = list(table.columns)
    values = [[str(item) for item in row] for row in table.astype(object).to_numpy()]
    widths = []
    for col_idx, column in enumerate(columns):
        cell_widths = [len(row[col_idx]) for row in values]
        widths.append(max([len(column), *cell_widths]))

    def format_row(items: list[str]) -> str:
        cells = [item.ljust(width) for item, width in zip(items, widths)]
        return "| " + " | ".join(cells) + " |"

    header = format_row(columns)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [format_row(row) for row in values]
    return "\n".join([header, separator, *body])


def main() -> None:
    args = parse_args()
    pipeline_dir = args.pipeline_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else pipeline_dir / "metrics"

    table, method_parameters = compute_rows(pipeline_dir)
    paths = write_outputs(table, method_parameters, output_dir)

    print(f"Wrote {len(table)} metric rows")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
