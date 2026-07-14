"""Build an HTML report for Figure 2 visualizations and main metrics."""

from __future__ import annotations

import argparse
import base64
import html
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_figure2_vector_fields_1x4 import (
    METHODS,
    configure_matplotlib,
    default_output_base,
    render_panel,
)


DEFAULT_PIPELINE_DIR = PROJECT_ROOT / "outputs" / "figure2_pipeline"
DEFAULT_OUTPUT = DEFAULT_PIPELINE_DIR / "figure2_visual_metrics_report.html"


PANEL_FILES = {
    "Ground Truth": "figure2_ground_truth_vector_fields_1x4_highres.pdf",
    "Noisy": "figure2_noisy_vector_fields_1x4_highres.pdf",
    "ManfitVelo": "figure2_manfitvelo_vector_fields_1x4_highres.pdf",
    "Position-only MANFIT": "figure2_position_only_manfit_vector_fields_1x4_highres.pdf",
    "Local PCA": "figure2_local_pca_vector_fields_1x4_highres.pdf",
}

BASELINE_METHOD = "Noisy"
METRIC_EXPLANATIONS = {
    "position_rmse_rel": "Pointwise position RMSE relative to the noisy baseline.",
    "manifold_distance_rel": "Distance to the ground-truth manifold relative to the noisy baseline.",
    "velocity_rmse_rel": "Pointwise velocity RMSE relative to the noisy baseline.",
    "velocity_angle_mae_rel": "Mean velocity direction-angle error relative to the noisy baseline.",
    "potential_rmse_rel": "Affine-aligned potential RMSE relative to the noisy baseline, shown only for potential-field datasets.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, default=DEFAULT_PIPELINE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600, help="DPI for embedded PNG panels.")
    parser.add_argument(
        "--reuse-assets",
        action="store_true",
        help="Reuse existing high-resolution PNG assets instead of rendering them again.",
    )
    return parser.parse_args()


def render_highres_panels(pipeline_dir: Path, assets_dir: Path, dpi: int) -> dict[str, Path]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib(dpi)

    paths = {}
    ground_truth_base = assets_dir / "figure2_ground_truth_vector_fields_1x4_highres"
    render_panel(
        pipeline_dir / "simulated_data" / "manifest.json",
        ground_truth_base,
        dpi=dpi,
        layer="gt",
        formats=("png",),
    )
    paths["Ground Truth"] = ground_truth_base.with_suffix(".png")

    noisy_base = assets_dir / "figure2_noisy_vector_fields_1x4_highres"
    render_panel(
        pipeline_dir / "simulated_data" / "manifest.json",
        noisy_base,
        dpi=dpi,
        layer="noisy",
        formats=("png",),
    )
    paths["Noisy"] = noisy_base.with_suffix(".png")

    labels = {
        "manfitvelo": "ManfitVelo",
        "position_only_manfit": "Position-only MANFIT",
        "local_pca": "Local PCA",
    }
    for method in METHODS:
        output_base = assets_dir / default_output_base(pipeline_dir, method).name
        render_panel(
            pipeline_dir / method / "manifest.json",
            output_base,
            dpi=dpi,
            layer="fit",
            formats=("png",),
        )
        paths[labels[method]] = output_base.with_suffix(".png")
    return paths


def existing_panel_paths(pipeline_dir: Path) -> dict[str, Path]:
    assets_dir = pipeline_dir / "html_assets"
    labels = {
        "Ground Truth": "figure2_ground_truth_vector_fields_1x4_highres.png",
        "Noisy": "figure2_noisy_vector_fields_1x4_highres.png",
        "ManfitVelo": "figure2_manfitvelo_vector_fields_1x4_highres.png",
        "Position-only MANFIT": "figure2_position_only_manfit_vector_fields_1x4_highres.png",
        "Local PCA": "figure2_local_pca_vector_fields_1x4_highres.png",
    }
    paths = {label: assets_dir / filename for label, filename in labels.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing reusable HTML assets: " + ", ".join(missing))
    return paths


def image_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def format_value(value: object) -> str:
    if pd.isna(value):
        return "&mdash;"
    if isinstance(value, float):
        return f"{value:.4g}"
    return html.escape(str(value))


def metric_best_values(group: pd.DataFrame, metric_cols: list[str]) -> dict[str, float]:
    fitted = group[group["method"] != BASELINE_METHOD]
    if fitted.empty:
        fitted = group
    return {col: fitted[col].min(skipna=True) for col in metric_cols if fitted[col].notna().any()}


def dataset_name(value: str) -> str:
    return value.replace("_", " ").title()


def dataframe_to_html(table: pd.DataFrame) -> str:
    metric_cols = [col for col in table.columns if col not in {"dataset", "method"}]
    headers = "".join(f"<th>{html.escape(str(col))}</th>" for col in ["method", *metric_cols])
    rows = []
    for dataset, group in table.groupby("dataset", sort=False):
        rows.append(
            '<tr class="dataset-row">'
            f'<th colspan="{len(metric_cols) + 1}">{html.escape(dataset_name(str(dataset)))}</th>'
            "</tr>"
        )
        best_values = metric_best_values(group, metric_cols)
        for _, row in group.iterrows():
            cells = []
            for col in ["method", *metric_cols]:
                value = row[col]
                cls = "metric" if col != "method" else "method"
                is_best = (
                    col in best_values
                    and row["method"] != BASELINE_METHOD
                    and abs(float(value) - float(best_values[col])) < 1e-12
                )
                content = format_value(value)
                if is_best:
                    content = f"<strong>{content}</strong>"
                    cls += " best"
                cells.append(f'<td class="{cls}">{content}</td>')
            rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def flat_dataframe_to_html(table: pd.DataFrame) -> str:
    headers = "".join(f"<th>{html.escape(str(col))}</th>" for col in table.columns)
    rows = []
    for _, row in table.iterrows():
        cells = []
        for col in table.columns:
            value = row[col]
            cls = "metric" if col not in {"dataset", "method"} else ""
            cells.append(f'<td class="{cls}">{format_value(value)}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def metric_explanations_to_html(metric_cols: list[str]) -> str:
    items = []
    for col in metric_cols:
        explanation = METRIC_EXPLANATIONS.get(col)
        if explanation is None:
            continue
        items.append(
            f"<li><strong>{html.escape(col)}</strong>: {html.escape(explanation)}</li>"
        )
    return f"<ul class=\"metric-notes\">{''.join(items)}</ul>"


def build_html(pipeline_dir: Path, output_path: Path, dpi: int, reuse_assets: bool) -> None:
    assets_dir = pipeline_dir / "html_assets"
    panel_pngs = existing_panel_paths(pipeline_dir) if reuse_assets else render_highres_panels(pipeline_dir, assets_dir, dpi=dpi)
    image_blocks = []
    for label in PANEL_FILES:
        png_path = panel_pngs[label]
        image_blocks.append(
            f"""
            <section class="method-panel">
              <h2>{html.escape(label)}</h2>
              <img src="{image_data_uri(png_path)}" alt="{html.escape(label)} vector field panel">
            </section>
            """
        )

    metrics_path = pipeline_dir / "metrics" / "figure2_main_metrics.csv"
    table = pd.read_csv(metrics_path)
    metric_cols = [col for col in table.columns if col not in {"dataset", "method"}]
    table[metric_cols] = table[metric_cols].round(4)

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Figure 2 Visualizations and Metrics</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #151515;
      --muted: #5f6368;
      --line: #d8d8d2;
      --soft: #ecebe6;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }}
    main {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 18px;
      font-size: 22px;
      font-weight: 500;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 15px;
      font-weight: 500;
    }}
    .grid {{
      display: grid;
      gap: 12px;
    }}
    .method-panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
    }}
    .method-panel img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .metrics {{
      margin-top: 18px;
      overflow-x: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
    }}
    .metric-notes {{
      display: grid;
      grid-template-columns: repeat(2, minmax(280px, 1fr));
      gap: 4px 18px;
      margin: 0 0 12px;
      padding-left: 18px;
      color: var(--muted);
      font-size: 12px;
    }}
    .metric-notes strong {{
      color: var(--ink);
      font-weight: 500;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 980px;
    }}
    th, td {{
      border-bottom: 1px solid var(--soft);
      padding: 7px 8px;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-weight: 500;
      background: #fbfbf9;
    }}
    td.metric {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    td.method {{
      font-weight: 500;
    }}
    .dataset-row th {{
      padding-top: 13px;
      color: var(--ink);
      background: #f0f0eb;
      font-weight: 600;
      text-align: left;
    }}
    .best strong {{
      font-weight: 700;
    }}
    tbody tr:hover {{
      background: #f2f2ee;
    }}
    @media (max-width: 760px) {{
      main {{ padding: 14px; }}
      h1 {{ font-size: 18px; }}
      .method-panel {{ padding: 8px; }}
      .metric-notes {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Figure 2 Visualizations and Relative Metrics</h1>
    <div class="grid">
      {''.join(image_blocks)}
    </div>
    <section class="metrics">
      <h2>Main Metrics</h2>
      {metric_explanations_to_html(metric_cols)}
      {dataframe_to_html(table)}
    </section>
  </main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document)


def main() -> None:
    args = parse_args()
    build_html(args.pipeline_dir.resolve(), args.output.resolve(), dpi=args.dpi, reuse_assets=args.reuse_assets)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
