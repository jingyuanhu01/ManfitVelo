"""Render Figure 2 vector-field examples as high-resolution 1x4 panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from scripts.potential_from_gradient import potential_from_gradient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE_DIR = PROJECT_ROOT / "outputs" / "figure2_pipeline"
DEFAULT_OUTPUT_BASE = DEFAULT_PIPELINE_DIR / "figure2_vector_fields_1x4_highres"
METHODS = ("manfitvelo", "position_only_manfit", "local_pca")
POTENTIAL_DERIVED_SIGN = -1.0
POTENTIAL_DERIVED_K = 25
POTENTIAL_DERIVED_CELL_CONSISTENCY = 0.1
POTENTIAL_DERIVED_LAPLACIAN = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=DEFAULT_PIPELINE_DIR,
        help="Figure 2 pipeline directory containing simulated_data/manifest.json.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=None,
        help="Output path without extension.",
    )
    parser.add_argument(
        "--method",
        choices=("ground_truth", "noisy", "all", *METHODS),
        default="ground_truth",
        help="Panel source. Method panels use X_fit/V_fit.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG export DPI.")
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
        help="Output formats to write.",
    )
    return parser.parse_args()


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.min()) / (values.max() - values.min() + 1e-12)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-12)


def equalize_limits_3d(ax, x_mat: np.ndarray, pad: float = -0.18) -> None:
    mins = x_mat.min(axis=0)
    maxs = x_mat.max(axis=0)
    centers = 0.5 * (mins + maxs)
    span = np.max(maxs - mins) * (1 + pad)
    ax.set_xlim(centers[0] - 0.5 * span, centers[0] + 0.5 * span)
    ax.set_ylim(centers[1] - 0.5 * span, centers[1] + 0.5 * span)
    ax.set_zlim(centers[2] - 0.5 * span, centers[2] + 0.5 * span)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def style_3d_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0)
        axis.pane.set_edgecolor("white")
        axis.line.set_color((1, 1, 1, 0))
    ax.set_facecolor("white")


def draw_half_sphere_surface(ax, x_mat: np.ndarray, color: str = "#cbd5e1") -> None:
    theta = np.linspace(0, 2 * np.pi, 96)
    rho = np.linspace(0, 1.0, 36)
    rr, tt = np.meshgrid(rho, theta, indexing="ij")
    u = rr * np.cos(tt)
    v = rr * np.sin(tt)
    z = np.sqrt(np.maximum(1.0 - rr**2, 0.0))
    center = np.asarray(x_mat, dtype=float).mean(axis=0)
    ax.plot_surface(
        u - center[0],
        v - center[1],
        z - center[2],
        color=color,
        alpha=0.42,
        linewidth=0,
        antialiased=True,
        shade=False,
        zorder=1,
    )


def draw_saddle_chip_surface(ax, x_mat: np.ndarray, color: str = "#cbd5e1") -> None:
    theta = np.linspace(0, 2 * np.pi, 96)
    rho = np.linspace(0, 1.12, 36)
    rr, tt = np.meshgrid(rho, theta, indexing="ij")
    u = rr * np.cos(tt)
    w = rr * np.sin(tt)
    y = 0.55 * (u**2 - w**2)
    center = np.asarray(x_mat, dtype=float).mean(axis=0)
    ax.plot_surface(
        u - center[0],
        y - center[1],
        w - center[2],
        color=color,
        alpha=0.48,
        linewidth=0,
        antialiased=True,
        shade=False,
        zorder=1,
    )


def visible_half_sphere_arrow_indices(record: dict, x_mat: np.ndarray, arrow_idx: np.ndarray) -> np.ndarray:
    if record["name"] != "half_sphere_single_basin" or arrow_idx.size == 0:
        return arrow_idx

    view = record.get("view") or {}
    azim = np.deg2rad(float(view.get("azim", -58)))
    camera_xy = np.array([np.cos(azim), np.sin(azim)])
    centered_xy = x_mat[:, :2] - x_mat[:, :2].mean(axis=0, keepdims=True)
    depth = centered_xy @ camera_xy
    return arrow_idx[depth[arrow_idx] >= np.median(depth)]


def sparse_swiss_roll_arrow_indices(record: dict, arrow_idx: np.ndarray, max_arrows: int = 24) -> np.ndarray:
    if record["name"] != "swiss_roll_velocity_flow" or arrow_idx.size <= max_arrows:
        return arrow_idx
    keep = np.linspace(0, arrow_idx.size - 1, max_arrows).round().astype(int)
    return arrow_idx[keep]


def maybe_use_potential_derived_field(
    record: dict,
    data: np.lib.npyio.NpzFile,
    layer: str,
    x_mat: np.ndarray,
    v_mat: np.ndarray,
    color: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if layer != "fit" or record.get("family") != "Potential":
        return v_mat, color

    learned = potential_from_gradient(
        x_mat,
        v_mat,
        sign=POTENTIAL_DERIVED_SIGN,
        n_neighbors=POTENTIAL_DERIVED_K,
        cell_consistency_reg=POTENTIAL_DERIVED_CELL_CONSISTENCY,
        laplacian_reg=POTENTIAL_DERIVED_LAPLACIAN,
    )
    velocity = POTENTIAL_DERIVED_SIGN * np.asarray(learned["gradient"], dtype=float)
    learned_color = np.asarray(learned["potential"], dtype=float)
    if "color_gt" in data.files and np.std(learned_color) > 0:
        reference = np.asarray(data["color_gt"], dtype=float)
        if np.std(reference) > 0 and np.corrcoef(learned_color, reference)[0, 1] < 0:
            learned_color = -learned_color
    return velocity, normalize01(learned_color)


def plot_panel(ax, record: dict, data: np.lib.npyio.NpzFile, layer: str = "gt") -> None:
    record = dict(record)

    layer_to_suffix = {"gt": "gt", "noisy": "noisy", "fit": "fit"}
    suffix = layer_to_suffix[layer]
    x_mat = np.asarray(data[f"X_{suffix}"], dtype=float)
    v_mat = normalize_rows(data[f"V_{suffix}"])
    color = normalize01(data[f"color_{suffix}"])
    v_mat, color = maybe_use_potential_derived_field(record, data, layer, x_mat, v_mat, color)
    v_mat = normalize_rows(v_mat)
    if record["name"] == "half_sphere_single_basin":
        color = 0.18 + 0.82 * color

    x_plot = x_mat - x_mat.mean(axis=0)
    cmap = "viridis" if record["family"] == "Velocity" else "magma_r"

    if record["name"] == "half_sphere_single_basin":
        draw_half_sphere_surface(ax, x_mat)
    elif record["name"] == "saddle_surface_single_basin":
        draw_saddle_chip_surface(ax, x_mat)

    ax.scatter(
        x_plot[:, 0],
        x_plot[:, 1],
        x_plot[:, 2],
        c=color,
        cmap=cmap,
        vmin=0,
        vmax=1,
        s=15,
        alpha=0.80,
        linewidths=0,
        rasterized=True,
        depthshade=False,
        zorder=10,
    )

    arrow_idx = np.asarray(data["arrow_indices"], dtype=int)
    arrow_idx = visible_half_sphere_arrow_indices(record, x_mat, arrow_idx)
    arrow_idx = sparse_swiss_roll_arrow_indices(record, arrow_idx)

    span = np.max(np.ptp(x_plot, axis=0))
    arrow_length = 0.18 * span
    ax.quiver(
        x_plot[arrow_idx, 0],
        x_plot[arrow_idx, 1],
        x_plot[arrow_idx, 2],
        v_mat[arrow_idx, 0],
        v_mat[arrow_idx, 1],
        v_mat[arrow_idx, 2],
        length=arrow_length,
        normalize=True,
        pivot="middle",
        color="#050505",
        alpha=0.98,
        linewidth=1.45,
        arrow_length_ratio=0.48,
        zorder=20,
    )

    if record["name"] == "swiss_roll_velocity_flow":
        limit_pad = -0.36
    elif record["family"] == "Potential":
        limit_pad = -0.34
    else:
        limit_pad = -0.18
    equalize_limits_3d(ax, x_plot, pad=limit_pad)
    view = record.get("view") or {"elev": 22, "azim": -58, "roll": 0}
    try:
        ax.view_init(elev=view["elev"], azim=view["azim"], roll=view.get("roll", 0))
    except TypeError:
        ax.view_init(elev=view["elev"], azim=view["azim"])
    style_3d_axis(ax)


def configure_matplotlib(dpi: int) -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": dpi,
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def render_panel(
    manifest_path: Path,
    output_base: Path,
    dpi: int,
    layer: str,
    formats: tuple[str, ...] | list[str],
) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())

    fig = plt.figure(figsize=(12.4, 2.35))
    axes = []
    for col, record in enumerate(manifest, start=1):
        ax = fig.add_subplot(1, len(manifest), col, projection="3d", computed_zorder=False)
        data = np.load(PROJECT_ROOT / record["npz"])
        plot_panel(ax, record, data, layer=layer)
        axes.append(ax)

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.98, wspace=-0.18)

    if "png" in formats:
        png_path = output_base.with_suffix(".png")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        print(png_path)
    if "pdf" in formats:
        pdf_path = output_base.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(pdf_path)
    plt.close(fig)


def default_output_base(pipeline_dir: Path, method: str) -> Path:
    if method == "ground_truth":
        return DEFAULT_OUTPUT_BASE
    if method == "noisy":
        return pipeline_dir / "figure2_noisy_vector_fields_1x4_highres"
    return pipeline_dir / f"figure2_{method}_vector_fields_1x4_highres"


def main() -> None:
    args = parse_args()
    pipeline_dir = args.pipeline_dir.resolve()
    configure_matplotlib(args.dpi)

    if args.method == "all":
        for method in METHODS:
            render_panel(
                pipeline_dir / method / "manifest.json",
                default_output_base(pipeline_dir, method),
                dpi=args.dpi,
                layer="fit",
                formats=args.formats,
            )
        return

    output_base = (args.output_base or default_output_base(pipeline_dir, args.method)).resolve()
    if args.method in {"ground_truth", "noisy"}:
        manifest_path = pipeline_dir / "simulated_data" / "manifest.json"
        layer = "gt" if args.method == "ground_truth" else "noisy"
    else:
        manifest_path = pipeline_dir / args.method / "manifest.json"
        layer = "fit"
    render_panel(manifest_path, output_base, dpi=args.dpi, layer=layer, formats=args.formats)


if __name__ == "__main__":
    main()
