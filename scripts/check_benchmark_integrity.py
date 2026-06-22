"""Quick integrity checks for the benchmark pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.geometry_velocity_metrics import (  # noqa: E402
    local_covariance_spectrum,
    normal_energy_ratio,
    velocity_neighbor_direction_agreement,
    velocity_smoothness,
    velocity_tangent_alignment,
)
from scripts.html_report_utils import write_html_report  # noqa: E402
from scripts.pca_denoisers import global_pca_denoise  # noqa: E402
from scripts.run_simulation_benchmark import run_benchmark  # noqa: E402


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_pca_denoiser() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 5))
    X_hat, info = global_pca_denoise(X, rank=3, return_info=True)
    assert_true(X_hat.shape == X.shape, "PCA denoiser changed shape")
    assert_true(info["rank"] == 3, "PCA rank diagnostic is wrong")

    X_full, _ = global_pca_denoise(X, rank=5, return_info=True)
    assert_true(np.allclose(X_full, X, atol=1e-10), "full-rank PCA did not reconstruct X")

    try:
        global_pca_denoise(X, rank=0)
    except ValueError:
        pass
    else:
        raise AssertionError("rank 0 should be disallowed")

    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    try:
        global_pca_denoise(X_bad, rank=2)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN input should be rejected")


def check_metrics() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(24, 6))
    V = rng.normal(size=(24, 6))
    zero_V = np.zeros_like(V)
    spectrum = local_covariance_spectrum(X, n_neighbors=5)
    assert_true(spectrum.shape == X.shape, "local spectrum has wrong shape")
    assert_true(np.isfinite(normal_energy_ratio(X, 2, n_neighbors=5)["mean"]), "normal energy ratio is not finite")
    assert_true(
        np.isfinite(velocity_tangent_alignment(X, V, 2, n_neighbors=5)["mean"]),
        "velocity-tangent alignment failed on random velocities",
    )
    zero_alignment = velocity_tangent_alignment(X, zero_V, 2, n_neighbors=5)
    assert_true(zero_alignment["zero_velocity_fraction"] == 1.0, "zero velocity fraction not detected")
    velocity_neighbor_direction_agreement(X, zero_V, n_neighbors=5)
    velocity_smoothness(X, zero_V, n_neighbors=5)


def check_noisy_plane_sanity() -> None:
    rng = np.random.default_rng(2)
    clean = rng.uniform(-1, 1, size=(160, 2))
    X_clean = np.hstack([clean, np.zeros((clean.shape[0], 4))])
    V = np.hstack([np.column_stack([-clean[:, 1], clean[:, 0]]), np.zeros((clean.shape[0], 4))])
    noisy = X_clean + rng.normal(scale=0.15, size=X_clean.shape)
    X_hat, _ = global_pca_denoise(noisy, rank=2, return_info=True)
    raw_rmse = float(np.sqrt(np.mean(np.sum((noisy - X_clean) ** 2, axis=1))))
    pca_rmse = float(np.sqrt(np.mean(np.sum((X_hat - X_clean) ** 2, axis=1))))
    assert_true(pca_rmse < raw_rmse, "PCA rank-d did not reduce plane noise")
    assert_true(
        normal_energy_ratio(X_hat, 2, n_neighbors=12)["mean"] < normal_energy_ratio(noisy, 2, n_neighbors=12)["mean"],
        "normal energy ratio did not decrease after PCA denoising",
    )
    assert_true(
        velocity_tangent_alignment(X_hat, V, 2, n_neighbors=12)["mean"] > 0.8,
        "velocity-tangent alignment is unexpectedly low on a clean plane velocity field",
    )


def check_html_report() -> None:
    with tempfile.TemporaryDirectory(prefix="manfitvelo_html_") as tmp:
        tmp_path = Path(tmp)
        image_path = tmp_path / "dummy.png"
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1], [0, 1])
        fig.savefig(image_path)
        plt.close(fig)
        report = write_html_report(
            "Dummy Report",
            [{"heading": "Section", "text": "Smoke test.", "images": [image_path]}],
            tmp_path / "index.html",
        )
        assert_true(report.exists(), "dummy HTML report was not written")
        assert_true("Dummy Report" in report.read_text(encoding="utf-8"), "dummy HTML report content missing")


def check_tiny_simulation_benchmark() -> None:
    with tempfile.TemporaryDirectory(prefix="manfitvelo_sim_") as tmp:
        args = argparse.Namespace(
            datasets="flat_rotation",
            n_seeds=1,
            n_samples=32,
            position_noise=0.12,
            velocity_noise=0.12,
            extra_dims=4,
            seed=3,
            n_neighbors=8,
            fit_neighbors=8,
            fit_iterations=2,
            eta_g=0.25,
            theta=0.1,
            kappa=1.0,
            include_local_pca=False,
            include_position_manfit=False,
            position_manfit_max_n=0,
            output_dir=Path(tmp) / "simulation_benchmark",
        )
        result = run_benchmark(args)
        assert_true(Path(result["report_path"]).exists(), "tiny simulation report was not written")
        metrics = result["metrics"]
        assert_true(not metrics.empty, "tiny simulation metrics are empty")
        assert_true("velocity_manifold_fitter" in set(metrics["method"]), "VMF method row missing")


def main() -> None:
    checks = [
        check_pca_denoiser,
        check_metrics,
        check_noisy_plane_sanity,
        check_html_report,
        check_tiny_simulation_benchmark,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print("All benchmark integrity checks passed.")


if __name__ == "__main__":
    main()
