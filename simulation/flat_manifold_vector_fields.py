"""Flat manifold vector-field data generation.

This module lifts the old 2D vector-field notebook block into reusable
generation functions. It does not run manifold fitting.

Names follow the pattern:

    flat_manifold__<field>_vector_field
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


FIELD_NAMES = (
    "rotation",
    "spiral",
    "saddle",
    "quadratic_source_sink",
    "radial_source",
    "radial_sink",
    "horizontal_shear",
    "sinusoidal",
)

FIELD_LABELS = {
    field_name: f"flat manifold, {field_name.replace('_', ' ')} field"
    for field_name in FIELD_NAMES
}


@dataclass(frozen=True)
class FlatVectorFieldConfig:
    """Config for flat manifold vector-field data generation."""

    field_name: str = "rotation"
    n_samples: int = 1000
    position_noise: float = 0.3
    velocity_noise: float = 0.3
    extra_dims: int = 5
    seed: int = 42

    @property
    def simulation_name(self) -> str:
        return f"flat_manifold__{self.field_name}_vector_field"

    def to_dict(self) -> dict[str, object]:
        return asdict(self) | {"simulation_name": self.simulation_name}


def scale_to_unit_box(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Scale 2D coordinates into a unit box while preserving aspect ratio."""

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    x_range = x_max - x_min
    y_range = y_max - y_min

    if x_range >= y_range:
        scale = 2.0 / x_range
        x_scaled = (x - x_min) * scale - 1.0
        y_scaled = (y - y_min) * scale - (y_range / x_range)
    else:
        scale = 2.0 / y_range
        y_scaled = (y - y_min) * scale - 1.0
        x_scaled = (x - x_min) * scale - (x_range / y_range)

    return x_scaled, y_scaled, scale


def sample_points_in_circle(
    n: int,
    r_min: float = 0.0,
    r_max: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Sample points uniformly by area from an annulus."""

    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(rng.uniform(r_min**2, r_max**2, n))
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def sample_points_in_ball(n: int, radius: float = 3.0, seed: int = 999) -> np.ndarray:
    """Sample points uniformly by area from a 2D disk."""

    return radius * sample_points_in_circle(n, seed=seed)


def quadratic_field(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quadratic source-sink field from the original notebook."""

    bx = x**2 - y**2 - 4
    by = 2 * x * y
    return bx, by


def make_base_field(field_name: str, n_samples: int = 1000, seed: int = 42) -> dict[str, object]:
    """Create clean 2D flat manifold positions, velocities, and time colors."""

    if field_name not in FIELD_NAMES:
        raise ValueError(f"field_name must be one of {FIELD_NAMES}")

    if field_name == "rotation":
        position = sample_points_in_circle(n_samples, r_min=0.3, r_max=1.0, seed=seed)
        x, y = position[:, 0], position[:, 1]
        velocity = np.stack([-y, x], axis=1)
        theta = np.arctan2(y, x)
        time = ((theta + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi)

    elif field_name == "spiral":
        position = sample_points_in_circle(n_samples, r_min=0.0, r_max=1.0, seed=seed)
        x, y = position[:, 0], position[:, 1]
        velocity = np.stack([x - y, x + y], axis=1)
        time = np.sqrt(x**2 + y**2)
        time /= time.max()

    elif field_name == "saddle":
        rng = np.random.default_rng(seed)
        position = rng.uniform(-2, 2, size=(n_samples, 2))
        theta = np.pi / 4
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )
        saddle = rotation @ np.diag([-1, 1]) @ rotation.T
        velocity = position @ saddle.T
        x, y, scale = scale_to_unit_box(position[:, 0], position[:, 1])
        position = np.column_stack([x, y])
        velocity = velocity * scale
        unstable_dir = np.array([1, -1]) / np.sqrt(2)
        time = np.abs(position @ unstable_dir)
        time /= time.max()

    elif field_name == "quadratic_source_sink":
        position = sample_points_in_ball(n_samples, radius=3.0, seed=seed)
        bx, by = quadratic_field(position[:, 0], position[:, 1])
        velocity = np.stack([bx, by], axis=1)
        x, y, scale = scale_to_unit_box(position[:, 0], position[:, 1])
        position = np.column_stack([x, y])
        velocity = velocity * scale
        reference_dir = np.array([-1.0, 0.0])
        projection = position @ reference_dir
        time = (projection - projection.min()) / (projection.max() - projection.min())

    elif field_name == "radial_source":
        position = sample_points_in_circle(n_samples, r_min=0.0, r_max=1.0, seed=seed)
        x, y = position[:, 0], position[:, 1]
        velocity = np.stack([x, y], axis=1)
        time = np.sqrt(x**2 + y**2)
        time /= time.max()

    elif field_name == "radial_sink":
        position = sample_points_in_circle(n_samples, r_min=0.0, r_max=1.0, seed=seed)
        x, y = position[:, 0], position[:, 1]
        velocity = np.stack([-x, -y], axis=1)
        time = 1.0 - np.sqrt(x**2 + y**2) / np.sqrt(x**2 + y**2).max()

    elif field_name == "horizontal_shear":
        rng = np.random.default_rng(seed)
        position = rng.uniform(-1, 1, size=(n_samples, 2))
        x, y = position[:, 0], position[:, 1]
        velocity = np.stack([y, np.zeros_like(y)], axis=1)
        time = (y - y.min()) / (y.max() - y.min())

    else:
        rng = np.random.default_rng(seed)
        position = rng.uniform(-1, 1, size=(n_samples, 2))
        x, y = position[:, 0], position[:, 1]
        velocity = np.stack([np.sin(np.pi * y), np.cos(np.pi * x)], axis=1)
        phase = np.sin(np.pi * x) + np.cos(np.pi * y)
        time = (phase - phase.min()) / (phase.max() - phase.min())

    return {
        "position": position,
        "velocity": velocity,
        "time": time,
        "name": field_name,
        "label": FIELD_LABELS[field_name],
    }


def make_flat_manifold_vector_field(config: FlatVectorFieldConfig) -> dict[str, object]:
    """Generate clean, noisy, and extra-dimensional flat vector-field data."""

    base = make_base_field(config.field_name, config.n_samples, config.seed)
    rng = np.random.default_rng(config.seed)

    x_gt = np.asarray(base["position"])
    v_gt = np.asarray(base["velocity"])

    x_noisy = x_gt + rng.normal(scale=config.position_noise, size=x_gt.shape)
    v_noisy = v_gt + rng.normal(scale=config.velocity_noise, size=v_gt.shape)

    x_dummy = rng.normal(scale=config.position_noise, size=(x_gt.shape[0], config.extra_dims))
    v_dummy = rng.normal(scale=config.velocity_noise, size=(v_gt.shape[0], config.extra_dims))

    x = np.hstack([x_noisy, x_dummy])
    v = np.hstack([v_noisy, v_dummy])

    return {
        "config": config.to_dict(),
        "base": base,
        "X": x,
        "V": v,
        "X_gt": x_gt,
        "V_gt": v_gt,
        "true_time": base["time"],
    }


def make_all_flat_manifold_vector_fields(
    n_samples: int = 1000,
    position_noise: float = 0.3,
    velocity_noise: float = 0.3,
    extra_dims: int = 5,
    seed: int = 42,
) -> dict[str, dict[str, object]]:
    """Generate all starter flat manifold vector-field simulations."""

    return {
        field_name: make_flat_manifold_vector_field(
            FlatVectorFieldConfig(
                field_name=field_name,
                n_samples=n_samples,
                position_noise=position_noise,
                velocity_noise=velocity_noise,
                extra_dims=extra_dims,
                seed=seed,
            )
        )
        for field_name in FIELD_NAMES
    }


def save_npz(simulation: dict[str, object], output_dir: str | Path = "simulation/outputs") -> Path:
    """Save one simulation dataset as NPZ plus a sibling JSON config."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / f"{simulation['config']['simulation_name']}.npz"
    payload = {
        "X": simulation["X"],
        "V": simulation["V"],
        "X_gt": simulation["X_gt"],
        "V_gt": simulation["V_gt"],
        "true_time": simulation["true_time"],
    }
    np.savez_compressed(path, **payload)
    path.with_suffix(".json").write_text(json.dumps(simulation["config"], indent=2) + "\n")
    return path
