"""Potential-field data generation.

Each potential defines a scalar U(x, y). The generated velocity is the
gradient-flow direction V = -grad U. On curved manifolds, U is defined over
horizontal coordinates and grad U is lifted to the tangent space. Potential
simulations add noise to the observed scalar potential, not directly to the
gradient vectors. No manifold fitting is run here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


POTENTIAL_FIELD_NAMES = (
    "single_basin",
    "double_well",
    "saddle",
    "linear",
)

POTENTIAL_MANIFOLD_NAMES = ("flat", "half_sphere", "saddle_surface")

POTENTIAL_FIELD_LABELS = {
    "single_basin": "single basin potential",
    "double_well": "double well potential",
    "saddle": "saddle potential",
    "linear": "linear potential",
}


@dataclass(frozen=True)
class FlatPotentialFieldConfig:
    """Config for flat manifold potential-field data generation."""

    manifold_name: str = "flat"
    field_name: str = "single_basin"
    n_samples: int = 1000
    position_noise: float = 0.3
    potential_noise: float = 0.3
    extra_dims: int = 5
    seed: int = 42

    @property
    def simulation_name(self) -> str:
        return f"{self.manifold_name}_manifold__{self.field_name}_potential"

    def to_dict(self) -> dict[str, object]:
        return asdict(self) | {"simulation_name": self.simulation_name}


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """Normalize rows with zero-safe denominators."""

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def normalize_values(values: np.ndarray) -> np.ndarray:
    """Map values into [0, 1] with a zero-safe range."""

    value_min = np.min(values)
    value_range = np.max(values) - value_min
    if value_range == 0:
        return np.zeros_like(values)
    return (values - value_min) / value_range


def sample_square(n_samples: int, seed: int = 42, extent: float = 1.6) -> np.ndarray:
    """Sample flat 2D points from a square domain."""

    rng = np.random.default_rng(seed)
    return rng.uniform(-extent, extent, size=(n_samples, 2))


def potential_and_coordinate_gradient(
    field_name: str,
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate U and dU/dcoordinates for a named potential."""

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    if field_name == "single_basin":
        potential = 0.5 * (x * x + y * y)
        gradient = np.stack([x, y], axis=1)

    elif field_name == "double_well":
        potential = (x * x - 1.0) ** 2 + 0.45 * y * y
        gradient = np.stack([4.0 * x * (x * x - 1.0), 0.9 * y], axis=1)

    elif field_name == "saddle":
        potential = 0.5 * (x * x - y * y)
        gradient = np.stack([x, -y], axis=1)

    elif field_name == "linear":
        potential = x + 0.45 * y
        gradient = np.stack([np.ones_like(x), np.full_like(y, 0.45)], axis=1)

    else:
        raise ValueError(f"field_name must be one of {POTENTIAL_FIELD_NAMES}")

    return potential, gradient


def sample_half_sphere(n_samples: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Sample points on the upper half sphere plus horizontal coordinates."""

    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    vertical = rng.uniform(0.0, 1.0, size=n_samples)
    radius = np.sqrt(1.0 - vertical * vertical)
    x = radius * np.cos(theta)
    z = radius * np.sin(theta)
    position = np.stack([x, vertical, z], axis=1)
    coordinates = np.stack([x, z], axis=1)
    return position, coordinates


def sample_saddle_surface(
    n_samples: int,
    seed: int = 42,
    extent: float = 1.2,
    curvature: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample points on y = curvature * (x^2 - z^2)."""

    rng = np.random.default_rng(seed)
    x = rng.uniform(-extent, extent, size=n_samples)
    z = rng.uniform(-extent, extent, size=n_samples)
    y = curvature * (x * x - z * z)
    position = np.stack([x, y, z], axis=1)
    coordinates = np.stack([x, z], axis=1)
    return position, coordinates


def project_to_tangent(position: np.ndarray, ambient_gradient: np.ndarray) -> np.ndarray:
    """Project ambient vectors to the tangent plane of the unit sphere."""

    normal = normalize_rows(position)
    normal_component = np.sum(ambient_gradient * normal, axis=1, keepdims=True)
    return ambient_gradient - normal_component * normal


def saddle_surface_gradient(
    coordinate_gradient: np.ndarray,
    coordinates: np.ndarray,
    curvature: float = 0.55,
) -> np.ndarray:
    """Convert coordinate gradients to tangent gradients on the saddle surface."""

    x = coordinates[:, 0]
    z = coordinates[:, 1]
    fu = coordinate_gradient[:, 0]
    fv = coordinate_gradient[:, 1]
    ru_y = 2.0 * curvature * x
    rv_y = -2.0 * curvature * z
    g11 = 1.0 + ru_y * ru_y
    g22 = 1.0 + rv_y * rv_y
    g12 = ru_y * rv_y
    determinant = g11 * g22 - g12 * g12
    a = (g22 * fu - g12 * fv) / determinant
    b = (-g12 * fu + g11 * fv) / determinant
    return np.stack([a, a * ru_y + b * rv_y, b], axis=1)


def make_base_potential_field(
    manifold_name: str,
    field_name: str,
    n_samples: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    """Create clean positions, potential values, and tangent gradients."""

    if manifold_name not in POTENTIAL_MANIFOLD_NAMES:
        raise ValueError(f"manifold_name must be one of {POTENTIAL_MANIFOLD_NAMES}")
    if field_name not in POTENTIAL_FIELD_NAMES:
        raise ValueError(f"field_name must be one of {POTENTIAL_FIELD_NAMES}")

    if manifold_name == "flat":
        position = sample_square(n_samples, seed=seed)
        coordinates = position
        potential, coordinate_gradient = potential_and_coordinate_gradient(field_name, coordinates)
        gradient = coordinate_gradient
    elif manifold_name == "half_sphere":
        position, coordinates = sample_half_sphere(n_samples, seed=seed)
        potential, coordinate_gradient = potential_and_coordinate_gradient(field_name, coordinates)
        ambient_gradient = np.stack(
            [coordinate_gradient[:, 0], np.zeros(n_samples), coordinate_gradient[:, 1]],
            axis=1,
        )
        gradient = project_to_tangent(position, ambient_gradient)
    else:
        position, coordinates = sample_saddle_surface(n_samples, seed=seed)
        potential, coordinate_gradient = potential_and_coordinate_gradient(field_name, coordinates)
        gradient = saddle_surface_gradient(coordinate_gradient, coordinates)

    velocity = -normalize_rows(gradient)

    return {
        "position": position,
        "coordinates": coordinates,
        "velocity": velocity,
        "gradient": gradient,
        "potential": potential,
        "potential_normalized": normalize_values(potential),
        "name": field_name,
        "label": f"{manifold_name.replace('_', ' ')} manifold, {POTENTIAL_FIELD_LABELS[field_name]}",
    }


def make_flat_manifold_potential_field(config: FlatPotentialFieldConfig) -> dict[str, object]:
    """Generate clean, noisy, and extra-dimensional flat potential data."""

    base = make_base_potential_field(
        config.manifold_name,
        config.field_name,
        config.n_samples,
        config.seed,
    )
    rng = np.random.default_rng(config.seed)

    x_gt = np.asarray(base["position"])
    v_gt = np.asarray(base["velocity"])
    grad_f_gt = np.asarray(base["gradient"])
    potential_gt = np.asarray(base["potential"])

    x_noisy = x_gt + rng.normal(scale=config.position_noise, size=x_gt.shape)
    potential_noisy = potential_gt + rng.normal(scale=config.potential_noise, size=potential_gt.shape)

    x_dummy = rng.normal(scale=config.position_noise, size=(x_gt.shape[0], config.extra_dims))
    v_dummy = np.zeros((v_gt.shape[0], config.extra_dims))

    x = np.hstack([x_noisy, x_dummy])
    v = np.hstack([v_gt, v_dummy])

    return {
        "config": config.to_dict(),
        "base": base,
        "X": x,
        "V": v,
        "X_gt": x_gt,
        "V_gt": v_gt,
        "f": potential_noisy,
        "f_gt": potential_gt,
        "grad_f_gt": grad_f_gt,
        "potential": potential_noisy,
        "potential_gt": potential_gt,
        "potential_normalized": normalize_values(potential_noisy),
        "potential_gt_normalized": base["potential_normalized"],
    }


def make_all_flat_manifold_potential_fields(
    manifold_name: str = "flat",
    n_samples: int = 1000,
    position_noise: float = 0.3,
    potential_noise: float = 0.3,
    extra_dims: int = 5,
    seed: int = 42,
) -> dict[str, dict[str, object]]:
    """Generate all potential simulations for one manifold."""

    return {
        field_name: make_flat_manifold_potential_field(
            FlatPotentialFieldConfig(
                manifold_name=manifold_name,
                field_name=field_name,
                n_samples=n_samples,
                position_noise=position_noise,
                potential_noise=potential_noise,
                extra_dims=extra_dims,
                seed=seed,
            )
        )
        for field_name in POTENTIAL_FIELD_NAMES
    }


def save_npz(simulation: dict[str, object], output_dir: str | Path = "simulation/outputs") -> Path:
    """Save one potential-field dataset as NPZ plus a sibling JSON config."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / f"{simulation['config']['simulation_name']}.npz"
    payload = {
        "X": simulation["X"],
        "V": simulation["V"],
        "X_gt": simulation["X_gt"],
        "V_gt": simulation["V_gt"],
        "f": simulation["f"],
        "f_gt": simulation["f_gt"],
        "grad_f_gt": simulation["grad_f_gt"],
        "potential": simulation["potential"],
        "potential_gt": simulation["potential_gt"],
        "potential_normalized": simulation["potential_normalized"],
        "potential_gt_normalized": simulation["potential_gt_normalized"],
    }
    np.savez_compressed(path, **payload)
    path.with_suffix(".json").write_text(json.dumps(simulation["config"], indent=2) + "\n")
    return path
