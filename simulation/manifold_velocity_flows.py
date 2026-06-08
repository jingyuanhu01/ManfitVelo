"""3D manifold velocity-flow data generation.

This module covers manifolds where the clean coordinates are intrinsically
3D before optional noisy extra dimensions are appended. It does not run
manifold fitting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


MANIFOLD_NAMES = (
    "s_curve",
    "swiss_roll",
    "half_sphere",
)

VECTOR_FIELD_NAMES = (
    "velocity_flow",
    "rotation",
    "spiral",
    "saddle",
    "radial_source",
    "radial_sink",
    "quadratic_source_sink",
)

MANIFOLD_FIELD_NAMES = {
    "s_curve": ("velocity_flow",),
    "swiss_roll": ("velocity_flow",),
    "half_sphere": VECTOR_FIELD_NAMES,
}

MANIFOLD_LABELS = {
    "s_curve": "s curve manifold, velocity flow field",
    "swiss_roll": "swiss roll manifold, velocity flow field",
    "half_sphere": "half sphere manifold, velocity flow field",
}

FIELD_LABELS = {
    "velocity_flow": "velocity flow",
    "rotation": "rotation",
    "spiral": "spiral",
    "saddle": "saddle",
    "radial_source": "radial source",
    "radial_sink": "radial sink",
    "quadratic_source_sink": "quadratic source sink",
}


@dataclass(frozen=True)
class ManifoldVelocityFlowConfig:
    """Config for 3D manifold velocity-flow data generation."""

    manifold_name: str = "s_curve"
    field_name: str = "velocity_flow"
    n_samples: int = 1000
    position_noise: float = 0.3
    velocity_noise: float = 0.3
    extra_dims: int = 5
    seed: int = 42

    @property
    def simulation_name(self) -> str:
        return f"{self.manifold_name}_manifold__{self.field_name}_vector_field"

    def to_dict(self) -> dict[str, object]:
        return asdict(self) | {"simulation_name": self.simulation_name}


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """Normalize rows with zero-safe denominators."""

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def scale_to_unit_box_3d(position: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center and scale positions to a comparable 3D box."""

    center = position.mean(axis=0)
    centered = position - center
    scale = np.max(np.ptp(centered, axis=0))
    if scale == 0:
        scale = 1.0
    return centered * (2.0 / scale), velocity * (2.0 / scale)


def project_to_tangent(position: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Project ambient vectors onto the local sphere tangent planes."""

    normals = normalize_rows(position)
    normal_component = np.sum(vector * normals, axis=1, keepdims=True) * normals
    return vector - normal_component


def make_half_sphere_field(position: np.ndarray, field_name: str) -> np.ndarray:
    """Create an ambient field and project it onto the half-sphere tangent plane."""

    x = position[:, 0]
    z = position[:, 2]

    if field_name == "velocity_flow" or field_name == "rotation":
        ambient = np.stack([-z, np.zeros_like(x), x], axis=1)
    elif field_name == "spiral":
        ambient = np.stack([x - z, np.zeros_like(x), x + z], axis=1)
    elif field_name == "saddle":
        c = np.sqrt(0.5)
        u = c * x + c * z
        v = -c * x + c * z
        fx = c * (-u) - c * v
        fz = c * (-u) + c * v
        ambient = np.stack([fx, np.zeros_like(x), fz], axis=1)
    elif field_name == "radial_source":
        ambient = np.stack([x, np.zeros_like(x), z], axis=1)
    elif field_name == "radial_sink":
        ambient = np.stack([-x, np.zeros_like(x), -z], axis=1)
    elif field_name == "quadratic_source_sink":
        ambient = np.stack([x * x - z * z - 0.45, np.zeros_like(x), 2.0 * x * z], axis=1)
    else:
        raise ValueError(f"field_name must be one of {VECTOR_FIELD_NAMES}")

    return project_to_tangent(position, ambient)


def make_base_flow(
    manifold_name: str,
    n_samples: int = 1000,
    seed: int = 42,
    field_name: str = "velocity_flow",
) -> dict[str, object]:
    """Create clean 3D manifold positions, velocity flow, and time colors."""

    if manifold_name not in MANIFOLD_NAMES:
        raise ValueError(f"manifold_name must be one of {MANIFOLD_NAMES}")
    if field_name not in MANIFOLD_FIELD_NAMES[manifold_name]:
        raise ValueError(f"field_name must be one of {MANIFOLD_FIELD_NAMES[manifold_name]} for {manifold_name}")

    rng = np.random.default_rng(seed)

    if manifold_name == "s_curve":
        t = 3 * np.pi * (rng.uniform(size=n_samples) - 0.5)
        height = 2.0 * rng.uniform(size=n_samples)
        position = np.empty((n_samples, 3), dtype=float)
        position[:, 0] = np.sin(t)
        position[:, 1] = height
        position[:, 2] = np.sign(t) * (np.cos(t) - 1)

        velocity = np.stack(
            [
                np.cos(t),
                np.zeros_like(t),
                -np.sin(t) * np.sign(t),
            ],
            axis=1,
        )
        time = (t - t.min()) / (t.max() - t.min())

    elif manifold_name == "swiss_roll":
        t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, size=n_samples)
        height = rng.uniform(-1.0, 1.0, size=n_samples)
        position = np.empty((n_samples, 3), dtype=float)
        position[:, 0] = t * np.cos(t)
        position[:, 1] = height
        position[:, 2] = t * np.sin(t)

        velocity = np.stack(
            [
                np.cos(t) - t * np.sin(t),
                np.zeros_like(t),
                np.sin(t) + t * np.cos(t),
            ],
            axis=1,
        )
        time = (t - t.min()) / (t.max() - t.min())

    else:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
        vertical = rng.uniform(0.0, 1.0, size=n_samples)
        radius = np.sqrt(1.0 - vertical * vertical)
        horizontal_x = radius * np.cos(theta)
        horizontal_z = radius * np.sin(theta)

        position = np.stack([horizontal_x, vertical, horizontal_z], axis=1)
        velocity = make_half_sphere_field(position, field_name)
        time = theta / (2.0 * np.pi)

    position, velocity = scale_to_unit_box_3d(position, velocity)
    velocity = normalize_rows(velocity)

    return {
        "position": position,
        "velocity": velocity,
        "time": time,
        "name": manifold_name,
        "field_name": field_name,
        "label": (
            MANIFOLD_LABELS[manifold_name]
            if field_name == "velocity_flow"
            else f"{manifold_name.replace('_', ' ')} manifold, {FIELD_LABELS[field_name]} field"
        ),
    }


def make_manifold_velocity_flow(config: ManifoldVelocityFlowConfig) -> dict[str, object]:
    """Generate clean, noisy, and extra-dimensional manifold velocity-flow data."""

    base = make_base_flow(
        config.manifold_name,
        n_samples=config.n_samples,
        seed=config.seed,
        field_name=config.field_name,
    )
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


def make_all_manifold_velocity_flows(
    n_samples: int = 1000,
    position_noise: float = 0.3,
    velocity_noise: float = 0.3,
    extra_dims: int = 5,
    seed: int = 42,
) -> dict[str, dict[str, object]]:
    """Generate all 3D manifold velocity-flow simulations."""

    simulations = {}
    for manifold_name in MANIFOLD_NAMES:
        for field_name in MANIFOLD_FIELD_NAMES[manifold_name]:
            simulation = make_manifold_velocity_flow(
                ManifoldVelocityFlowConfig(
                    manifold_name=manifold_name,
                    field_name=field_name,
                    n_samples=n_samples,
                    position_noise=position_noise,
                    velocity_noise=velocity_noise,
                    extra_dims=extra_dims,
                    seed=seed,
                )
            )
            simulations[simulation["config"]["simulation_name"]] = simulation
    return simulations


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
