"""Flat manifold potential-field data generation.

Each potential defines a scalar U(x, y). The generated velocity is the
gradient-flow direction V = -grad U. Potential simulations add noise to the
observed scalar potential, not directly to the velocity vectors. No manifold
fitting is run here.
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
    "entropy_like",
)

POTENTIAL_FIELD_LABELS = {
    "single_basin": "flat manifold, single basin potential",
    "double_well": "flat manifold, double well potential",
    "saddle": "flat manifold, saddle potential",
    "entropy_like": "flat manifold, entropy-like potential",
}


@dataclass(frozen=True)
class FlatPotentialFieldConfig:
    """Config for flat manifold potential-field data generation."""

    field_name: str = "single_basin"
    n_samples: int = 1000
    position_noise: float = 0.3
    potential_noise: float = 0.3
    extra_dims: int = 5
    seed: int = 42

    @property
    def simulation_name(self) -> str:
        return f"flat_manifold__{self.field_name}_potential"

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


def potential_and_gradient(field_name: str, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate U and grad U for a named flat potential field."""

    x = position[:, 0]
    y = position[:, 1]

    if field_name == "single_basin":
        potential = 0.5 * (x * x + y * y)
        gradient = np.stack([x, y], axis=1)

    elif field_name == "double_well":
        potential = (x * x - 1.0) ** 2 + 0.45 * y * y
        gradient = np.stack([4.0 * x * (x * x - 1.0), 0.9 * y], axis=1)

    elif field_name == "saddle":
        potential = 0.5 * (x * x - y * y)
        gradient = np.stack([x, -y], axis=1)

    elif field_name == "entropy_like":
        eps = 1e-3
        radius_sq = x * x + y * y
        potential = 0.5 * radius_sq * np.log(radius_sq + eps)
        factor = np.log(radius_sq + eps) + radius_sq / (radius_sq + eps)
        gradient = np.stack([x * factor, y * factor], axis=1)

    else:
        raise ValueError(f"field_name must be one of {POTENTIAL_FIELD_NAMES}")

    return potential, gradient


def make_base_potential_field(field_name: str, n_samples: int = 1000, seed: int = 42) -> dict[str, object]:
    """Create clean positions, potential values, and negative-gradient velocities."""

    if field_name not in POTENTIAL_FIELD_NAMES:
        raise ValueError(f"field_name must be one of {POTENTIAL_FIELD_NAMES}")

    position = sample_square(n_samples, seed=seed)
    potential, gradient = potential_and_gradient(field_name, position)
    velocity = -normalize_rows(gradient)

    return {
        "position": position,
        "velocity": velocity,
        "gradient": gradient,
        "potential": potential,
        "potential_normalized": normalize_values(potential),
        "name": field_name,
        "label": POTENTIAL_FIELD_LABELS[field_name],
    }


def make_flat_manifold_potential_field(config: FlatPotentialFieldConfig) -> dict[str, object]:
    """Generate clean, noisy, and extra-dimensional flat potential data."""

    base = make_base_potential_field(config.field_name, config.n_samples, config.seed)
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
    n_samples: int = 1000,
    position_noise: float = 0.3,
    potential_noise: float = 0.3,
    extra_dims: int = 5,
    seed: int = 42,
) -> dict[str, dict[str, object]]:
    """Generate all flat potential simulations."""

    return {
        field_name: make_flat_manifold_potential_field(
            FlatPotentialFieldConfig(
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
