"""Generate flat manifold vector-field datasets.

Examples:
    python simulation/generate_flat_manifold_vector_fields.py --field all
    python simulation/generate_flat_manifold_vector_fields.py --field rotation --position-noise 0.2 --velocity-noise 0.4 --extra-dims 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.flat_manifold_vector_fields import (
    FIELD_NAMES,
    FlatVectorFieldConfig,
    make_all_flat_manifold_vector_fields,
    make_flat_manifold_vector_field,
    save_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", choices=("all", *FIELD_NAMES), default="all")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--position-noise", type=float, default=0.3)
    parser.add_argument("--velocity-noise", type=float, default=0.3)
    parser.add_argument("--extra-dims", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("simulation/outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.field == "all":
        simulations = make_all_flat_manifold_vector_fields(
            n_samples=args.n_samples,
            position_noise=args.position_noise,
            velocity_noise=args.velocity_noise,
            extra_dims=args.extra_dims,
            seed=args.seed,
        )
    else:
        simulation = make_flat_manifold_vector_field(
            FlatVectorFieldConfig(
                field_name=args.field,
                n_samples=args.n_samples,
                position_noise=args.position_noise,
                velocity_noise=args.velocity_noise,
                extra_dims=args.extra_dims,
                seed=args.seed,
            )
        )
        simulations = {args.field: simulation}

    for simulation in simulations.values():
        path = save_npz(simulation, output_dir=args.output_dir)
        print(path)


if __name__ == "__main__":
    main()
