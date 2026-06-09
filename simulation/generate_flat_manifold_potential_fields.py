"""Generate potential datasets.

Examples:
    python simulation/generate_flat_manifold_potential_fields.py --field all
    python simulation/generate_flat_manifold_potential_fields.py --field double_well --position-noise 0.2 --potential-noise 0.1
    python simulation/generate_flat_manifold_potential_fields.py --manifold half_sphere --field all
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.flat_manifold_potential_fields import (
    POTENTIAL_FIELD_NAMES,
    POTENTIAL_MANIFOLD_NAMES,
    FlatPotentialFieldConfig,
    make_all_flat_manifold_potential_fields,
    make_flat_manifold_potential_field,
    save_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifold", choices=POTENTIAL_MANIFOLD_NAMES, default="flat")
    parser.add_argument("--field", choices=("all", *POTENTIAL_FIELD_NAMES), default="all")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--position-noise", type=float, default=0.3)
    parser.add_argument("--potential-noise", type=float, default=0.3)
    parser.add_argument("--velocity-noise", type=float, dest="potential_noise", help=argparse.SUPPRESS)
    parser.add_argument("--extra-dims", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("simulation/outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.field == "all":
        simulations = make_all_flat_manifold_potential_fields(
            manifold_name=args.manifold,
            n_samples=args.n_samples,
            position_noise=args.position_noise,
            potential_noise=args.potential_noise,
            extra_dims=args.extra_dims,
            seed=args.seed,
        )
    else:
        simulation = make_flat_manifold_potential_field(
            FlatPotentialFieldConfig(
                manifold_name=args.manifold,
                field_name=args.field,
                n_samples=args.n_samples,
                position_noise=args.position_noise,
                potential_noise=args.potential_noise,
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
