"""Generate 3D manifold velocity-flow datasets.

Examples:
    python simulation/generate_manifold_velocity_flows.py --manifold all
    python simulation/generate_manifold_velocity_flows.py --manifold swiss_roll --position-noise 0.2 --velocity-noise 0.4
    python simulation/generate_manifold_velocity_flows.py --manifold half_sphere --extra-dims 8
    python simulation/generate_manifold_velocity_flows.py --manifold half_sphere --field saddle
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.manifold_velocity_flows import (
    MANIFOLD_FIELD_NAMES,
    MANIFOLD_NAMES,
    ManifoldVelocityFlowConfig,
    VECTOR_FIELD_NAMES,
    make_all_manifold_velocity_flows,
    make_manifold_velocity_flow,
    save_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifold", choices=("all", *MANIFOLD_NAMES), default="all")
    parser.add_argument("--field", choices=("all", *VECTOR_FIELD_NAMES), default="all")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--position-noise", type=float, default=0.3)
    parser.add_argument("--velocity-noise", type=float, default=0.3)
    parser.add_argument("--extra-dims", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("simulation/outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.manifold == "all":
        if args.field != "all":
            raise ValueError("--field can only be set with a specific --manifold")
        simulations = make_all_manifold_velocity_flows(
            n_samples=args.n_samples,
            position_noise=args.position_noise,
            velocity_noise=args.velocity_noise,
            extra_dims=args.extra_dims,
            seed=args.seed,
        )
    else:
        field_names = MANIFOLD_FIELD_NAMES[args.manifold] if args.field == "all" else (args.field,)
        invalid_fields = [field_name for field_name in field_names if field_name not in MANIFOLD_FIELD_NAMES[args.manifold]]
        if invalid_fields:
            raise ValueError(f"{args.manifold} supports fields: {MANIFOLD_FIELD_NAMES[args.manifold]}")
        simulations = {}
        for field_name in field_names:
            simulation = make_manifold_velocity_flow(
                ManifoldVelocityFlowConfig(
                    manifold_name=args.manifold,
                    field_name=field_name,
                    n_samples=args.n_samples,
                    position_noise=args.position_noise,
                    velocity_noise=args.velocity_noise,
                    extra_dims=args.extra_dims,
                    seed=args.seed,
                )
            )
            simulations[simulation["config"]["simulation_name"]] = simulation

    for simulation in simulations.values():
        path = save_npz(simulation, output_dir=args.output_dir)
        print(path)


if __name__ == "__main__":
    main()
