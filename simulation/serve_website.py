"""Serve the simulation website with a local data-generation endpoint.

GitHub Pages can only host the static preview. Run this local server when the
Generate Data button should write matrices into simulation/data/.

    python simulation/serve_website.py
"""

from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEBSITE_DIR = ROOT / "simulation" / "website"
DATA_DIR = ROOT / "simulation" / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.flat_manifold_vector_fields import (  # noqa: E402
    FIELD_NAMES,
    FlatVectorFieldConfig,
    make_flat_manifold_vector_field,
)
from simulation.manifold_velocity_flows import (  # noqa: E402
    MANIFOLD_FIELD_NAMES,
    MANIFOLD_NAMES,
    ManifoldVelocityFlowConfig,
    make_manifold_velocity_flow,
)


def clean_number(value: float | int) -> str:
    """Return a filesystem-friendly numeric token."""

    token = f"{value:g}" if isinstance(value, float) else str(value)
    return token.replace("-", "m").replace(".", "p")


def safe_dir_name(config: dict[str, Any]) -> str:
    """Build the requested simulation/data output folder name."""

    name = str(config["simulation_name"])
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError("Invalid simulation_name")
    return (
        f"{name}"
        f"_position_noise_{clean_number(config['position_noise'])}"
        f"_velocity_noise_{clean_number(config['velocity_noise'])}"
        f"_extra_dims_{config['extra_dims']}"
    )


def parse_payload(raw_body: bytes) -> dict[str, Any]:
    """Validate and normalize the generate request body."""

    payload = json.loads(raw_body.decode("utf-8"))
    field_name = str(payload["field_name"])
    manifold = str(payload["manifold"])
    n_samples = int(payload["n_samples"])
    extra_dims = int(payload["extra_dims"])
    position_noise = float(payload["position_noise"])
    velocity_noise = float(payload["velocity_noise"])
    seed = int(payload.get("seed", 42))

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if extra_dims < 0:
        raise ValueError("extra_dims must be non-negative")
    if position_noise < 0 or velocity_noise < 0:
        raise ValueError("Noise values must be non-negative")

    if manifold == "flat":
        if field_name not in FIELD_NAMES:
            raise ValueError(f"Unknown flat field: {field_name}")
        simulation_name = f"flat_manifold__{field_name}_vector_field"
    else:
        if manifold not in MANIFOLD_NAMES:
            raise ValueError(f"Unknown manifold: {manifold}")
        if field_name not in MANIFOLD_FIELD_NAMES[manifold]:
            raise ValueError(f"{manifold} does not support field {field_name}")
        simulation_name = f"{manifold}_manifold__{field_name}_vector_field"

    requested_name = str(payload.get("simulation_name", simulation_name))
    if requested_name != simulation_name:
        raise ValueError("simulation_name does not match manifold and field_name")

    return {
        "simulation_name": simulation_name,
        "manifold": manifold,
        "field_name": field_name,
        "n_samples": n_samples,
        "position_noise": position_noise,
        "velocity_noise": velocity_noise,
        "extra_dims": extra_dims,
        "seed": seed,
    }


def generate_simulation(config: dict[str, Any]) -> dict[str, object]:
    """Run the matching reusable generator."""

    if config["manifold"] == "flat":
        return make_flat_manifold_vector_field(
            FlatVectorFieldConfig(
                field_name=config["field_name"],
                n_samples=config["n_samples"],
                position_noise=config["position_noise"],
                velocity_noise=config["velocity_noise"],
                extra_dims=config["extra_dims"],
                seed=config["seed"],
            )
        )

    return make_manifold_velocity_flow(
        ManifoldVelocityFlowConfig(
            manifold_name=config["manifold"],
            field_name=config["field_name"],
            n_samples=config["n_samples"],
            position_noise=config["position_noise"],
            velocity_noise=config["velocity_noise"],
            extra_dims=config["extra_dims"],
            seed=config["seed"],
        )
    )


def write_simulation(simulation: dict[str, object], config: dict[str, Any]) -> Path:
    """Write X and V matrices into simulation/data/<config-derived-name>/."""

    output_dir = DATA_DIR / safe_dir_name(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X.npy", simulation["X"])
    np.save(output_dir / "V.npy", simulation["V"])
    (output_dir / "metadata.json").write_text(
        json.dumps(simulation["config"], indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


class SimulationRequestHandler(SimpleHTTPRequestHandler):
    """Static website handler plus POST /api/generate."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEBSITE_DIR), **kwargs)

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/api/generate":
            self.send_json(404, {"error": "Unknown endpoint"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            config = parse_payload(self.rfile.read(length))
            simulation = generate_simulation(config)
            output_dir = write_simulation(simulation, config)
        except Exception as exc:
            self.send_json(400, {"error": str(exc)})
            return

        self.send_json(
            200,
            {
                "ok": True,
                "output_dir": str(output_dir.relative_to(ROOT)),
                "files": ["X.npy", "V.npy", "metadata.json"],
            },
        )


def main() -> None:
    port = 8765
    server = ThreadingHTTPServer(("127.0.0.1", port), SimulationRequestHandler)
    print(f"Serving simulation website at http://127.0.0.1:{port}")
    print("Generated matrices will be written under simulation/data/")
    server.serve_forever()


if __name__ == "__main__":
    main()
