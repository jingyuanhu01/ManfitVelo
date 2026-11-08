# ManfitVelo

Velocity-aware manifold fitting from state and velocity matrices, with scalar-gradient fitting, potential reconstruction, and intrinsic-Jacobian geometry utilities.

This repository contains the reusable method implementation. Simulation benchmarks, biological analyses, datasets, figures, and manuscript reproduction workflows live in [ManfitVelo-manuscript](https://github.com/jingyuanhu01/ManfitVelo-manuscript).

## Install

Python 3.10 or newer:

```bash
python -m pip install "git+https://github.com/jingyuanhu01/ManfitVelo.git"
```

For local development, clone this repository and run `python -m pip install -e ".[test]"`.

## Usage

```python
import numpy as np
from manfitvelo import VelocityManifoldFitter

t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
Y = np.column_stack((np.cos(t), np.sin(t), np.zeros_like(t)))
W = np.column_stack((-np.sin(t), np.cos(t), np.zeros_like(t)))
result = VelocityManifoldFitter(Y, W, use_PCA=False).fit(return_dict=True)
X_fit, V_fit = result["X"], result["V"]
```

`Y` and `W` must have matching shapes `(n_samples, n_features)`. Global PCA is enabled by default with `PCA_dim=30`; fitted outputs are in the fitting coordinate system. The default `lambda_v=0.0` is preserved. Manuscript experiments explicitly supply their own frozen settings, including `lambda_v=1.0` for the directly observed velocity protocol; see the manuscript repository for those settings.

## API

- `manfitvelo.VelocityManifoldFitter`: joint state/velocity manifold fitting.
- `manfitvelo.fit_scalar_gradient_manfit`: fitting from noisy scalar observations.
- `manfitvelo.potential_from_gradient`: reconstruct a scalar potential from a gradient field.
- `manfitvelo.geometry.estimate_jacobian_field`: intrinsic/ambient Jacobian estimation.
- `manfitvelo.geometry.connection_aware_smooth`: connection-aware Jacobian smoothing.
- `manfitvelo.geometry.downstream_geometry`: acceleration, strain/rotation, and fixed-point utilities.

See [the fitter documentation](docs/velocity_manifold_fitter.md) for parameters. Build the API reference with `python -m pip install -e ".[docs]"` followed by `sphinx-build -b html docs docs/_build/html`.

## Tests

```bash
python -m pytest
```

The package has no dependency on manuscript datasets, plotting libraries, or benchmark configurations. Existing `scripts.*` imports should be changed to `manfitvelo.*`; no second copy of the method is maintained in the manuscript repository.

## Provenance

Extracted without numerical algorithm changes from ManfitVelo commit `5b17c165f3d27c71a758156c9629ee9a1fee2dab`. Both repositories preserve the original Git history. Historical manuscript files remain available in earlier commits.
