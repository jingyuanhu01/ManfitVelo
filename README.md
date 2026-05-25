# ManfitVelo

Velocity-aware manifold fitting for simulation and RNA-velocity workflows.

The main implementation is `VelocityManifoldFitter` in
`scripts/velocity_manifold_fitter.py`. It takes a state matrix `Y` and matching
velocity matrix `W`, builds a velocity-aware neighbor graph, estimates local
tangent spaces with weighted PCA, projects velocities onto those tangent spaces,
and updates points with a normal-only manifold correction by default.

## Quick Start

```python
from scripts.velocity_manifold_fitter import VelocityManifoldFitter

fitter = VelocityManifoldFitter(Y, W)
result = fitter.fit(return_dict=True)

X_fit = result["X"]
V_fit = result["V"]
local_dims = result["local_dims"]
```

By default, high-dimensional inputs are globally reduced before fitting with
`use_PCA=True` and `PCA_dim=30`. If the original feature dimension is already
less than or equal to `PCA_dim`, the data are left unchanged. The returned
coordinates and velocities are in the fitting space.

## Parameter Priority

The class arguments are ordered by tuning priority.

### High-Priority Tuning Parameters

These are the parameters to tune first across datasets:

- `d_mode`: local PCA dimension mode. The default is `"adaptive"`.
- `adaptive_variance_threshold`: local explained-variance target for adaptive
  dimensions. The default is `0.8`.
- `adaptive_d_min`: minimum adaptive local dimension. The default is `2`.
- `adaptive_d_max`: optional maximum adaptive local dimension.
- `k`: local neighborhood size. The default is `25`.
- `T`: number of fitting iterations. The default is `5`.
- `eta_g`: normal correction step size. Smaller values are usually more stable.
- `theta`: velocity-aware neighbor scoring strength. Smaller values are usually
  more stable.

### Default Modes

These should usually stay fixed unless a diagnostic sweep suggests otherwise:

- `fit(update_mode="normal_only")`: recommended update rule. It removes the
  local tangent component of the mean-shift update and moves points only in the
  estimated normal direction.
- `fit(update_mode="original")`: comparison mode for the older mean-shift plus
  tangential velocity transport update.
- `bandwidth_mode="variable"`: recommended bandwidth mode.
- `bandwidth_mode="fixed"`: diagnostic option using the fixed bandwidth `h`.

### Lower-Priority Parameters

These are mostly diagnostic or dataset-specific controls:

- `global_d`: fixed local PCA dimension used only when `d_mode="global"`.
- `use_PCA`: whether to globally reduce `Y` and `W` before fitting.
- `PCA_dim`: target global PCA dimension when `use_PCA=True`.
- `gamma`, `beta`, `kappa`: velocity-aware scoring and kernel shape controls.
- `cv`: tangential velocity transport strength for `update_mode="original"`.
- `max_step_frac`: step-size cap as a fraction of local bandwidth.
- `h`: fixed bandwidth when `bandwidth_mode="fixed"`.
- `use_abs_cos`, `weight_use_abs_cos`: cosine sign conventions.
- `recompute_neighbors`, `candidate_mult`, `neighbor_update_freq`: neighbor
  search controls.

## Local PCA Dimension

The default mode is adaptive:

```python
fitter = VelocityManifoldFitter(
    Y,
    W,
    d_mode="adaptive",
    adaptive_variance_threshold=0.8,
    adaptive_d_min=2,
)
```

For each point, the local covariance spectrum is used to choose the smallest
dimension whose cumulative explained variance reaches
`adaptive_variance_threshold`, clipped by `adaptive_d_min` and
`adaptive_d_max`. The selected per-point dimensions are returned as
`result["local_dims"]`.

The vanilla fixed-dimension mode is still available:

```python
fitter = VelocityManifoldFitter(Y, W, d_mode="global", global_d=2)
```

## Global PCA

For RNA-velocity-scale data, the fitter can reduce the ambient space before
local fitting:

```python
fitter = VelocityManifoldFitter(Y, W, use_PCA=True, PCA_dim=30)
```

PCA is fit on `Y`, and velocities are projected into the same basis with
`W @ pca.components_.T`. Disable this behavior with `use_PCA=False` when the
input is already in the intended fitting space.

## Helper Functions

`select_adaptive_local_pca_dimension(eigvals, variance_threshold=0.8, d_min=2)`
chooses one local PCA dimension from a covariance spectrum.

`reduce_global_dimension(X, V, n_components=30)` reduces state and velocity
matrices with a shared PCA basis.

## Documentation

Sphinx source files are in `docs/`. To generate the HTML API documentation:

```bash
sphinx-build -b html docs docs/_build/html
```

## Notebooks

Useful notebooks live in `notebooks/method_tests/`:

- `ring_method_comparison.ipynb`
- `s_curve_parameter_workflow.ipynb`
- `rna_velocity_pca_manifold_fit.ipynb`

Reference notebooks and older implementations are kept in
`notebooks/reference_notebooks/` and `scripts/reference_implementations/`.
