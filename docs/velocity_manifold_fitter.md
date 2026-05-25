# VelocityManifoldFitter API

This page mirrors the Sphinx-style documentation in
`scripts/velocity_manifold_fitter.py`.

## Class

`VelocityManifoldFitter(Y, W, ...)`

The fitter takes a state matrix `Y` and matching velocity matrix `W`. It builds
a velocity-aware neighbor graph, estimates local tangent spaces with weighted
PCA, projects velocities onto those tangent spaces, and updates positions with a
normal-only default rule.

### High-Priority Tuning Parameters

Tune these first:

- `d_mode`: `"adaptive"` by default. Use `"global"` to force a fixed local
  dimension.
- `adaptive_variance_threshold`: default `0.8`.
- `adaptive_d_min`: default `2`.
- `adaptive_d_max`: optional adaptive cap.
- `k`: local neighborhood size.
- `T`: number of fitting iterations.
- `eta_g`: normal correction step size.
- `theta`: velocity-aware neighbor scoring strength.

### Default Modes

These are intended to remain fixed for most runs:

- `fit(update_mode="normal_only")`
- `bandwidth_mode="variable"`

Comparison and diagnostic alternatives are available with
`fit(update_mode="original")` and `bandwidth_mode="fixed"`.

### Lower-Priority Parameters

These are useful for diagnostics but should not usually be the first tuning
target:

- `global_d`
- `use_PCA`
- `PCA_dim`
- `gamma`
- `beta`
- `kappa`
- `cv`
- `max_step_frac`
- `h`
- `use_abs_cos`
- `weight_use_abs_cos`
- `recompute_neighbors`
- `candidate_mult`
- `neighbor_update_freq`

## Methods

`fit(update_mode="normal_only", velocity_mode="projected", blend_lambda=0.0, return_dict=False)`

Runs manifold fitting. `update_mode="normal_only"` is the recommended default.
It moves each point only in the estimated normal direction and uses the tangent
space to project velocities. `update_mode="original"` retains the older
mean-shift plus tangential velocity transport update and is mainly useful as a
comparison mode.

When `return_dict=True`, the result includes:

- `X`: fitted positions.
- `V`: projected velocities.
- `neighbors`: velocity-aware neighbor indices.
- `weights`: local fitting weights.
- `U`: local tangent bases.
- `P`: local tangent projectors.
- `local_dims`: selected local PCA dimensions.
- `bandwidths`: local kernel bandwidths.
- `global_pca`: fitted global PCA object, or `None`.
- `history`: per-iteration step summaries.

## Helper Functions

`select_adaptive_local_pca_dimension(eigvals, variance_threshold=0.8, d_min=2, d_max=None, eps=1e-12)`

Chooses the smallest local PCA dimension whose cumulative explained variance
reaches the threshold, clipped to `[d_min, d_max]`.

`reduce_global_dimension(X, V, n_components=30, random_state=0)`

Fits PCA on `X`, transforms `X`, and projects velocities into the same basis
with `V @ pca.components_.T`.
