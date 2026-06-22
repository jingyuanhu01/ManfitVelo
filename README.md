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
- `s_curve_gradient_field_embedding.ipynb`
- `protein_latent_gradient_pipeline.ipynb`

Reference notebooks and older implementations are kept in
`notebooks/reference_notebooks/` and `scripts/reference_implementations/`.

## Position + Potential Experiments

The current scalar-potential proof of concept compares position-only manifold
fitting against manifold fitting with an estimated local gradient field. The
gradient is estimated from the observed scalar potential by local ridge
regression over neighborhoods, then passed to `VelocityManifoldFitter`.

Two notebooks are the main entry points:

- `notebooks/method_tests/s_curve_gradient_field_embedding.ipynb`: simulated
  S-curve with noisy positions, noisy scalar potential, and known ground truth.
- `notebooks/method_tests/protein_latent_gradient_pipeline.ipynb`: P450 protein
  fitness landscape example, using measured `T50` as the scalar potential.

The protein data come from the Nature Communications paper:

> Ding, X., Zou, Z. & Brooks III, C.L. Deciphering protein evolution and
> fitness landscapes with latent space models. Nature Communications 10, 5644
> (2019). https://doi.org/10.1038/s41467-019-13633-0

The repository does not commit the downloaded data files. To reproduce the
protein notebook, download the supplementary data files from the paper into
`data/protein_latent_paper/raw/`, then run
`scripts/prepare_protein_latent_paper_data.py` to create the local processed
files used by the notebook.

## Benchmark Pipeline

The benchmark scripts turn the exploratory notebooks into reproducible,
quantitative comparisons against global PCA denoising baselines.

Run the synthetic simulation benchmark:

```bash
python scripts/run_simulation_benchmark.py
```

Run the real-data application geometry report:

```bash
python scripts/run_application_geometry_report.py
```

Run integrity and numerical sanity checks:

```bash
python scripts/check_benchmark_integrity.py
```

Run a quick VMF parameter sweep:

```bash
python scripts/run_parameter_sweep.py --quick
```

Generated reports:

- `reports/simulation_benchmark/index.html`
- `reports/simulation_benchmark/simulation_results_long.csv`
- `reports/simulation_benchmark/simulation_results_summary.csv`
- `reports/application_geometry/index.html`
- `reports/application_geometry/application_results_long.csv`
- `reports/application_geometry/application_results_summary.csv`
- `reports/parameter_sweep/index.html`
- `reports/parameter_sweep/parameter_sweep_results.csv`
- `reports/parameter_sweep/parameter_sweep_summary.csv`

Implemented benchmark modules:

- `scripts/pca_denoisers.py`: fixed-rank PCA, variance-threshold PCA,
  optional local PCA, vector projection through retained PCA components, and
  oracle rank sweep for simulation only.
- `scripts/geometry_velocity_metrics.py`: reconstruction, clean-cloud
  distance, local spectrum, normal/tangent energy, local spectral gap,
  effective dimension, velocity-tangent alignment, velocity-neighbor direction
  agreement, velocity smoothness, displacement, kNN overlap, distance
  correlation, and trustworthiness helpers.
- `scripts/html_report_utils.py`: small self-contained HTML report writer.
- `scripts/run_parameter_sweep.py`: quick VMF sweep over `eta_g`, `theta`,
  `k`, `T`, and adaptive local dimension threshold, with alignment/movement
  tradeoff reports.

Compared methods in the simulation benchmark:

- raw noisy data;
- PCA rank `d`, `2d`, and `5d`;
- PCA variance thresholds at 90% and 95%;
- position-only MANFIT when the sample size is small enough for the legacy
  implementation;
- `VelocityManifoldFitter`.

Scientific interpretation:

- PCA denoising tests whether a simple global linear subspace explains the
  apparent improvement.
- Velocity-aware MANFIT should improve local geometry and velocity
  compatibility beyond global PCA when the data lie near a curved manifold or
  when velocity contains useful tangent information.
- Real-data reports do not use ground-truth reconstruction metrics. They focus
  on geometry and velocity utility: local low-dimensionality, velocity-tangent
  compatibility, movement size, and neighborhood preservation.

### Latest benchmark run notes

Run date: 2026-06-22 00:33:26 CST.

Commands run:

- `python scripts/check_benchmark_integrity.py`
- `python scripts/run_simulation_benchmark.py --datasets all --n_seeds 10`
- `python scripts/run_application_geometry_report.py`
- `python scripts/run_parameter_sweep.py --quick`

Generated reports:

- `reports/simulation_benchmark/index.html`
- `reports/simulation_benchmark/simulation_results_long.csv`
- `reports/simulation_benchmark/simulation_results_summary.csv`
- `reports/application_geometry/index.html`
- `reports/application_geometry/application_results_long.csv`
- `reports/application_geometry/application_results_summary.csv`
- `reports/parameter_sweep/index.html`
- `reports/parameter_sweep/parameter_sweep_results.csv`
- `reports/parameter_sweep/parameter_sweep_summary.csv`

Simulation datasets: `flat_rotation`, `flat_spiral`, `flat_saddle`,
`s_curve`, `swiss_roll`, `half_sphere_rotation`, and
`potential_saddle_surface_saddle`; 10 seeds, `n_samples=120`.

Key simulation findings from `simulation_results_summary.csv`:

- On `flat_rotation`, `flat_spiral`, and `flat_saddle`, PCA-rank-d had the
  best RMSE and clean-cloud distance. This is expected because the true
  synthetic manifold is globally linear, making PCA-rank-d close to an oracle
  baseline.
- On `s_curve`, PCA-rank-d and `VelocityManifoldFitter` were nearly tied on
  RMSE (`0.5613` vs `0.5616`), with PCA-rank-d better on clean-cloud distance
  (`0.3374` vs `0.4146`). In this setting, VMF does not clearly outperform PCA
  baselines under the current parameter choices.
- On `swiss_roll`, PCA-rank-d beat VMF on RMSE (`0.4868` vs `0.5308`) and
  clean-cloud distance (`0.2999` vs `0.3887`). In this setting, VMF does not
  clearly outperform PCA baselines under the current parameter choices.
- On `half_sphere_rotation`, VMF had the best RMSE (`0.5405` vs PCA-rank-d at
  `0.5588`), but PCA-rank-d had the best clean-cloud distance.
- On `potential_saddle_surface_saddle`, VMF had the best RMSE (`0.5228` vs
  PCA-rank-d at `0.5684`), while PCA-rank-d had a slightly better clean-cloud
  distance (`0.3377` vs `0.3461`).
- PCA-rank-d had the best normal energy and velocity-tangent alignment on every
  simulation because it forces a global rank-d reconstruction. This should not
  be interpreted as true nonlinear manifold recovery, because PCA-rank-d can
  reduce normal energy by collapsing data into a global linear rank-d subspace.
- VMF movement cost was moderate across simulations, typically around
  `0.51-0.54` local scales on average; PCA-rank-d moved points more on the flat
  and nonlinear simulations, around `0.63-0.66` local scales.

Application findings from `application_results_summary.csv`:

- On sampled cell-cycle data, VMF reduced normal energy ratio from `0.7071`
  to `0.5443` and improved velocity-tangent alignment from `0.3802` to
  `0.7044`.
- VMF improved velocity-tangent alignment more than PCA-rank-10 (`0.7044` vs
  `0.5434`), while PCA-rank-10 had similar normal energy (`0.5605`) but larger
  movement (`0.6721` local scales).
- VMF mean displacement was `0.5834` local scales and kNN overlap was `0.4506`.
  VMF improves velocity-geometry compatibility, but this comes with a movement
  and neighborhood-preservation tradeoff.

Parameter sweep findings from `parameter_sweep_summary.csv`:

- Quick sweep date: 2026-06-22 00:50:51 CST. Scope was
  `s_curve`, `swiss_roll`, and `half_sphere_rotation`, 2 seeds,
  `n_samples=100`, `eta_g in {0.2, 0.35, 0.5}`,
  `theta in {0.05, 0.15, 0.3}`, `k in {15, 25}`, `T=3`, and
  adaptive thresholds `{0.8, 0.9}`.
- On `s_curve`, best velocity-tangent alignment was `0.7561` with
  `eta_g=0.5`, `theta=0.05`, `k=25`, `T=3`, threshold `0.8`; this moved
  points `0.4399` local scales with kNN overlap `0.7232`. The balanced setting
  was `eta_g=0.35`, `theta=0.3`, `k=25`, `T=3`, threshold `0.8`.
- On `swiss_roll`, the same high-utility setting `eta_g=0.5`, `theta=0.3`,
  `k=25`, `T=3`, threshold `0.8` was also the best balanced and best RMSE
  setting, with alignment `0.8535`, displacement `0.4455`, and kNN overlap
  `0.7220`.
- On `half_sphere_rotation`, best alignment used `eta_g=0.5`, `theta=0.3`,
  `k=25`, `T=3`, threshold `0.8`, with alignment `0.8492`, displacement
  `0.4484`, and kNN overlap `0.7312`. The balanced setting reduced step size
  to `eta_g=0.35` while keeping `theta=0.3`, `k=25`, `T=3`, threshold `0.8`.
- Lowest movement and best kNN preservation generally came from
  `eta_g=0.2`, `theta=0.05`, `k=15`, `T=3`, threshold `0.8`, but these settings
  had worse RMSE and lower alignment. This confirms the expected
  utility-versus-movement tradeoff.

Failed or skipped cases:

- Integrity checks passed.
- No simulation or cell-cycle application method failures were recorded.
- Palantir and P450 protein landscape were skipped because their external data
  files were not available in this checkout.
- Quick parameter sweep completed with no failed settings.

Current issues and next actions:

- Expand the parameter sweep to more seeds and include `T=5` before making
  stronger claims about VMF.
- Add multi-seed biological sanity checks for cell-cycle phase ordering and
  movement size.
- Treat PCA-rank-d as a strong, partly oracle-like synthetic baseline whenever
  the intrinsic dimension is known.

### Known Limitations

- Real data do not provide a true clean manifold, so application metrics are
  diagnostic rather than ground-truth utility.
- Velocity estimates may be noisy; velocity-aware metrics can reward aggressive
  smoothing if movement and neighborhood preservation are not checked.
- Global PCA is a strong baseline in approximately linear embeddings and can be
  oracle-like in flat synthetic data when the intrinsic dimension is known.
- Local geometry metrics depend on the neighborhood size and can favor
  low-rank collapse unless paired with reconstruction, displacement, and
  preservation metrics.
- The scalar-potential branch remains experimental; default benchmarks focus on
  stable vector-field simulations and the committed cell-cycle data.
