# ManfitVelo

Velocity-aware manifold fitting for simulation and RNA-velocity workflows.

The main implementation is `VelocityManifoldFitter` in
`scripts/velocity_manifold_fitter.py`. It takes a state matrix `Y` and matching
velocity matrix `W`, builds a velocity-aware neighbor graph, estimates local
tangent spaces with weighted PCA, projects velocities onto those tangent spaces,
and updates points with a normal-only manifold correction by default.

## Repository structure

| Path | What it is |
|---|---|
| `scripts/` | The method implementation: the core `VelocityManifoldFitter` class, its scalar-gradient variant, baseline comparators, and the shared scenario library the `simulation/` suite is built on. See below. |
| `simulation/` | The paper-facing experiment suite (P0–P5 frozen protocol): canonical benchmark, robustness scans, controlled vector/scalar-field experiments, significance testing, and the consolidated report generator. See below. |
| `archive/` | Retired/superseded code and old result snapshots — **not** part of the active protocol; kept for provenance and easy recovery, not for reuse. Gitignored (not part of the delivered repo). |
| `results/` | Generated outputs from running `simulation/` scripts — reappears automatically when you run them. Gitignored. |
| `notebooks/` | Interactive usage examples; see the Notebooks section below (two are currently stale). |
| `data/` | Small example datasets (e.g. `data/cell_cycle/`) used by notebooks. |
| `docs/` | Sphinx API documentation source. |

### `scripts/` — method implementation

| File | Role |
|---|---|
| `velocity_manifold_fitter.py` | Core `VelocityManifoldFitter` algorithm (see Quick Start below). |
| `scalar_potential_manfit.py` | Scalar-gradient analog: estimates a gradient field from noisy scalar observations and fits it jointly via `VelocityManifoldFitter`, the same way real velocity is used. |
| `benchmark_scenarios.py` | Shared scenario-generator/fitting-variant library most of `simulation/`'s active scripts import from (`vector_data`, `scalar_data`, `hairpin`, `fit_vmf_variant`, the position-only M5 baseline `position_only_trajectory`, evaluation helpers). Despite the name, this is a library, not a one-off script. |
| `simulation_baselines.py` | Shared baseline pipelines (cosine-kernel, Global/Local PCA, Joint Low-Rank denoising, downstream velocity reconstruction) used across the benchmark suite. |
| `graphvelo_official_adapter.py` | Vendored, pinned port of the official GraphVelo package's numerical core (see the file's own docstring for provenance/license) — the M1 baseline. |
| `pca_denoisers.py` | Global/local PCA denoising primitives used by several baselines. |
| `ambiguity_simulations.py` | Synthetic Y-branch flow generator. |
| `html_report_utils.py` | Small self-contained HTML report helper. |

### `simulation/` — experiment suite

Formal entry points and shared infrastructure are described in the Simulation Benchmark Suite section below; `simulation/README.md`, `simulation/current_plan.md`, `simulation/log.md`, and `simulation/parameter_rules.md` are the authoritative reference docs (not duplicated here).

## Quick Start

Install dependencies (see `requirements.txt`):

```bash
pip install -r requirements.txt
```

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
- `lambda_v`: strength of the trace-normalized local velocity covariance blended
  into tangent estimation (`C = C_position + lambda_v * C_velocity`). The class
  default is `0.0` (velocity-free tangent estimation, i.e. position-only
  behavior). This is the parameter that actually lets velocity information
  improve manifold recovery rather than only reweighting neighbors or
  transporting tangential velocity; the `simulation/` benchmark suite currently
  freezes it at `1.0` for directly-observed velocity after a dedicated
  re-selection round (see `simulation/parameter_rules.md` §3a), and at `0.0`
  for the scalar-gradient branch after a separate selection (§3b–§3c). Callers
  who want the velocity-aware tangent behavior described in
  `simulation/methods_config.yaml` must set it explicitly.

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
- `velocity_covariance_mode`: how the local velocity covariance is built before
  blending into tangent estimation — `"centered"`, `"uncentered"` (the current
  `simulation/` default), or `"covariance_plus_mean"`. The latter two are
  algebraically equivalent under common weights and are kept mainly as explicit
  audit labels.
- `velocity_trace_normalization`: normalization applied to the velocity
  covariance before combining it with the position covariance. Only
  `"match_position_trace"` is currently supported.
- `lambda_v_confidence_scaling` / `lambda_v_confidence_power` /
  `lambda_v_relative_error`: optionally discount `lambda_v` per point by
  confidence/fitting-error before it enters the covariance blend —
  `"none"` (default, bit-identical to not having this option at all),
  `"linear"`, `"power"`, `"inverse_error"`, `"rank"`. Added for the
  scalar-gradient pipeline, where per-point confidence varies more than for
  directly-observed velocity; see `VelocityManifoldFitter`'s own docstring
  for each mode's formula and `simulation/parameter_rules.md` §3b–§3c for
  which ones are actually frozen for use.
- `record_tangent_diagnostics` / `return_tangent_diagnostics`: save pointwise
  covariance spectra and matrices at every tangent update. Intended for
  synthetic mechanism diagnostics (see `simulation/`); disabled by default to
  avoid quadratic-in-ambient-dimension storage in ordinary application runs.

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

Reference notebooks are kept in `notebooks/reference_notebooks/`. The older
standalone implementations they and the two "Position + Potential
Experiments" notebooks below historically imported (`scripts/manfit.py`,
`scripts/manfit_ours.py`, `scripts/reference_implementations/`) are no
longer present under `scripts/` (see `archive/scripts/`); those two
notebooks will currently fail on import until ported to a surviving module.

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

Both notebooks currently import `scripts.manfit_ours`/`scripts.manfit`, which
are not present under `scripts/` (see the Notebooks section above); treat
these two notebooks as stale until they are ported to a surviving module.
`scripts/scalar_potential_manfit.py` is a separate, currently maintained
scalar-field implementation used by the `simulation/` suite's
scalar-benchmark scaffolding (see below) rather than by these notebooks.

The protein data come from the Nature Communications paper:

> Ding, X., Zou, Z. & Brooks III, C.L. Deciphering protein evolution and
> fitness landscapes with latent space models. Nature Communications 10, 5644
> (2019). https://doi.org/10.1038/s41467-019-13633-0

The repository does not commit the downloaded data files. The script that
previously prepared local processed files from the paper's supplementary data
(`scripts/prepare_protein_latent_paper_data.py`) is no longer present under
`scripts/` (see `archive/scripts/`); reproducing the protein notebook
currently requires restoring that script or writing an equivalent one.

## Simulation Benchmark Suite

The primary experimental and paper-facing work lives under `simulation/`, not
under the older `reports/` pipeline described in earlier versions of this
README — that pipeline has been moved in full to `archive/`; see
`code_cleanup_manifest.md` for the audit that did it.

The suite compares seven methods — Ambient noisy input (M0), GraphVelo (M1),
Cosine kernel (M2), Joint Low-Rank denoising (M3), Local PCA (M4),
Position-only MANFIT (M5), and ManfitVelo (M6) — across nine canonical
scenarios (Circle, S-curve, Flat Rotation Annulus, Half-sphere Tangent, Swiss
Roll, Saddle Surface, Curved Hairpin, Near Intersection, Y-branch), with 15
final evaluation seeds per scenario and a strict separation between tuning
seeds (used for parameter selection) and final seeds (used only for reporting).
Beyond the canonical benchmark, the suite includes ambient-dimension
scalability, controlled vector-field experiments (same manifold/different
fields, same dynamics/different manifolds) and their scalar-gradient analogs,
paired significance testing, and a consolidated report — see
`simulation/current_plan.md` for the full P0–P5 plan and results.

Formal entry points:

```bash
# Core canonical benchmark
python simulation/run_manfitvelo_benchmark.py          # canonical 9-scenario benchmark
python simulation/run_manfitvelo_benchmark.py --report-only

python simulation/run_sphere_scalability.py             # ambient-D scalability (S^2 in R^D)
python simulation/run_sphere_scalability.py --report-only

python simulation/run_stress_scans.py                    # Scan A/B/C: sample size, position noise, velocity noise
python simulation/run_lambda_sensitivity.py               # vector-field lambda_v selection
python simulation/run_wilcoxon_test.py                    # M5-vs-M6 significance test

# Baseline-fairness / parameter-selection audits
python simulation/run_c_selection.py                      # global k(n,d) constant selection
python simulation/run_half_sphere_diagnosis.py             # half-sphere diagnosis
python simulation/run_joint_low_rank_threshold_sensitivity.py  # M3 threshold sensitivity
python simulation/run_manifold_dimension_scalability.py    # Circle/Saddle ambient-D scalability

# Controlled vector-field experiments (P3)
python simulation/run_v1_field_family.py                   # same manifold, different vector fields
python simulation/run_v2_manifold_family.py                 # same dynamics, different manifolds

# Scalar-gradient branch (P4)
python simulation/run_scalar_lambda_v_selection.py          # scalar-branch lambda_v/scaling selection
python simulation/run_p4_1_scalar_oracle_ablation.py        # oracle vs estimated gradient
python simulation/run_s1_scalar_landscape_family.py         # same manifold, different scalar landscapes
python simulation/run_s2_manifold_landscape_family.py        # same landscape, different manifolds

# Consolidated report
python simulation/build_experiment_report.py              # consolidated HTML report
```

`simulation/run_dt_sensitivity.py` (Euler-step tau sensitivity) predates the
current global-k(n,d) rule and is retired to `archive/simulation/` — not
part of the current protocol.

Run the test suite:

```bash
pip install -r requirements.txt
python -m pytest -q simulation
```

Each script above writes to a like-named directory under `results/`
(gitignored, reappears when you run the script); `results/experiment_report/`
is the consolidated report pulling all of them together.

Reference documentation for the suite (design, frozen protocol, and full
history):

- `simulation/README.md` — research question, algorithms, scenarios, metrics,
  reproduction instructions.
- `simulation/methods_config.yaml`, `simulation/scenario_config.yaml`,
  `simulation/parameter_rules.md`, `simulation/metric_definitions.md`,
  `simulation/simulation_protocol.md` — frozen, human-readable snapshots of the
  current protocol.
- `simulation/history.md` — condensed summary of how the protocol reached its
  current frozen state.
- `simulation/log.md` — full chronological experiment log, including
  debugging and false starts.
- `simulation/current_plan.md` — the forward-looking P0–P5 experiment plan
  and its results, updated in place as each phase completes.

## License

Not yet added. Decide and add a `LICENSE` file before making the repository
public.
