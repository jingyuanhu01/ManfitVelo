# Simulation cleanup manifest

Audit date: 2026-08-03.

## Formal entry points

- `python simulation/run_manfitvelo_benchmark.py [--report-only]`
- `python simulation/run_sphere_scalability.py [--report-only]`

The audit used `git status --short`, recursive file listings under `simulation`,
`scripts`, `tests`, and `results`, and `rg` searches for imports, command names,
output paths, GraphVelo tuning, matched/refined pipelines, MMLS, and PCA
variance-selection variants. User changes unrelated to these formal simulations
were not modified.

## Deleted after reference audit

- `simulation/run_graphvelo_cosine_benchmark.py`: superseded runner containing
  zero/default/double noise output and GraphVelo grid tuning.
- `simulation/run_unified_state_benchmark.py`: superseded report/metric runner;
  its still-required evaluation utilities were consolidated into
  `simulation/benchmark_core.py`.
- `simulation/test_unified_state_benchmark.py`: tested the deleted runner; its
  relevant invariants are now covered by the formal main/scalability tests.
- `scripts/graphvelo_baselines.py`: internal approximate/tunable GraphVelo core.
  Formal GraphVelo now uses the pinned official analytical-manifold adapter.
- `simulation/__pycache__` and `scripts/__pycache__`: generated caches only.

These working-tree files were untracked, so their deletion is not recoverable
through this repository's Git history. Their formal functionality is retained
in the new entry points and shared modules.

## Consolidated or added

- `scripts/graphvelo_official_adapter.py`: one pinned GraphVelo notebook call
  path plus provenance, the exact raw-scale wrapper, and one fixed truth-free
  standardized wrapper. Raw and oracle-rescaled results remain diagnostics;
  only standardized GraphVelo enters the formal seven-method table.
- `scripts/simulation_baselines.py`: shared cosine, Global PCA, Local PCA, and
  downstream velocity wrappers.
- `simulation/benchmark_core.py`: shared frozen-config and identity/location/
  Euler evaluation utilities.
- `simulation/run_manfitvelo_benchmark.py`: sole main benchmark/report entry.
- `simulation/run_sphere_scalability.py`: sole ambient-dimension scalability
  entry.
- `simulation/README.md`: formal design and reproduction documentation.

## Deliberately retained

- `scripts/pca_denoisers.py` still contains variance-threshold and oracle PCA
  helpers because independent application/legacy experiments import them.
  Neither formal entry point exposes or calls PCA-90/PCA-95/oracle selection.
- Historical report directories and prior benchmark result directories remain
  as audit artifacts. Formal commands do not read them.
- `simulation/run_velocity_augmented_main_benchmark.py`, Sasaki diagnostics,
  and tangent-scaling tests are independent historical/mechanism studies and
  were not proven redundant with the two formal entry points.
- Core ManfitVelo, data generators, and unrelated user files were retained.

Final `rg` audit finds no reference from active formal code to the deleted
runners or approximate GraphVelo module, and no GraphVelo grid/tuning or
matched/refined pipeline in the two formal call graphs.
