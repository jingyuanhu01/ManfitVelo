# ManfitVelo Simulation Protocol

Frozen-protocol reference for the two formal entry points (`run_manfitvelo_benchmark.py`,
`run_sphere_scalability.py`). Written for someone picking this up cold — for *how we got here* and
*why specific choices were made*, see `log.md` (chronological); for exact frozen numeric values, see
`methods_config.yaml` / `scenario_config.yaml`; for metric formulas, see `metric_definitions.md`; for
the neighborhood-rule mathematics, see `parameter_rules.md`.

## 1. What this answers

Four questions (Weekly Plan v1.1 §15), each mapped to a specific method comparison:

| # | Question | Comparison |
|---|---|---|
| Q1 | Does velocity information improve manifold recovery? | Position-only MANFIT (M5) vs. ManfitVelo (M6) |
| Q2a | Does using manifold information at all improve velocity recovery? | Cosine Kernel / GraphVelo (M1/M2) vs. ManfitVelo |
| Q2b | Given manifold info is used, which fitting strategy recovers velocity best? | Local PCA (M4) vs. Position-only MANFIT (M5) vs. ManfitVelo (M6) |
| Q3 | Does joint recovery improve dynamics? | One-Step Flow Error across all methods |
| Q4 | In what regimes does ManfitVelo help or fail? | Stress-test scans (n, σ_X, σ_V, D) — **not yet implemented**, see §5 |

Explicit non-goal: "ManfitVelo wins everywhere." The target finding is a clearly identifiable regime
(noisy position, informative velocity) where it helps — see the Curved-Hairpin / Near-Intersection
finding in `log.md` for the current best evidence toward Q1.

## 2. Methods (7 total; M3 replaces the original Global PCA baseline)

| ID | Method | Updates x̂? | Updates v̂? | Tuned? |
|---|---|---|---|---|
| M0 | Ambient Noisy Input | no | no | reference only, never ranked |
| M1 | GraphVelo | no | yes | official algorithm untouched; input rescaled by a fixed truth-free rule |
| M2 | Cosine Kernel | no | yes | k(n,d) only |
| M3 | Joint Low-Rank | yes | yes | fixed 0.90 variance threshold |
| M4 | Local PCA | yes | yes | k(n,d) only |
| M5 | Position-only MANFIT | yes | yes | k(n,d) + shared (T, η_g) |
| M6 | ManfitVelo | yes | yes | k(n,d) + shared (T, η_g, θ, κ, θ_schedule) |

M1/M2 never change x̂, so their G1/G2 geometry metrics equal M0 by construction (see
`metric_definitions.md` §A) and are not separately ranked. Full per-method configuration:
`methods_config.yaml`.

**Core fairness principle** (Weekly Plan §4): every method has exactly one fixed
parameter-selection rule applied identically across all scenarios. No scenario ever gets its own
hand-picked constant. Where a rule is itself a function of the data (k(n,d), the curvature-aware
refinement, M3's rank threshold), it is *allowed* to output different numbers on different
scenarios — that's what makes it "data-adaptive" rather than "scenario-specific tuning."

## 3. Scenarios (9 total)

**Group A — regular smooth-manifold benchmarks** (primary comparison regime):
Circle, S-curve (1D); Flat Rotation Annulus, Half-sphere-tangent, Swiss Roll, Saddle Surface (2D).
Deliberately spans curvature sign/magnitude: Flat Rotation Annulus is exactly flat (0 curvature),
Half-sphere-tangent and Swiss Roll are positively/extrinsically curved, Saddle Surface has
negative/mixed Gaussian curvature.

**Group B — geometry stress tests** (small reach / ambiguous neighborhoods):
Curved Hairpin, Near Intersection.

**Group C — out-of-assumption robustness**:
Y-branch (non-smooth branch point; excluded from geometry metrics within a small radius of the branch).

Full generative formulas: `scenario_config.yaml`.

## 4. Seeds

- **Tuning seeds** `42000–42002`: development only — geometry selection, parameter grid search,
  curvature-aware k calibration. Never scored in the final report.
- **Final seeds** `43000–43014` (15 total): scored once, after every configuration is frozen. Never
  used for any selection (`selection_uses_final_seeds` / `final_seeds_used_for_selection` asserted
  `False` throughout the pipeline and re-checked by `validate()` at report-build time).

## 5. What's in scope vs. deferred

**In scope / implemented**: M0–M6 comparison on the canonical `(n₀, σ_X0, σ_V0, D₀)` setting for all
9 scenarios; curvature-aware k(n,d) neighborhood rule; `run_sphere_scalability.py`'s ambient-dimension
scan (D∈{3,5,10,20,50}) holding intrinsic geometry fixed.

**Explicitly deferred** (not this round):
- Stress-test sweeps Scan A (sample size n), Scan B (position noise σ_X — flagged in the Weekly Plan
  as one of the most important, since it's the direct test of Q1), Scan C (velocity noise σ_V). When
  implemented, **k(n,d) and its curvature-aware refinement must be recomputed fresh at every scan
  point from that point's own development-seed draws** — never reuse the canonical setting's frozen
  k, since the whole rule is defined to be a function of (n, d, and the observed noise/curvature
  signal), not a constant. See `parameter_rules.md` §Adaptivity.
- Phase-diagram experiment (Δ = E_ManifoldOnly − E_ManfitVelo over a (σ_X, σ_V) grid).
- Δt sensitivity check (Curved Hairpin), paired Wilcoxon signed-rank secondary test for the M5-vs-M6
  ablation.
- A dedicated failure-rate reporting protocol (currently just a lightweight `nan_inf_count` column;
  worth expanding once high-noise stress sweeps actually risk numerical failures).

## 6. Reproducibility

```bash
python simulation/run_manfitvelo_benchmark.py     # ~15-20 min; regenerates results/manfitvelo_benchmark/
python simulation/run_sphere_scalability.py       # ~1 min; regenerates results/sphere_scalability/
```

Both are single-command, deterministic given the fixed seeds; each writes its own
`environment_provenance.json` (package versions) and `sanity_checks.json` (self-audit — every run
must show `all_checks_pass: true`). Per-seed raw metrics (`final_seed_metrics.csv` /
`seed_metrics.csv`) and every frozen hyperparameter (`selected_hyperparameters.json`) are saved
alongside the summarized report so every figure can be regenerated from raw CSVs without rerunning
the fits (`--report-only`).

## 7. Limitations

- **Intrinsic dimension `d` is assumed known** by every manifold-fitting method (M3–M6). Real
  single-cell data needs `d` estimated. Sensitivity to `d` misspecification is out of scope for this
  round — a candidate independent follow-up experiment, not a stress-test axis here.
- **GraphVelo failure modes are not separately audited.** It is a vendored, unmodified official
  implementation (see `scripts/graphvelo_official_adapter.py`'s provenance block) — its own numerical
  behavior is not the object of this study, so beyond the shared `nan_inf_count` column it isn't
  given dedicated failure-reporting space here.
- **Curvature-aware k(n,d) is a heuristic, not exact.** It reliably distinguishes flat from curved
  geometry and improves on the plain formula in aggregate, but is not perfectly calibrated per
  scenario (some residual gap vs. the old scenario-specific hand-tuned numbers remains on
  Half-sphere-tangent) — see `log.md` Round 2 for the validation data and the honest gap.
