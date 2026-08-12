# Parameter Rules

The complete "how is every number chosen" reference. Companion to `methods_config.yaml` (the frozen
output values) — this document is the *rule*, that file is the *result of applying the rule once*.

## 1. The fairness principle (Weekly Plan v1.1 §4)

> Each method has one fixed parameter-selection rule across all scenarios.

Priority order for any parameter, highest first:

1. **Official default** (e.g. GraphVelo's own published defaults — never touched).
2. **Data-adaptive rule** — a deterministic function of the *observed* data (n, d, the noisy
   observations themselves). Allowed to output different numbers on different scenarios/draws,
   because it's the same *rule*, not a different *choice* per scenario.
3. **Once-for-all development tuning** — a grid search scored on development (tuning) seeds,
   *pooled across all scenarios*, picking one winner applied everywhere.
4. **Scenario-specific tuning** — hand-picking a value because it looks best on one particular
   scenario. **Forbidden in final experiments.**

The distinction that matters: tier 2 and 3 both apply the *same procedure* everywhere; tier 4 applies
a *different procedure* (or an ad hoc choice) per scenario. A rule that legitimately outputs
`k=14` on Curved Hairpin and `k=40` on Flat Rotation Annulus is tier 2, not tier 4, as long as both
numbers came out of running the identical formula/procedure on that scenario's own data.

## 2. Neighborhood size — k(n, d), curvature-refined

Applies to Cosine Kernel (M2), Local PCA (M4), Position-only MANFIT (M5), ManfitVelo (M6). Two
stages, both tier 2 (data-adaptive, no ground truth used by either stage):

### Stage 1 — base formula

MISE-optimal local-linear bandwidth scaling `h_n ~ n^{-1/(d+4)}` implies

```
k(n, d) = clip(ceil(C · n^(4/(d+4))), 10, 200)
```

**`C = 0.60`, a single dimension-independent constant (updated 2026-08-11→2026-08-12, current_plan.md
P0.1)**. Originally two separate constants `C_1 ≈ 0.3606` (Circle, `n₀=360`, analytically calibrated
so `k(360,1)=40`) and `C_2 ≈ 0.7132` (Flat Rotation Annulus, `n₀=420`, `k(420,2)=40`) — reverse-
engineered anchor points, one per intrinsic dimension. Replaced by a single global `C`, selected by
`simulation/run_c_selection.py`: candidates `C ∈ {0.30, 0.45, 0.60, 0.75, 0.90}` (spanning and
straddling the two old anchors), each candidate's per-scenario `k` derived by the same two-stage
procedure documented in this section (Stage 1 ceiling from that candidate `C`, then the unchanged
Stage 2 below), scored by ManfitVelo (M6)'s pooled `tuning_score` on **tuning seeds only**, mean over
all 9 canonical scenarios (other VMF/Position-only hyperparameters held at their then-current frozen
values, not re-tuned per candidate). `C=0.60` won clearly (`results/c_selection/
c_selection_summary.csv`); Position-only MANFIT's own pooled `tuning_score` prefers `C=0.45` by a
negligible margin (not part of the selection rule — M6 is primary, matching how §3's grid is scored).

**Known limitation discovered during this selection (accepted 2026-08-12, not fixed — see Stage 2
below for the mechanism)**: at the two losing higher-`C` candidates (0.75, 0.90), Stage 2's turn
detection failed on Curved Hairpin specifically, returning `k` at the Stage-1 ceiling (105/126)
instead of a genuine early optimum. Confirmed this does not affect the `C=0.60` winner (or 0.30/0.45)
on any scenario (`results/c_selection/c_selection_k_table.csv`), so it doesn't change the outcome —
only inflates by how much the two already-losing candidates lose by. Left as-is rather than fixed,
since a fix touches Stage 2's shared turn-detection rule for every scenario, not just this choice of
`C`, and was out of scope for P0.1.

This stage alone was applied identically to all 9 canonical scenarios with **no exception for
Curved Hairpin or Near Intersection**, by explicit decision — see `log.md` for the discussion of why
Hairpin's existing purely-geometric reach diagnostic (`hairpin_reach_diagnostics()`) was *not* used
to override this rule.

### Stage 2 — curvature-aware refinement

Stage 1 only sees `(n, d)`; it can't distinguish a flat scenario from a curved one, so it
systematically overshoots the bias/variance-optimal k on curved geometry (root-caused on
Half-sphere-tangent — see `log.md` Round 2). Stage 2 refines it using a second, still ground-truth-free
signal:

1. Probe a 14-point geometric grid of candidate k from `max(2d+2, 8)` up to the Stage-1 ceiling
   (never above it).
2. At each candidate k, fit `local_pca_denoise` and take the **normal-direction residual**: the sum
   of the smallest `ambient_dim − d` eigenvalues of the local covariance (how much local spread a
   rank-d tangent plane fails to explain), averaged across all points and across the 3 tuning seeds.
3. On a log(residual) vs. log(k) plot, track the *slope*. It starts high (finite-sample
   eigenvalue-estimation bias, universal), then decreases (more points → better covariance estimate)
   and, **only on genuinely curved geometry**, turns back upward (curvature bias now dominating).
4. The chosen k is the grid point right after the slope's minimum. On a flat manifold the slope never
   turns back up, so this exactly reproduces the Stage-1 ceiling — confirmed on Flat Rotation Annulus
   and (after the winding-count fix, see §4) Swiss Roll's tractable range.

Implementation: `benchmark_core.curvature_probe_k_grid` / `local_pca_normal_residual` /
`curvature_aware_neighbor_count`; orchestration (looping over all 9 scenarios × 3 tuning seeds):
`run_manfitvelo_benchmark.curvature_aware_scenario_k`. Full per-scenario numeric output and
validation against `clean_point_rmse_rel`: `log.md` Round 2, `curvature_aware_k_diagnostics.csv`.

**Known limitation (found 2026-08-12 during the §2 `C` re-selection, not fixed)**: step 4's
"grid point right after the slope's minimum" implicitly assumes the log-log slope curve is unimodal
(one dip, one later rise). At a large enough Stage-1 ceiling this can be false — Curved Hairpin's
slope curve, probed out to a ceiling of 105–126 (only reachable at the losing `C=0.75/0.90`
candidates during selection, not at the frozen `C=0.60`), dips early (normal regime), rises
(curvature bias), then dips again at very large `k` for a reason not yet investigated, and the
argmin-based rule picks that second, spurious dip. Confirmed not to affect any scenario at the
frozen `C=0.60`. Left as a known limitation rather than fixed, since a fix touches every scenario's
turn detection, not just this specific case.

**Rejected alternative**: a forward-tolerance extension past the slope minimum (to rescue scenarios
whose minimum is reached very early). Validated worse in aggregate than the plain "stop at the
minimum" rule — see `log.md` for the comparison.

## 3. ManfitVelo (M6) hyperparameters — tier 3, once-for-all

`T, eta_g, kappa, theta, theta_schedule` (bandwidth_mode fixed at `variable`;
`velocity_covariance_mode=uncentered`, `velocity_trace_normalization=match_position_trace` fixed —
these were already constant across the old per-scenario configs; `lambda_v` handled separately, §3a).
Grid: `T∈{3,5,8} × eta_g∈{0.35,0.5,0.7} × kappa∈{0,1,2} × theta∈{0.02,0.05,0.1} ×
theta_schedule∈{flat,ramp}` (162 candidates). Each candidate scored by mean `tuning_score`
(log-mean of 4 relative metrics — see `run_manfitvelo_benchmark.tuning_score`) pooled over **all 9
scenarios × 3 tuning seeds** (equal weight per scenario-seed pair); winner frozen for every final
scenario/seed. `k` is excluded from this grid (supplied by §2). Implementation:
`run_manfitvelo_benchmark.shared_vmf_grid` / `tune_shared_vmf`.

### 3a. lambda_v — tier 2/3 hybrid, selected separately (updated 2026-08-11)

`lambda_v` controls how much weight the velocity second-moment matrix gets when blended into the
local position covariance before the tangent-space eigendecomposition
(`VelocityManifoldFitter._compute_local_tangent`) — the one parameter that actually implements "use
velocity to improve manifold recovery." It is deliberately **not** included in §3's 162-candidate
grid, to keep its selection procedure and audit trail separate and legible
(`run_lambda_sensitivity.py`) rather than conflated with the other five parameters.

**History**: originally a carried-forward constant, `lambda_v=0.1`, inherited unchanged from every
old per-scenario config, which in turn came from a dedicated prior study
(`archive/simulation/results_legacy/velocity_augmented_main_benchmark_20260717/`) that swept
`lambda∈{0,0.1,0.25,0.5,1.0,2.0}` on the pre-fairness-fix, pre-curvature-aware-k, 7-scenario
protocol and picked 0.1 as a deliberately conservative choice.

**Re-selection under the current protocol**: re-swept the same grid on **tuning seeds only**
(42000–42002 — final seeds never enter this computation), pooling each candidate's score as the
mean of `log(clean_point_rmse_rel) + log(distance_to_manifold_rel) + log(velocity_rmse_loc_rel) +
log(joint_euler_state_rmse_rel)` across all 9 scenarios × 3 tuning seeds — position, *location*-
anchored velocity (not identity-anchored — matches this report's own headline metric family, see
`metric_definitions.md` §B2), and joint flow. The naive pooled argmin was `lambda_v=2.0`, essentially
tied with `1.0` (~1% apart), but `2.0` makes Swiss Roll's `clean_point_rmse_rel`/
`distance_to_manifold_rel` *worse than its own lambda_v=0 baseline* while gaining almost nothing
further on `velocity_rmse_loc_rel` over `1.0` (0.916 vs 0.917) — a Pareto-inefficient trade, not a
genuine improvement. **Selected: `lambda_v=1.0`.** A confirmatory sensitivity curve on all 15 final
seeds (reporting only, this curve is not itself a selection step) is in
`results/lambda_sensitivity_final/`. Full audit: `results/lambda_sensitivity_tuning/`
(`lambda_selection_audit.csv`, `lambda_seed_metrics.csv`).

### 3b. Scalar-branch lambda_v / lambda_v_confidence_scaling — tier 3, selected separately (added 2026-08-12)

The scalar-gradient pipeline (`scripts/scalar_potential_manfit.fit_scalar_gradient_manfit`) reuses
`VelocityManifoldFitter` with an estimated gradient standing in for velocity, so it inherits the same
`lambda_v`/`lambda_v_confidence_scaling` mechanism as §3a — but §3a's `lambda_v=1.0` was tuned for
*directly observed* velocity and was never itself selected for a *noisily estimated* gradient, whose
per-point reliability varies far more (see `current_plan.md` P4.1).

**Scaling mode**: fixed to `"rank"` by direct user decision (2026-08-12, not a data-driven selection)
— `lambda_v_effective_i = lambda_v * (1 - percentile_rank(relative_error_i))`, purely ordinal, zero
free parameters (see `VelocityManifoldFitter`'s own docstring for the full family comparison against
`"power"`/`"inverse_error"`, none of which was frozen).

**lambda_v magnitude**: selected via a small tuning-seed grid search mirroring §3a's own procedure
(`run_scalar_lambda_v_selection.py`) — swept `{0.0, 0.5, 1.0, 2.0, 4.0}` on `scalar_s_curve` +
`scalar_saddle`, **tuning seeds only** (42000–42002), pooled score = mean of
`log(clean_point_rmse)` across scenarios, safeguard: no candidate may regress a scenario's own
`clean_point_rmse` below its `lambda_v=0` baseline. Result: pooled score increases monotonically with
`lambda_v` (every candidate above 0 fails the safeguard on `scalar_saddle`) — **selected:
`lambda_v=0.0`**. In other words: even with the "rank" scaling mechanism, blending the estimated
gradient's covariance into the tangent estimate does not currently help on the one real curved scalar
scenario available, under today's local-regression gradient-estimation quality — a legitimate,
honestly-reported outcome, not a workaround-in-progress. `scalar_s_curve` (degenerate `z≡0` geometry)
contributes no signal either way, as established throughout P4.1.

Practically: `fit_scalar_gradient_manfit`'s frozen-protocol default for S1/S2 is
**`lambda_v=0.0`** (`lambda_v_confidence_scaling` is then moot — `"rank"` is a no-op at `lambda_v=0`).
Full audit: `results/scalar_lambda_v_selection/` (`tuning_seed_grid.csv`, `tuning_seed_selection_audit.csv`,
`final_seed_confirmation.csv` — the last is confirmatory only, not a selection input).

**[UPDATE 2026-08-12, see §3c below]**: this selection was originally run under
`fit_scalar_gradient_manfit`'s own function defaults (`inner_T=2, eta_g=0.35, theta=0.2, kappa=2.0`),
not any frozen shared value. §3c adopts the vector-field's frozen `theta`/`kappa` (0.02/0.0) — but NOT
its `T`/`eta_g` (3/0.7), which was tried first and found to overshoot badly — which required rerunning
this selection for self-consistency; the files above now reflect that rerun (`inner_T=2, eta_g=0.35`
unchanged, `theta=0.02, kappa=0.0`), not the original.

### 3c. Scalar-branch theta / kappa — reused from the vector-field's frozen values; T / eta_g deliberately NOT reused (2026-08-12, user decision + empirical correction)

`fit_scalar_gradient_manfit` has its own `inner_T`/`eta_g`/`theta`/`kappa` parameters (passed straight
into the inner `VelocityManifoldFitter` call each outer iteration), separate from `outer_iterations`
(how many times the gradient is re-estimated from the current fitted geometry — a structural parameter
of the alternating scheme itself, with no vector-field analog). Until this decision they were simply
left at the function's own standalone-usability defaults (`inner_T=2, eta_g=0.35, theta=0.2,
kappa=2.0`) — never selected, never even deliberately chosen for the frozen protocol, just whatever the
function signature happened to default to.

**Original decision**: rather than run a separate tier-3 grid search for the scalar branch, reuse the
vector-field M6's already-frozen shared values directly for all four — `inner_T=3, eta_g=0.7,
theta=0.02, kappa=0.0` (§3's own table).

**Caught empirically before it propagated anywhere**: literally copying `T→inner_T`/`eta_g` overshoots
badly. `fit_scalar_gradient_manfit` calls `fit()` `outer_iterations=4` times, each with `inner_T` steps
— so the *total* position-update budget becomes `4×3=12` steps at M6's own aggressive per-step size,
versus M6's own single `fit()` call using only `1×3=3` steps. On `scalar_saddle`/seed=43000, `lambda_v=0`
(safe baseline): raw noisy input's own `clean_point_rmse` is 0.0518; the *old* scalar defaults reached
0.0250 (clearly better than doing nothing); literally copying `inner_T=3, eta_g=0.7` reached
**0.0587 — worse than not fitting at all**, and not from a few outliers (median per-point error nearly
doubled, 38% of points landed >3x their old error). This is a real structural mismatch (single-shot vs.
nested-loop iteration budgets), not evidence against the "reuse ManfitVelo's settings" idea itself.

**Corrected decision**: reuse only `theta=0.02, kappa=0.0` (the parameters the "reuse ManfitVelo's
settings" consistency argument was actually motivated by — they control the velocity-aware
neighbor-reranking mechanism S1/S2's own headline finding traces the whole `joint_scalar_aware` vs.
`geometry_only` gap to). `inner_T`/`eta_g` stay at `fit_scalar_gradient_manfit`'s own values (2, 0.35).
Re-verified this doesn't overshoot: same seed/scenario, theta/kappa-only change moves the safe baseline
from 0.0250 to 0.0249 — negligible, no instability. `outer_iterations=4`, `gradient_n_neighbors=42`,
`gradient_ridge=5e-2`, `confidence_power=1.0`, `adaptive_variance_threshold=0.85`, `adaptive_d_min=2`
remain at their function defaults (no vector-field analog to borrow from).

**Consequence, handled rather than ignored**: since `theta`/`kappa` did materially change (even though
`inner_T`/`eta_g` didn't), everything computed under the old `theta=0.2, kappa=2.0` needed rerunning for
self-consistency: `run_scalar_lambda_v_selection.py` (§3b, rerun — result unchanged, still
`lambda_v=0.0`), `run_s1_scalar_landscape_family.py`, `run_s2_manifold_landscape_family.py` (both
rerun, current_plan.md P4 Experiment S1/S2 sections updated with the new numbers — patterns held, magnitudes
shifted mildly). **Not rerun**: P4.1's own oracle-gradient ablation (current_plan.md P4.1) — a diagnostic
isolating local-regression vs. joint-fitting error, not part of the frozen protocol; its qualitative
conclusion doesn't depend on this choice. Its numbers are flagged in-place as computed under the prior
`theta=0.2, kappa=2.0` (and the function's own `inner_T=2, eta_g=0.35`, unaffected either way) rather
than silently presented as current.

**lambda_v reselection under `theta=0.02, kappa=0.0`** (`run_scalar_lambda_v_selection.py`, same
tuning-seed grid/safeguard procedure as §3b): result unchanged, **`lambda_v=0.0`** still wins — every
candidate above 0 still regresses `scalar_saddle` below its own safe baseline. Full rerun audit in the
same `results/scalar_lambda_v_selection/` files referenced in §3b (previous run's numbers were
overwritten, not archived under a separate name, since §3b's own text now points to this rerun; the
very first — literal-copy, overshooting — attempt was never written to that directory at all).

## 4. Position-only MANFIT (M5) hyperparameters — tier 3, once-for-all

Same idea, smaller grid: `T∈{3,5,8} × eta_g∈{0.35,0.5,0.7}` (9 candidates), same pooled scoring.
Implementation: `shared_position_only_grid` / `tune_shared_position_only`.

## 5. GraphVelo (M1) — tier 1, untouched

Official defaults only: 15-NN, cosine kernel, density correction, `a=1, b=0, r=1, loss_func=linear`,
no log transform, no PCA preprocessing. The primary reported row additionally applies one fixed,
truth-free unit standardization (median-15-NN-distance / median-noisy-speed) before calling the
unmodified official objective — necessary because the official objective isn't scale-invariant, but
itself never tuned or selected by performance. Raw (unstandardized) output retained as a
sensitivity/provenance diagnostic only.

## 6. M3 Joint Low-Rank — tier 2, data-adaptive

Rank threshold (cumulative explained variance ≥ 0.90) is fixed across all scenarios; the resulting
*rank* is a deterministic function of each sample's own observed singular-value spectrum — see
`methods_config.yaml` for the formula and `scripts/simulation_baselines.joint_low_rank_state`.

## 7. Adaptivity requirement for future stress-test scans

**This is a hard constraint for any future Scan A/B/C implementation** (sample size / position noise
/ velocity noise sweeps): §2's k(n,d) and its curvature-aware refinement must be **recomputed fresh
at every scan point**, using that scan point's own development-seed draws — never frozen once at the
canonical setting and reused across different n or noise levels. The whole point of a *rule* (as
opposed to a *tuned constant*) is that it stays a function of the data; a Scan that reused the
canonical `k` at every n/σ would silently reintroduce scenario(-condition)-specific tuning through the
back door and undermine exactly the comparison the scan is meant to make. §3/§4's shared VMF/
Position-only hyperparameters, by contrast, are genuinely meant to stay frozen across a scan (that's
what "one fixed parameter-selection rule" means for a tier-3 parameter) — only the tier-2 rules in §2
are supposed to re-evaluate per condition.

## 8. Position-noise generation mode — `noise_mode` (added 2026-08-12, current_plan.md P1.2)

Two standard, documented modes, recorded per scenario in `scenario_config.yaml`'s `noise_modes` /
per-scenario `noise_mode` fields — a config schema, not a one-off branch inside a single experiment:

- **`normal_only`** (used by all 9 canonical scenarios, and by
  `run_manifold_dimension_scalability.py`'s Circle/Saddle Surface ambient-dimension experiment): a
  single scalar draw per point along the manifold's own analytic normal direction. Magnitude is
  independent of ambient dimension `D` by construction (a scalar times a unit vector), so embedding
  into higher `D` needs no `D`-dependent noise-scaling rule at all.
- **`isotropic_gaussian`** (used only by `run_sphere_scalability.py`'s S² → R^D experiment): iid
  per-ambient-coordinate Gaussian noise. Under the "fixed_coordinate" regime (per-coordinate variance
  held constant as `D` grows), total noise magnitude grows as `sqrt(D)`.

`noise_mode` only ever describes *position* noise. Velocity noise has always been ambient-isotropic
Gaussian in this codebase regardless of `noise_mode` (see `add_noise()` in
`scripts/benchmark_scenarios.py`) — `run_manifold_dimension_scalability.py` uses the
same "fixed_coordinate" isotropic velocity-noise convention as `run_sphere_scalability.py` even though
its position noise is `normal_only`.

These two ambient-dimension experiments are intentionally scoped to different, non-overlapping
questions rather than duplicating each other (2026-08-12 scoping decision): sphere scalability
isolates "does the isotropic-Gaussian-noise mechanism work at all" (answered once, on a positive-
curvature d=2 manifold); Circle/Saddle Surface scalability isolates "does ambient dimension itself
hurt manifold recovery, independent of noise mode" (a question sphere alone can't cleanly separate
from the noise-mode question). Together they cover d=1 and d=2, and both curvature signs.
