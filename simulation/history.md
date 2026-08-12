# ManfitVelo Simulation — History

Condensed, high-value summary of the work that took the simulation suite from "plan exists on paper"
to "frozen, paper-ready protocol with a consolidated report." For full blow-by-blow detail (including
false starts and how they were caught) see `log.md`; for the living reference docs this work produced,
see the pointer map at the bottom.

## Where this started

`ManfitVelo_Simulation_Weekly_Plan_v1.1.md` described a protocol — M0–M6 methods, 9 scenarios, a
fairness principle ("one fixed parameter-selection rule per method, no scenario-specific tuning"),
metrics, stress-test scans — that the actual code hadn't caught up to. `results/manfitvelo_benchmark/`
already existed and looked like a finished formal benchmark, but on inspection its "frozen"
hyperparameters were hand-tuned differently for every one of the 7 scenarios it covered — the exact
thing the plan's own fairness principle forbids. That gap, found during a routine review, is what
triggered everything below.

## What changed, in order

1. **Fairness-principle fix.** Replaced per-scenario hand-tuned `k`/`T`/`eta_g`/`kappa`/`theta` with:
   a shared `k(n,d) = C_d·n^(4/(d+4))` formula (one rule, calibrated once per intrinsic dimension, no
   scenario exceptions — confirmed even for Curved Hairpin, the scenario most tempted to special-case);
   and pooled once-for-all grid search for `T`/`eta_g`/`theta`/`kappa`/`theta_schedule`, scored on
   development seeds only and applied identically everywhere.

2. **Curvature-aware `k(n,d)` refinement.** The plain formula overshoots on curved geometry (it only
   sees sample size and dimension, not curvature) — root-caused via a controlled ablation after
   Half-sphere-tangent's numbers got *worse than noisy input*. Fix: a second, still ground-truth-free
   signal — sweep `k`, track the local-PCA normal-direction residual's log-log growth rate against
   `log(k)`, stop at the rate's minimum (Lepski-style bandwidth selection). Validated against
   `clean_point_rmse_rel` on all 7 original scenarios: 4 clear wins, 2 exact ties (correctly detects
   the flat/near-flat scenarios and leaves them alone), 2 negligible losses.

3. **M3 Joint Low-Rank Denoising** replaced the old Global PCA baseline (block-normalized joint
   `[X,V]` SVD, rank chosen by a fixed 90% explained-variance threshold, exact affine inverse) — as
   the Weekly Plan specified. **Swiss Roll and Saddle Surface** scenarios were implemented to fill the
   plan's two missing Group-A slots; Swiss Roll's first design (classic 1.5-turn spiral) made *every*
   method, including ManfitVelo, worse than noisy input at any tested neighborhood size — a classic
   Euclidean-kNN-bridges-across-windings failure. Reduced to one full winding, which restored a normal
   bias/variance tradeoff without changing the detection machinery.

4. **Weekend deliverables** (`methods_config.yaml`, `scenario_config.yaml`, `metric_definitions.md`,
   `parameter_rules.md`, `simulation_protocol.md`) written as real, standalone reference files rather
   than a "see the log" pointer — the point of freezing a protocol is having something stable to point
   a reviewer at.

5. **Stress-test scans A/B/C** (sample size, position noise, velocity noise), all 9 scenarios, all 15
   final seeds. Key design rule, now load-bearing for any future scan too: **k(n,d) and its
   curvature-aware refinement are recomputed fresh at every scan point** from that point's own
   development-seed draws; the pooled `T`/`eta_g`/`theta`/`kappa`/`theta_schedule`/`lambda_v` stay
   frozen. Reusing the canonical setting's `k` across different `n`/noise levels would silently
   reintroduce scenario-specific tuning through the back door.

6. **`lambda_v` re-selection** — the round that mattered most for the paper. `lambda_v` (how much the
   velocity second-moment matrix is trusted when shaping the local tangent estimate) is the one
   parameter that actually instantiates "velocity helps manifold recovery," yet it had been a frozen
   0.1 inherited from an older, differently-protocoled study, never covered by the new pooled grid
   search. Investigating it properly took three iterations before it was defensible:
   - A first "reporting-only" sweep (deliberately not meant to change anything) used final seeds,
     since it wasn't supposed to be a selection step — then the decision was made to actually adopt a
     new value, which meant that curve could no longer be used (final seeds may never inform a
     selection anywhere in this pipeline). Caught before acting on it.
   - Redone on tuning seeds only. The naive pooled-best answer (`lambda_v=2.0`) turned out to make
     Swiss Roll's *position* accuracy worse than not using velocity at all (`lambda_v=0`) while its
     *aggregate* score still looked fine — because the aggregate metric used identity-anchored
     velocity terms that masked the position regression.
   - Recomputed the safeguard against the report's own headline metrics (position + *location*-
     anchored velocity, not identity-anchored) instead. Under that metric `lambda_v=1.0` and `2.0`
     score within ~1% of each other, but `1.0` captures essentially all of `2.0`'s velocity-metric
     gain without the position regression — a Pareto argument, not just a safeguard technicality.
   - **Final: `lambda_v=1.0`.** Effect on the headline result: ManfitVelo beat Position-only MANFIT
     (the core M5-vs-M6 ablation, i.e. "does velocity help") on **9/9 scenarios**, up from 5/9 at the
     old `lambda_v=0.1` — a substantially stronger and more consistent finding.

7. **Δt sensitivity check.** One-time confirmatory check, not a new stress axis: does the E_flow
   method ranking hold across `τ ∈ {0.5,1,2}×τ0` on Curved Hairpin / Near Intersection? Strictly, no
   (one near-tied swap between two other methods on one scenario) — but the finding that actually
   matters, **ManfitVelo stays the best method at every tested τ on both scenarios**, does hold.
   Reported both, rather than only the strict check, which would have been technically true but
   misleading about what's robust.

8. **Consolidated report** (`results/experiment_report/index.html`) — one self-contained HTML file
   with method descriptions (including a literal step-by-step of ManfitVelo's algorithm), metric
   definitions, scenario definitions, frozen parameters, the primary comparison, the M4→M5→M6
   ablation, the `lambda_v` selection/confirmation curves, representative figures, and all three scan
   curves. Includes a short discussion of where GraphVelo/Cosine Kernel/Joint Low-Rank sit relative to
   the wider RNA-velocity literature (most published methods solve a different problem — estimating
   velocity from counts — so aren't directly comparable here without a different simulation design).

## Principles established (apply to any future work on this suite)

- **Parameter priority, strict**: official default > data-adaptive rule (may vary by scenario, since
  it's the same *rule*) > once-for-all development-seed tuning (pooled, one winner for everyone) >
  scenario-specific tuning (forbidden in anything called final).
- **Final seeds never inform a selection**, full stop — not k, not shared hyperparameters, not
  `lambda_v`, not geometry. This rule caught a real near-miss in Round 5 (see above).
- **A scan point must re-derive every data-adaptive (tier-2) parameter fresh**, never reuse the
  canonical value, even when it's provably a no-op (Scan C's `k` — computed explicitly every time
  anyway, to keep the rule exception-free and verifiable rather than assumed).
- **Safeguard any aggregate selection metric against the metric family the paper actually reports.**
  An aggregate that blends in metrics you don't headline (e.g. identity-anchored velocity when the
  report's primary velocity metric is location-anchored) can hide a regression in what you do headline.
- Prefer **archiving over deleting** anything not obviously disposable, and snapshot before overwriting
  any result that's about to change under a config update — every `archive/*_pre_*_20260811/` directory
  exists because of this.

## Bottom line, right now

- Formal comparison: 7 methods (M0 noisy / M1 GraphVelo / M2 Cosine Kernel / M3 Joint Low-Rank / M4
  Local PCA / M5 Position-only MANFIT / M6 ManfitVelo) × 9 scenarios (Circle, S-curve, Flat Rotation
  Annulus, Half-sphere-tangent, Swiss Roll, Saddle Surface, Curved Hairpin, Near Intersection,
  Y-branch) × 15 final seeds, frozen config in `methods_config.yaml`.
- Q1 (does velocity help manifold recovery): **yes, 9/9 scenarios** (M6 vs M5).
- Q2a (does manifold info help velocity at all): supported — M1/M2 underperform M6 on every 2D-curved
  scenario.
- Q4 (regimes where it helps/fails): characterized via Scan A (sample size), B (position noise, the
  most direct Q1 test), C (velocity noise) — all three in the consolidated report.
- M3 fails (worse than noisy) on every scenario with real ambient curvature, as expected for a global
  linear low-rank method — informative negative result, not a bug.

## Deferred (not done, explicitly out of scope so far)

Paired Wilcoxon significance test for the M5-vs-M6 ablation; the (σ_X, σ_V) phase-diagram experiment
the Weekly Plan calls its headline figure; real single-cell validation (everything here is synthetic);
head-to-head comparison against count-based RNA-velocity methods (scVelo, veloVI, DeepVelo, …) or
Dynamo's vector-field reconstruction; Near-Intersection's separation has no reach-audit analogous to
Curved Hairpin's; the top-level project `README.md` still references paths this work archived.

## Pointer map

| Want to know... | Look at |
|---|---|
| The full chronological story, including debugging/dead-ends | `log.md` |
| Method descriptions, metric formulas, scenario formulas, frozen parameters | `simulation_protocol.md`, `metric_definitions.md`, `scenario_config.yaml`, `methods_config.yaml`, `parameter_rules.md` |
| The finished, presentable report | `results/experiment_report/index.html` |
| Raw results | `results/manfitvelo_benchmark/`, `results/sphere_scalability/`, `results/stress_scans/`, `results/lambda_sensitivity_tuning/`, `results/lambda_sensitivity_final/`, `results/dt_sensitivity/` |
| Formal entry points (rerun everything) | `run_manfitvelo_benchmark.py`, `run_sphere_scalability.py`, `run_stress_scans.py`, `run_lambda_sensitivity.py`, `run_dt_sensitivity.py`, `build_experiment_report.py` |
| Anything archived along the way | `archive/` (scripts, legacy reports, pre-update result snapshots) |
