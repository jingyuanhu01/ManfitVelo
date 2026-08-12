# ManfitVelo Simulation — Experiment Log

Running log of work done against `ManfitVelo_Simulation_Weekly_Plan_v1.1.md`. Newest entries at the top.

---

## 2026-08-12 — Round 6: current_plan.md audit, README refresh, P0.1 (global C) selection

### Context

Asked to continue `simulation/current_plan.md` (the v2.2 forward-looking plan), but first audit it for errors
against the actual repo state. Three parallel read-only Explore agents checked P0/P1/P2/P3/P4/P5 plus
the test suite and top-level `README.md` against code and `results/*`. Findings written back into
`current_plan.md` in place (kept structure/numbering, only corrected inaccurate descriptions) rather than
only reported in chat, per the user's explicit choice. Confirmed scope for this round: (1) update
`README.md`, (2) verify the test suite still passes with the uncommitted `lambda_v` core changes, (3)
do not commit until the whole plan is frozen. P0–P5 execution itself was explicitly deferred to a
following round — which then started later the same day once the user answered the open
methodology questions the audit had surfaced.

### Audit findings (see `current_plan.md` for the in-place corrections)

Highlights: P1.2's premise was wrong — `run_sphere_scalability.py` already implements the isotropic
Gaussian ambient-D noise the plan described as new work (sphere-only; Circle/Saddle versions and a
`noise_mode` config field are the real gap). P0.2's visual claim ("ManfitVelo visibly tightens like
Local PCA" on Half-sphere) didn't match `state_half_sphere_tangent.png` on a fresh look — corrected.
P2.1's wording fix turned out to be two identical short label strings, not a rewrite. P4.0's own
requested audit (scalar-field code check) was completed: the wired-in `fit_potential_aware_neighborhoods`
fails all three of P4.0's fidelity criteria against M6; the unused `fit_self_consistent_gradient_manfit`
satisfies the first two (reuses `VelocityManifoldFitter` directly) but has zero validation history and
never passes `lambda_v`. P5's "Scan A/B final-seed?" question resolved: yes, confirmed from
`run_stress_scans.py` and `scan_seed_metrics.csv`.

### README.md

Rewrote the stale "Benchmark Pipeline" section (referenced only `archive/`d scripts/`reports/` paths
from a 2026-06-22 run) to point at the current `simulation/` suite's entry points, outputs, and
reference docs; added `lambda_v`/`velocity_covariance_mode`/`velocity_trace_normalization`/diagnostic
flags to "Parameter Priority" (previously undocumented, despite `lambda_v` being the parameter that
actually lets velocity improve manifold recovery); flagged the two "Position + Potential Experiments"
notebooks and `prepare_protein_latent_paper_data.py` as broken/archived. All referenced paths verified
to exist.

### Test verification

`/opt/anaconda3/bin/python3.13 -m pytest -q simulation` → 20/20 passing, confirming the uncommitted
`lambda_v` changes to `scripts/velocity_manifold_fitter.py`/`scripts/pca_denoisers.py` haven't broken
anything. (`scripts/.venv` is an empty 3.9.6 virtualenv with no pytest/numpy — not usable; no
`requirements.txt` exists.)

### Workflow agreed for executing the rest of the plan

Recorded in `current_plan.md`'s "执行节奏约定": (1) within P0, do P0.1 (global C) before P0.2 (Half-sphere
diagnosis), since P0.2's task 2 needs a "pooled (T,η_g)" reference point that itself depends on
whatever C is chosen; (2) pause for user confirmation before every final-seed (15-seed) full rerun,
since that overwrites the current `results/` snapshot — dev-seed checks are shown first; (3) P4.0's
`lambda_v`/`neighbor_count(n,d)` wiring into `fit_self_consistent_gradient_manfit` waits for P4's own
turn in the P0→P1→P2→P3→P4→P5 order, not done early even though the code change itself is small.

### P0.1 — global C selection

Discussed the open methodology gap (what objective, what candidate grid, dimension-(in)dependence)
with the user directly. Agreed: score by the existing `tuning_score` (same function `tune_shared_vmf`
already uses — no new scoring mechanism); candidates `C ∈ {0.30, 0.45, 0.60, 0.75, 0.90}`, five values
spanning and straddling the two old anchors `C_1≈0.361`/`C_2≈0.713`; evaluate on `TUNING_SEEDS` only,
pooled over all 9 scenarios; `C` genuinely dimension-independent (one scalar, not `C_d`); frozen across
every scenario once chosen. New standalone script `simulation/run_c_selection.py` (not part of the
frozen `main()` pipeline) applies the *same* two-stage k(n,d) procedure already in use — Stage-1
ceiling from the candidate `C`, then the unchanged Stage-2 curvature-aware refinement — while holding
every other VMF/Position-only hyperparameter at its then-current frozen value (re-tuning the T/eta_g/
theta/kappa/theta_schedule grid per candidate was explicitly out of scope; that grid gets re-run once,
after `C` is frozen, as a separate follow-up).

Ran in ~30s on 3 tuning seeds × 9 scenarios × 5 candidates (270 VMF/position-only fits, ~1900
curvature-probe `local_pca_denoise` calls). **Result: `C=0.60` wins clearly on ManfitVelo's pooled
`tuning_score`** (`-0.7529`, vs. `-0.7478`/`-0.7470` for 0.45/0.30, and `-0.598`/`-0.581` for 0.90/0.75
— a real gap opens up only above `C≈0.6`, consistent with larger `k` costing more via curvature bias
than it gains via noise-averaging). Position-only MANFIT's own pooled score (reported, not part of the
selection rule) prefers `C=0.45` by a negligible margin over `0.60` — same qualitative pattern.

**Bug found during this run, not fixed (user confirmed 2026-08-12: accept `C=0.60`, document the bug,
move on)**: Stage 2's `argmin(slope)` turn-detection implicitly assumes the log-log residual-vs-`k`
slope curve is unimodal. At the two losing higher-`C` candidates (0.75, 0.90), Curved Hairpin's
Stage-1 ceiling grows large enough (105/126) to expose a second, spurious downturn in the slope curve
late in the probed range — `argmin` grabs that instead of the true early optimum, so Stage 2 returns
`k` right at the ceiling (i.e., no shrinkage at all) instead of ~14–15 like every other candidate.
Confirmed by direct inspection of the k-grid/slope arrays (`results/c_selection/`) that this does
**not** touch the `C=0.60` winner (nor 0.30/0.45) on any of the 9 scenarios — the one `turn_index` at
the grid's last point under `C=0.60` is `flat_rotation_annulus`, which is the already-known, correct
"flat scenario, no curvature penalty" behavior from Round 2, not this bug. Net effect: the two losing
candidates look somewhat worse than they might with a fixed turn-detection rule, but the ranking and
winner are unaffected. Documented as a known Stage-2 limitation in `parameter_rules.md` §2 rather than
fixed, since a fix would change turn detection for every scenario's `k`, not just this comparison, and
was out of scope for P0.1.

**Applied**: `simulation/benchmark_core.py`'s `NEIGHBOR_SCALING_CONSTANT` changed from the
`{1: C_1, 2: C_2}` dict to the scalar `0.60`; `neighbor_count(n, d)` updated to match;
`run_manfitvelo_benchmark.py`'s `selected_hyperparameters.json` builder updated to emit a single `C`
instead of a per-dimension `C_d` map. `parameter_rules.md` §2 and `current_plan.md` P0.1 updated with the
full selection record. Test suite re-verified: 20/20 still passing after the constant change (nothing
hardcodes the old per-dimension values).

### Dev-seed tier-3 re-selection under the new k(n,d) (same day, continued)

Re-ran `curvature_aware_scenario_k()` (now C=0.60) followed by `tune_shared_vmf`/
`tune_shared_position_only` on dev seeds (results:
`results/c_selection/tier3_reselection_vmf_dev_seeds.csv` /
`tier3_reselection_position_only_dev_seeds.csv`). New per-scenario `k` shifts are all small — Stage 2
absorbed most of the Stage-1 ceiling change (e.g. `flat_rotation_annulus` 40→34, `y_branch` 33→28,
most others ±0-2). **The 162-candidate grid's winner is unchanged**: `T=3, eta_g=0.7, theta=0.02,
kappa=0.0, theta_schedule=flat, lambda_v=1.0, velocity_covariance_mode=uncentered,
velocity_trace_normalization=match_position_trace` — identical to the value already frozen before
this round's `C` change. Position-only MANFIT's winner is also unchanged (`T=3, eta_g=0.7`). Lower
risk than expected for the upcoming final-seed rerun, since it isn't gambling on a genuinely new
shared-hyperparameter combination, only the (small) per-scenario `k` shifts above.

`methods_config.yaml`'s `shared_graph_k` snapshot intentionally left stale for now — it mirrors
`results/manfitvelo_benchmark/selected_hyperparameters.json`, which only regenerates on a real
`main()` run; both stay in this known transitional state until that rerun happens.

### Final-seed canonical benchmark rerun (same day, user confirmed)

User confirmed the final-seed rerun. Archived the pre-change snapshot to
`archive/manfitvelo_benchmark_pre_globalC0.60_20260812/`, then ran
`python simulation/run_manfitvelo_benchmark.py` (new `k(n,d)` under `C=0.60`; `T=3, eta_g=0.7,
theta=0.02, kappa=0.0, theta_schedule=flat, lambda_v=1.0` unchanged, matching the dev-seed
re-selection above). `sanity_checks.json`: `all_checks_pass: true`,
`final_seeds_used_for_selection: false`.

**Headline "9/9" claim (M6 beats M5 on `clean_point_rmse_rel`, every scenario) no longer holds at the
median level.** 8/9 scenarios: identical direction as before (`curved_hairpin`/`saddle_surface` numbers
are bit-for-bit unchanged, since their `k` didn't move; the other 6 changed a little but M6 still wins).
**`swiss_roll` flips**: M5=0.7152 vs M6=0.7269 by median-of-ratios (M6 ~1.6% worse) — but checked the
per-seed pairs directly (`final_seed_metrics.csv`) and **M6 still wins 11/15 individual final seeds**;
the median flip comes from M6's few losing seeds losing by more than its many winning seeds win by, not
from a systematic regression. This is exactly the kind of case `current_plan.md`'s own pre-freeze claim-review
checklist (item 2) already flagged `swiss_roll` (G1) as needing a paired Wilcoxon test for before making
strong claims — it just went from "flagged as thin-margin" to "flagged as thin-margin *and now flipped
under the median statistic*." Updated `current_plan.md`'s P0.1 section and claim-review item 2 with the full
before/after table and this caveat; the "9/9" number cannot be used again until the Wilcoxon test (still
on the P5 backlog) is run.

### Cascading reruns: sphere scalability, stress scans, lambda_v confirmation (same day, user confirmed)

User confirmed. Archived `results/sphere_scalability/`, `results/stress_scans/`,
`results/lambda_sensitivity_final/` to their `archive/*_pre_globalC0.60_20260812/` counterparts, then
ran the three in sequence (each pulls the just-refrozen `k`/hyperparameters via `load_frozen_config()`
or `neighbor_count()`, so no code changes were needed beyond the `C` update already applied):

- `run_sphere_scalability.py`: `all_checks_pass: true`. M6 still beats M5 at every ambient dimension
  D∈{3,5,10,20,50} — no flips, direction fully stable.
- `run_stress_scans.py`: HTML/figure-count checks pass (this script's `sanity_checks.json` has no
  single `all_checks_pass` field, unlike the other two). Compared against the archived pre-`C`-change
  scan summaries: a handful of individual scan points flip which method wins on
  `clean_point_rmse_rel_median`, but they're isolated single points at edge conditions (very small n,
  very low/high noise multiplier), not a systematic pattern — except `swiss_roll`, which flips at 4
  separate points across Scan A/B/C (including its own canonical-noise point in Scan B/C). This is the
  *same* thin-margin `swiss_roll` behavior already found in the canonical benchmark, just visible across
  a noise/sample-size range rather than only at the one canonical point — reinforces, rather than adds
  to, that finding.
- `run_lambda_sensitivity.py --seeds final`: confirmatory sweep, `marker_lambda: 1.0`. Recomputed the
  headline-metric safeguard (mean log of G1/G2/location-anchored-V3/joint-Euler) across all 9 scenarios
  under the new `C`: **`lambda_v=1.0` still never regresses below its own `lambda_v=0` baseline on any
  scenario** — the safeguard that originally justified freezing `lambda_v=1.0` (see Round 5) still
  holds after the `C` change. `swiss_roll` shows the same non-monotonic (U-shaped, best near `lambda_v
  ~=0.5`, dips slightly at 1.0, recovers toward 2.0) pattern documented in the original archived study,
  not a new artifact. **No re-selection of `lambda_v` needed.**

Per the user's explicit instruction (2026-08-12, "还是按原来的顺序" — keep the original order), the
paired Wilcoxon test for `swiss_roll`/`circle`/`flat_rotation_annulus` stays on the P5 backlog rather
than being pulled forward, even though `swiss_roll`'s flip makes it materially more useful sooner.
Next in the agreed P0 order: P0.2 (Half-sphere diagnosis), using the now-confirmed final-seed pooled
`(T,η_g)=(3, 0.7)` as its reference point.

### P0.2 — Half-sphere anomaly diagnosis (same day, user confirmed "ok")

New standalone diagnostic script `simulation/run_half_sphere_diagnosis.py` (dev seeds only, holds
everything except the thing each task is probing at the current frozen values), covering the three
tasks the plan specified. Sanity-checked the manual internal-method-driven loop (needed for task 1's
per-iteration position snapshots, since `VelocityManifoldFitter.fit()` doesn't expose them) against
the library's own `fit()` output before trusting anything from it — exact bit-for-bit match.

**Task 2 (half-sphere-specific `(T,η_g)` grid vs. pooled) is the decisive result.** On half-sphere's
own 3 tuning seeds, holding `k=21` (frozen under the new `C=0.60`) and every other hyperparameter
fixed, swept `T∈{3,5,8}×η_g∈{0.35,0.5,0.7}`. Pooled `(T=3,η_g=0.7)` scores `clean_point_rmse_rel`
mean `0.7906`; half-sphere's own best, `(T=3,η_g=0.35)`, scores `0.5469` — a **44.6% relative gap**,
and monotonic in the expected direction (bigger `T`/`η_g` always worse on this scenario, worst
candidate `(T=8,η_g=0.7)` at `1.87`, worse than noisy input). Matches the same big-step/high-curvature
overshoot mechanism already root-caused for `k` in Round 1/2, now confirmed to also apply to
`(T,η_g)`. **Bonus finding**: half-sphere's own best `(T,η_g)` (`0.5469`) actually beats Local PCA
(M4)'s final-seed canonical number (`0.661`) — so the "M4 beats M5/M6 on half-sphere" pattern that
motivated this whole diagnosis (found via Scan A/B) is plausibly *entirely* a pooled-hyperparameter
artifact, not evidence that velocity information is unhelpful on this geometry. (Caveat: dev vs. final
seed sets aren't identical, so not a strictly apples-to-apples comparison, but the gap is large enough
that the direction should be robust.)

**Task 1 (per-iteration trajectory)**: traced both the frozen `T=3` and an extended diagnostic `T=15`
run; no anomalous growth or oscillation in step size near the `|z|<0.1` boundary vs. away from it.

**Task 3 (boundary z≈0 check) — no implementation bug found.** Noisy *input* never has `z<0` (position
noise is a radial scaling of each point's own position vector, which can't flip its sign) — confirmed,
0 occurrences. The fitted *output* does have a handful of points cross `z=0` during iteration (3 events
across 3 dev seeds at the frozen `T=3`, growing to 17 by `T=15`), but every single one starts and ends
within `|z|<0.01` — an order of magnitude below the position-noise scale — i.e. points that are
genuinely sitting on the equator wobble by a hair to the other side, not a divergence or runaway
correction. `distance_to_manifold`'s `X[:,2]<0` branch is confirmed to actually get exercised (not
dead code), and handles it correctly.

**Bug caught and fixed in this round's own diagnostic script, not in the algorithm**: the first run of
task 2 reported the pooled comparison as `NaN`. Root cause: `summary.T` in pandas is the DataFrame's
transpose attribute, which silently shadows a column literally named `"T"` — comparing `summary.T ==
cfg["T"]` produces garbage, not the intended column filter. Fixed to `summary["T"]`; the underlying
per-candidate fit data in `task2_grid_long.csv`/`task2_grid_summary.csv` was correct from the first run
(the bug was only in the pooled-vs-best comparison step), so only that comparison needed recomputing,
not the fits themselves.

**Decision (per `current_plan.md`'s pre-specified branch): confirmed pooled-hyperparameter trade-off, not
an implementation bug.** No change to the frozen `T/eta_g/theta/kappa/theta_schedule`. Updated the
Claim 语言复核清单 item 1 (§5.7 "successive improvement" wording) with the resolved conclusion and the
44.6% figure, per the plan's own pre-specified fix.

P0 is now essentially complete (P0.1 executed and cascaded through the canonical benchmark + sphere
scalability + stress scans + lambda_v confirmation; P0.2 diagnosed and resolved). Moved on to P1 the
same day, user confirmed "OK，现在开始".

### P1.1 — Scan C redesign (same day)

Rewrote `simulation/run_stress_scans.py`'s Scan C section: `scenario_velocity_scale()` (median
||V_true|| pooled over TUNING_SEEDS, a fixed dev-seed-only reference constant, not a selection),
`shuffle_velocity_field()` (row-permutes noisy velocity with an RNG stream independent of the data
generation seed), a new relative grid `r_V = sigma_V / scenario_velocity_scale in
{0.05,0.1,0.2,0.4,0.8,1.6}` replacing the old absolute `{0.05,...,0.30}`, one additional
shuffled-velocity condition per scenario, and tangent/normal-component decomposition
(`mechanism_tangential_component_rmse`/`mechanism_normal_component_rmse`, reusing
`run_manfitvelo_benchmark.mechanism_metrics`/`true_projector`, already scenario-general so no new
per-scenario ground truth was needed) for M5/M6/M4. Also added `--scans A,B,C` so a redesign of one
scan doesn't require re-running the other two -- used `--scans C` here to reuse the just-refreshed
A/B rows from the earlier C=0.60 cascade rather than repeat ~10 minutes of unrelated work.

Smoke-tested `evaluate_condition`/`summarize`/`plot_scan`/`shuffle_control_table` on a single
scenario/point before the full run, and confirmed a known correctness invariant holds exactly:
Position-only MANFIT's `clean_point_rmse_rel` is bit-for-bit identical across every `r_V` and the
shuffle condition for a given scenario (it never uses velocity for position updates) -- this held in
both the smoke test and the full run. Archived the pre-redesign Scan C to
`archive/stress_scans_pre_scanC_redesign_20260812/` before overwriting. Full run: ~6.6 minutes,
`sanity_checks.json`: `self_contained_html: true`, `14/14` figures.

**Result: a clean, monotonic M5-vs-M6 crossover on `clean_point_rmse_rel` in 8/9 scenarios.**
`flat_rotation_annulus`/`saddle_surface` flip earliest (between `r_V` 0.2 and 0.4); `half_sphere_
tangent`/`near_intersection`/`y_branch` flip mid-grid (0.4→0.8); `circle`/`curved_hairpin`/`s_curve`
hold out longest (0.8→1.6). `swiss_roll` never favors M6 anywhere on this grid, consistent with the
thin-margin finding already documented elsewhere (not new). Noted a clean consistency check: the
scenarios where `lambda_v` gave the biggest Round-5 gains (circle, s_curve, curved_hairpin) are also
the ones most robust to velocity noise here -- the scenarios that benefit most from velocity
information also tolerate the most noise in it before that information turns harmful, which is the
qualitatively expected relationship.

**Shuffled-velocity negative control** (row-permuted at each scenario's own canonical sigma_V, which
maps to `r_V` ~=0.09-0.13, just above the grid's smallest point): M6 loses to M5 in 7/9 scenarios once
velocity carries zero position-specific information -- direct evidence M6 is genuinely using velocity
information rather than being insensitive to it. Two exceptions (`curved_hairpin`,
`flat_rotation_annulus`) still favor M6 even under full randomization.
`flat_rotation_annulus`'s shuffled score (0.1754) is oddly *better* than its own real-noise score at
similarly-sized `r_V` (0.2097 at 0.05, 0.2124 at 0.10) -- flagged as an observed, unexplained anomaly
(plausibly related to the annulus's rotational symmetry making a randomly-swapped velocity from a
similar-radius point less wrong than it would be on a less symmetric scenario) rather than chased
further this round.

Full write-up with the per-scenario crossover table: `current_plan.md` P1.1. Raw data:
`results/stress_scans/scan_seed_metrics.csv`/`summary_metrics.csv`/`scan_report.html`.

### P1.2 — Circle/Saddle ambient-D scalability, scope split (same day)

Before implementing, the user re-scoped P1.2: it was implicitly bundling two separate questions
(does the isotropic-Gaussian-noise mechanism work at all; does ambient dimension itself hurt
recovery). The first is already answered by `run_sphere_scalability.py`. Agreed: keep P1.2's Circle +
Saddle Surface scenario choice, but drive them with `normal_only` position noise (the same convention
the 9 canonical scenarios already use) instead of isotropic Gaussian — decoupling "does D matter" from
"does noise mode matter," and reusing the cheaper, already-established noise-generation machinery
instead of re-implementing isotropic-Gaussian scale-matching for two more manifolds.

New script `simulation/run_manifold_dimension_scalability.py`, structured like
`run_sphere_scalability.py` (same deterministic orthonormal D x 3 embedding, same `FINAL_SEEDS`/
`DIMENSIONS`/`METHODS`). Circle/Saddle Surface's analytic geometry (position, tangent, normal) reused
directly from `vector_data()`'s existing formulas. Position noise: `normal_only`, D-independent
magnitude (a scalar along the embedded analytic normal). Velocity noise: same ambient-isotropic
"fixed_coordinate" convention as sphere (total magnitude grows as sqrt(D)) -- `noise_mode` only ever
described position noise, velocity noise has always been ambient-isotropic in this codebase. Circle
and Saddle Surface are literally 2 of the 9 canonical scenario names, so their frozen hyperparameters
were reused directly from `selected_hyperparameters.json`, no separate tuning stage.

**Two bugs caught before/right after the full run, neither in the algorithm itself:**
1. (Caught in smoke test, before the full run) The true tangent-projector formula initially used
   `I - normal⊗normal` uniformly for both manifolds. That's correct for Saddle Surface (codimension 1
   in R^3: one normal spans the full orthogonal complement) but wrong for Circle (codimension 2: one
   normal is not enough), which needs the direct `tangent⊗tangent` rank-1 formula instead — matching
   the existing generic d=1 fallback in `run_manfitvelo_benchmark.py::true_projector`. First smoke-test
   run showed Circle's `tangent_projector_error` pinned near 1.0 for every method (estimated and "true"
   projectors nearly orthogonal) — the tell that something was wrong with the reference, not the fits.
   Fixed by giving each manifold its own `tangent_projector_3d` function.
2. (Caught right after the full ~12-minute run finished) `config.json`'s provenance dump tried to
   JSON-serialize the `MANIFOLDS` config dict, which by then also held the new `tangent_projector_3d`
   function reference (forgot to add it to the same exclude-list as the other two function-valued
   keys) -- crashed with exit code 1 after `seed_metrics.csv`/`summary_metrics.csv` were already
   written. No data was lost; fixed the exclude list and re-ran only the report/validation step
   (`build_report`/`validate`/provenance dump) from the already-computed CSVs rather than repeating
   the 12-minute computation.

**Result: M6 beats M5 on `clean_point_rmse_rel` at every one of the 5 ambient dimensions on both
manifolds -- no flips.** Advantage size: Circle stays flat at 4-6% across D=3..50, no visible trend.
Saddle Surface holds 12-17% through D=20 but narrows to 3.5% at D=50 -- reported honestly rather than
smoothed over. Root cause visible in `velocity_rmse_id`: the raw noisy input's velocity error grows
~sqrt(D/3) as designed (0.17 -> 0.71, matching the fixed_coordinate regime), while M5/M6's *fitted*
velocity error grows far more slowly (M6 on saddle: 0.144 -> 0.179, +24%; M5: 0.156 -> 0.160, +3%) --
both methods denoise effectively, but at D=50 the velocity signal M6 can still usefully extract on top
of M5's baseline shrinks, without ever going negative. Position-only metrics (`clean_point_rmse`,
`distance_to_manifold`) stay nearly flat across D for every method on both manifolds -- the expected
signature of `normal_only` noise (D-independent magnitude by construction), a clean contrast against
sphere's isotropic-Gaussian experiment where position error visibly grows with D. That contrast is
itself direct evidence that "does ambient D matter" depends on noise mode, which was the whole point
of separating this from the sphere experiment.

`noise_mode` added to `simulation/scenario_config.yaml` (new top-level `noise_modes` schema with both
values documented + `used_by`/`used_by_supplement_experiments`; all 9 canonical scenarios explicitly
tagged `normal_only`) and `parameter_rules.md` (new §8). Full data: `results/manifold_dimension_
scalability/` (`seed_metrics.csv`/`summary_metrics.csv`/`scalability_report.html`/`config.json`).

P1 is now complete (P1.1 Scan C redesign, P1.2 Circle/Saddle ambient-D scalability). Moved on to P2
the same day, user confirmed "继续".

### P2.1 — GraphVelo wording fix (same day)

Exactly the two-file, one-string fix already scoped in an earlier round: `"official defaults,
untouched"` -> `"official algorithm untouched; input rescaled by a fixed truth-free rule"` in
`simulation/build_experiment_report.py:85` and `simulation/simulation_protocol.md:30`.

**Deliberately did not regenerate `results/experiment_report/index.html`.** Checked
`build_experiment_report.py::section_results()` first and found it hardcodes several claims this
session has already made stale: §5.7 Q1 still says "beats Position-only MANFIT (M5) on all 9/9
scenarios" (now 8/9 -- swiss_roll flipped, see the P0.1 entry above), §5.6 still describes the old
absolute Scan C grid `sigma_V in {0.05,...,0.30}` (redesigned in P1.1), and §5.7 Q2b still has the
exact "successive design choice ... incremental improvement" sentence the Claim 语言复核清单 already
has a resolved replacement for (P0.2 entry above) that was never applied to the actual report file.
Regenerating now would only fix the M1 label while leaving these other, larger staleness issues in
place -- arguably worse than leaving the whole report visibly last-generated pre-P0, since a partial
refresh could read as "this is now correct." Left for P5 (final freeze), which is supposed to
regenerate this report once, after the Wilcoxon test and all final numbers are in.

### P2.2 — Joint Low-Rank (M3) threshold sensitivity (same day)

New script `simulation/run_joint_low_rank_threshold_sensitivity.py`, `q in {0.80, 0.90, 0.95, 0.99}`
(frozen protocol uses 0.90) x 9 scenarios, on **final seeds** -- this is a pre-committed-decision-rule
robustness/reporting check (q stays 0.90 regardless of outcome, unless a genuine implementation
problem turned up), not a performance-based selection of q, so it doesn't carry the usual
final-seed-leakage risk that gates tuning-stage choices. Pure SVD, no iterative fitting: full run in
2.3 seconds.

**Rank pattern matches the plan's predicted "smoking gun" reasonably cleanly.** `half_sphere_tangent`
(the most curved scenario) needs the highest joint-SVD rank at every threshold (3/6 at q=0.80, climbing
to a full 6/6 by q=0.99) and has by far the worst `clean_point_rmse_rel` (8.2x noisy input at q=0.80).
`flat_rotation_annulus` (the only exactly-flat scenario) stays pinned at the lowest rank (2/6) through
q=0.80/0.90/0.95, only rising to 4/6 at q=0.99, and its `clean_point_rmse_rel` (1.42-1.47), while still
worse than noisy, is nowhere near as bad. V-only SVD rank is consistently <= the joint [X,V] rank
(sometimes by a lot, e.g. half_sphere_tangent v_only=3 vs joint=4-6 as q rises) -- velocity is
intrinsically more compressible than position, so M3's fixed rank budget spends disproportionately on
velocity at position's expense, which is the mechanism behind the already-documented "M3's velocity
reconstruction is oddly strong" observation from Round 3.

**The one nuance worth recording explicitly**: the script's own blunt "worse-than-noisy at every
threshold?" boolean flag is inconsistent for 3/9 scenarios (`flat_rotation_annulus`,
`half_sphere_tangent`, `y_branch`), all flipping at q=0.99 specifically. Traced this to a degenerate
edge, not a real reversal: at q=0.99, `half_sphere_tangent`/`y_branch`'s selected rank hits the full
joint rank (6/6) -- i.e. M3 stops truncating at all, `Xhat` becomes bit-identical to the noisy input,
so `clean_point_rmse_rel` is trivially exactly 1.0 by construction, not evidence M3 "works" there.
`flat_rotation_annulus` at q=0.99 is a genuine near-tie (0.9993) rather than a full degenerate case,
but still not a real reversal. Excluding this "threshold so high it barely truncates anything" regime,
every scenario's qualitative story is identical across q=0.80/0.90/0.95, the range where M3 is actually
doing something. **Decision: freeze M3 as-is (q=0.90)** -- the curved-geometry failure is not an
artifact of q=0.90, it holds at every threshold that meaningfully compresses the data.

Full data: `results/joint_low_rank_threshold_sensitivity/` (`threshold_sensitivity_long.csv`/
`threshold_sensitivity_summary.csv`/`p2_2_summary.json`).

P2 is now complete (P2.1 wording fix, P2.2 threshold sensitivity, M3 frozen as-is). Moved on to P3
the same day; user asked what "pushforward across four embeddings" (V2's core requirement) actually
meant, got the explanation, then said proceed and to ask now if anything else was unclear -- took that
as a green light rather than manufacturing more questions, and scoped this round to V1 only (more
self-contained, reuses more existing field-generator code), leaving V2 (needs genuinely new
pushforward infrastructure) for a following round.

### P3 Experiment V1 — same manifold, different vector fields (same day)

New script `simulation/run_v1_field_family.py`. Scope decisions made directly rather than asked
(stated plainly, not blocking): unit disk (not square), embedded at z=0 with the same normal
convention as `flat_rotation_annulus`; five fields (source, sink, saddle, rotation, nonlinear) --
double_well skipped, per the plan's own "optional" framing; n=480/sigma_X=0.05/sigma_V=0.10 matching
the other d=2 canonical scenarios; each field individually rescaled to median speed 1 before noise
(preserves each field's own speed-variation structure, e.g. rotation/source/saddle genuinely grow with
radius while nonlinear doesn't, while keeping noise-to-signal comparable across field types); shared
hyperparameters reused verbatim from the frozen protocol (identical across all 9 canonical scenarios,
borrowed from "circle"'s entry), only k recomputed fresh via `neighbor_count(n,d)`; 15 final seeds,
reporting only.

Smoke-tested on one field/one seed before the full run -- results already visibly confirmed the plan's
own prediction (Joint Low-Rank near-oracle on the four linear fields, breaking on nonlinear) even at
that small scale. Full run (5 fields x 15 seeds x 7 methods): ~23 seconds.

**Two bugs, both in the reporting script, not the underlying computation:**
1. `relative_state_metrics()` already suffixes its own keys with `_rel`; the row-building code in
   `run()` wrapped the result in another `_rel`-suffixing dict comprehension, producing columns named
   `..._rel_rel` and crashing `summarize()` with a `KeyError` looking for the un-double-suffixed name.
   Fixed by using the dict directly.
2. `final_seeds_used_for_selection: False` is correct *by design* here (nothing in this run selects
   anything), but got included in the generic "all boolean values must be True to pass" aggregation,
   making `all_checks_pass` report `false` even though every real check passed. Pulled it out into a
   separate informational field, not part of the pass/fail set.

**Result: M6 beats M5 on `clean_point_rmse_rel` in all 5 fields, no flips** (0.176/0.169/0.180/0.209/
0.185 vs. a flat 0.227 for M5 across source/sink/saddle/rotation/nonlinear respectively). **Joint
Low-Rank (M3) confirms the plan's own prediction cleanly**: on the four linear fields
(source/sink/saddle/rotation, all exactly `v=Ax`), M3's `velocity_rmse_loc_rel` is 0.09-0.11 -- an
order of magnitude better than every other method, essentially oracle-like, since a global linear SVD
is a near-perfect fit for a genuinely linear relationship. On `nonlinear`, M3 collapses:
`velocity_rmse_loc_rel`=1.005 (no better than noisy input), `clean_point_rmse_rel`=3.85 (3.85x *worse*
than noisy input) -- exactly the "the first four fields are too easy for M3, only the nonlinear one is
a real stress test" point the plan made in justifying why that field was necessary.

**An unplanned but clean internal-consistency check turned up in the data**: Local PCA's and
Position-only MANFIT's `clean_point_rmse_rel` are bit-for-bit identical across all 5 fields (Local PCA
pinned at 0.248675, M5 at 0.226992, to 6 decimal places) -- expected, since neither uses velocity for
position updates, and position sampling happens before the field-specific branch in `disk_data()`, so
a given seed's positions (and position noise) are literally identical draws regardless of which field
is active. This "should be exactly equal" prediction held exactly in the data, a strong (free)
correctness signal. M6 (which does use velocity for position updates) correctly shows real
field-to-field variation (0.169-0.209) instead.

Full data: `results/v1_field_family/` (`seed_metrics.csv`/`summary_metrics.csv`/`v1_report.html`/
`provenance.json`).

User asked what "pushforward across four embeddings" meant before greenlighting V2; explained it (a
shared latent-space dynamics rule pushed through each embedding's own Jacobian, rather than four
independently hand-written velocity fields, so any observed cross-manifold difference is attributable
to geometry alone) and proceeded once they said "great! proceed now" and to ask if anything else was
unclear.

### P3 Experiment V2 — same intrinsic dynamics, different manifolds (same day)

New script `simulation/run_v2_manifold_family.py`. Shared latent dynamics u_dot=1, v_dot=0 (the
optional rotational variant skipped, per the plan's own compute-budget language); pushforward
V = d(phi)/du hand-derived analytically per embedding (no autodiff needed): flat_plane and
sphere_patch newly written (sphere_patch's Jacobian reduces to the ambient-native (-y,x,0), same form
as flat_rotation_annulus's own rotation field, restricted to the sphere); swiss_roll and
saddle_surface reuse the canonical scenarios' exact phi(u,v) formulas and domains verbatim -- their
existing velocity fields already are pushforwards by construction (derivatives along one parameter
direction), a fact noticed while scoping this round.

**Deliberate departure from canonical convention, flagged explicitly**: canonical swiss_roll/
saddle_surface normalize velocity to unit speed *per point*, which would erase exactly the
embedding-curvature-changes-speed signal this experiment needs. V2 instead keeps the raw Jacobian
magnitude, rescaled by one *global* per-manifold constant (median speed over a fixed reference sample,
seed 90210, independent of FINAL_SEEDS) -- same convention as V1's per-field rescaling.

**Caught a real bug via smoke test + full run, not just an ugly result.** First version used only
`neighbor_count(n,d)`'s raw Stage-1 ceiling (k=37) for every manifold. Both the smoke test and the
first full 15-seed run showed catastrophic failure specifically on `sphere_patch`/`swiss_roll`: every
geometry-fitting method (including Local PCA) scored *worse than noisy input* -- swiss_roll's M6 hit a
relative 1.97, Local PCA 1.69. This is the exact Euclidean-kNN-bridges-across-curvature failure mode
already documented in Round 2/3, and the tell was concrete: canonical `swiss_roll`'s frozen k is 16,
`half_sphere_tangent`'s is 21, both far below 37 -- Stage 2 (curvature-aware refinement) had simply
been forgotten, only Stage 1's raw ceiling was applied. Added `curvature_aware_k_for_manifold()`
(identical two-stage procedure on TUNING_SEEDS, reusing `curvature_probe_k_grid`/
`local_pca_normal_residual`/`curvature_aware_neighbor_count` from `benchmark_core` unchanged). Result
after the fix is a strong correctness signal, not just a plausible-looking number: **swiss_roll's
computed k came out to exactly 16, saddle_surface exactly 26, sphere_patch 21 -- matching the
canonical scenario/half_sphere_tangent frozen values exactly**, despite being derived independently
through a differently-parametrized generator. Re-ran; every manifold's numbers landed in a sane range.

**Result (15 final seeds, post-fix)**: `flat_plane`/`saddle_surface` -- M6 beats M5 cleanly on every
metric (clean_point_rmse_rel 0.204 vs 0.221, and 0.264 vs 0.321). `sphere_patch`/`swiss_roll` -- M6
trails M5 on *position* metrics (clean_point_rmse/distance_to_manifold/joint_euler) but **still beats
M5 on `velocity_rmse_loc_rel` on all four manifolds**, including these two (sphere_patch 0.831 vs
0.846; swiss_roll 0.898 vs 1.079) -- not a blanket "M6 worse here," a genuine metric-dependent
thin-margin split. Both trailing cases connect cleanly to mechanisms already established this
session, not new problems: swiss_roll's V2 numbers (M5=0.7152, M6=0.7253) land almost exactly on the
canonical swiss_roll scenario's own final-seed numbers from the P0.1 rerun (0.7152/0.7269) despite
using a completely independent velocity-generation convention (raw pushforward+global-rescale vs.
per-point unit-normalization) -- strong cross-validation that this is the same known thin-margin
scenario, not an artifact of the new implementation. sphere_patch's shortfall matches the mechanism
P0.2 already quantified for half-sphere-like positive-curvature geometry: the pooled (T, eta_g) has a
real, measured cost there (44.6% in P0.2's own diagnosis) -- sphere_patch is another instance of the
same effect on a new geometry, not evidence of a new problem.

Full data: `results/v2_manifold_family/` (`seed_metrics.csv`/`summary_metrics.csv`/`v2_report.html`/
`provenance.json`).

P3 is now complete (V1 + V2). Before starting P4, user asked to rename
`fit_self_consistent_gradient_manfit` (drop the "self_consistent" implementation-detail wording) and
delete the other implementation, and to ask now if anything was unclear.

### Pre-P4 cleanup: rename + delete the unused scalar implementation (same day)

Flagged a real risk before touching anything: `fit_potential_aware_neighborhoods` (the implementation
being deleted) is actually called by `scripts/run_field_informed_manfit_benchmark.py`, and *that* file
is git-untracked (unrecoverable if broken) and is the load-bearing generator for the entire active
9-scenario pipeline (`vector_data`/`SETS`/`fit_vmf_variant`/`position_only_trajectory`/`hairpin` all
live there; `simulation/benchmark_core.py` imports directly from it). Checked with the user which file
they meant before deleting anything, given the ambiguity and the stakes; confirmed: delete
`fit_potential_aware_neighborhoods`-related code from `scalar_potential_manfit.py`, not the file that
imports it.

**Rename**: `fit_self_consistent_gradient_manfit` -> `fit_scalar_gradient_manfit` in
`scripts/scalar_potential_manfit.py`.

**Delete**: `fit_potential_aware_neighborhoods` and its exclusive helpers
(`_weighted_local_pca_basis`, `_normal_candidate_grid`, `_plane_basis_from_normal`,
`_tangent_constrained_basis`, `_local_geometry_fit`), plus the never-called
`fit_tangent_constrained_scalar`. `scalar_potential_manfit.py` itself is committed to git (recoverable
if this needs reverting), unlike the file that calls it.

**The cleanup turned out to be much more entangled than "delete an import."** Read the entirety of
`run_field_informed_manfit_benchmark.py` (previously only skimmed) to find every reference before
touching anything. `"scalar_potential_manfit"` (the method name, not just the function) is hardcoded
throughout that file's own independent legacy report generator: `SMETHODS`, `LABEL`, `geom_fit`,
`candidates`, `tune_scenario` (multiple branches, including one at the old line 253 that unconditionally
tries this method name whenever `kind=="scalar"`, not gated by `SMETHODS` membership -- would have
crashed with `KeyError: 'config_json'` on an empty dataframe if left as-is once the method stopped
being evaluated), `representative`, `sample_study`, `build_report`'s "Part II" section, and two
assertions inside `checks()`. Verified via grep that none of this machinery (`main`, `tune_scenario`,
`checks`, `geom_fit`, `candidates`, `representative`, `sample_study`, `build_report`) is imported by
anything under `simulation/` -- only `vector_data`/`SETS`/`add_noise`/`fit_vmf_variant`/
`position_only_trajectory`/`hairpin`/`HAIRPIN_DEFAULT_SEPARATION`/`HAIRPIN_LEGACY_SEPARATION` are, so
none of this dead code was actively reachable, but leaving it half-broken (referencing a now-undefined
function/method name inside function bodies that Python won't complain about until actually called)
would be a landmine for whoever next runs this file directly.

Removed: the import; `SMETHODS`/`LABEL` entries; `geom_fit`'s and `candidates`'s
`scalar_potential_manfit` branches; the two `checks()` assertions that exercised the removed function/
method (kept the other two scalar-related checks in the same function -- `common_gradient` invariance
and `scalar_saddle`'s analytic gradient tangency -- since those test things that still exist); and
`main()`/`build_report()`'s three hardcoded `("scalar", SCALAR, SMETHODS)`-style loop entries and the
"Part II. Manifold fitting with scalar functions" report section, which sidesteps the deeper
`tune_scenario`/`representative`/`sample_study` scalar branches entirely (never reached now, rather
than patched one by one).

**Verification, layered**: `ast.parse` on both edited files; fresh import of both modules plus every
active `simulation/*.py` entry point; called `checks()` directly (not part of the active pipeline, but
a real self-test) and confirmed it still returns cleanly with the two removed-function assertions gone
and everything else intact; full `pytest -q simulation` (20/20).

**`lambda_v` wiring for `fit_scalar_gradient_manfit`**: added `lambda_v=0.0` (matches
`VelocityManifoldFitter`'s own class default -- not silently enabled), `velocity_covariance_mode=
"centered"`, `velocity_trace_normalization="match_position_trace"`, threaded through both internal
`VelocityManifoldFitter` calls. **Verification caught an initially-alarming-looking non-bug**: first
tested `lambda_v in {0, 1, 5}` on `scalar_s_curve` and got bit-for-bit identical output regardless --
looked like the wiring silently wasn't working. Root cause, confirmed by testing directly against
`VelocityManifoldFitter` itself (bypassing the wrapper entirely) rather than assuming the wrapper was
at fault: `scalar_s_curve` is embedded with z=0 identically for every point, so the z-direction
contributes exactly zero variance to *both* the position and velocity covariance terms regardless of
`lambda_v` -- the top-2 eigenvectors are unambiguously the XY-plane either way, so there's no room for
the covariance blend to change anything. Re-tested on `scalar_saddle` (genuinely varies in all 3
ambient coordinates) and `lambda_v` immediately showed real, distinct effects on the fitted positions.
Kept `k` un-wired to `neighbor_count` internally (matching `fit_vmf_variant`'s own design -- the
orchestration layer supplies `k`, not the fitting routine), documented explicitly in the docstring
instead.

User said "开始吧" to start P4.1.

### P4.1 — oracle-gradient ablation (same day)

Added an `oracle_gradient` parameter to `fit_scalar_gradient_manfit`: when given, every outer
iteration uses the exact true gradient (confidence fixed at 1.0) instead of calling
`estimate_gradient_confidence_from_neighbors`, with every other mechanic (k, T, lambda_v, ...)
unchanged -- isolates joint geometric-fitting error from local-regression (gradient estimation) error.
New script `simulation/run_p4_1_scalar_oracle_ablation.py`: for the two existing scalar scenarios
(`scalar_s_curve`, `scalar_saddle` -- S1/S2 controlled scalar experiments are separate, later work),
15 final seeds, 5 pipeline variants per scenario/seed (raw local regression alone; {estimated, oracle}
gradient source crossed with lambda_v in {0.0, 1.0}, the latter matching the frozen vector-field M6
value).

Smoke-tested on one seed first and immediately saw a pattern worth double-checking before trusting:
`oracle_gradient`'s reported gradient error came out as *exactly* zero (~1e-16) on `scalar_s_curve`.
Verified this is real, not a bug -- `scalar_s_curve` is embedded with z=0 identically for every point
(same degeneracy already found during the lambda_v wiring verification), so the true gradient already
lies entirely within whatever 2D tangent subspace gets estimated (unambiguously the XY-plane), making
the projection step a no-op by construction. `scalar_saddle` (genuinely 3D-varying) showed a real,
nonzero oracle gradient error as expected. Full 15-seed run: ~35 seconds.

**Result 1**: raw local-regression gradient error is large on both scenarios (0.56-0.58, comparable in
magnitude to the gradient's own typical scale) -- expected, differentiating noisy data is inherently
noise-amplifying.

**Result 2, the important one**: `lambda_v` behaves in *opposite* directions on the estimated vs.
oracle pipeline for `scalar_saddle`. At `lambda_v=1.0` (the frozen vector-field M6 value):
oracle-pipeline position/gradient error both improve (0.0206->0.0168, 0.125->0.062) -- consistent with
the vector-field story, blending trustworthy gradient information into the tangent covariance helps.
But the realistic estimated-gradient pipeline gets *substantially worse* at `lambda_v=1.0`
(clean_point_rmse 0.0204->0.0511, a 2.5x regression; gradient_rmse 0.284->0.623, actually *worse* than
just doing raw local regression with no joint fitting at all, 0.576). `scalar_s_curve` shows zero
lambda_v effect either way, consistent with its z=0 degeneracy noted above -- not a contradiction, a
scenario property. Mechanism: covariance blending trusts the gradient input at strength `lambda_v`;
that's beneficial when the gradient is reliable (oracle) and actively harmful when it isn't (a
local-regression estimate whose own error is comparable in magnitude to the signal itself).

**Conclusion, directly answering the open question P4.0 flagged**: the vector-field-tuned
`lambda_v=1.0` does NOT transfer to the scalar pipeline in the realistic (non-oracle) case -- confirmed,
not just suspected. Not resolving this within this round (a proper scalar-specific `lambda_v` selection
needs its own tuning-seed procedure, same rigor as the original vector-field selection in Round 5, not
an ad hoc pick here), but flagging it as a clear prerequisite before S1/S2: the scalar branch needs its
own `lambda_v` (and possibly other hyperparameters) selected before those controlled experiments can
trust the frozen vector-field defaults.

Full data: `results/p4_1_scalar_oracle_ablation/` (`p4_1_long.csv`/`p4_1_summary.csv`/
`p4_1_decomposition.json`).

After P4.1's finding (fixed `lambda_v=1.0` helps the oracle scalar pipeline but hurts the realistic
one), user proposed making `lambda_v` itself adaptive: since gradient estimation already produces a
per-point confidence, scale `lambda_v` down for low-confidence points via some decreasing function
(they suggested inverse or similar) instead of relying on one fixed global compromise value.

### Confidence-scaled lambda_v (same day)

Checked first whether `velocity_confidence` (already existed) already touched the covariance blend --
it doesn't: it only modulates `_build_neighbors`'s reranking score and `_update_weights`'s directional
weighting; `_compute_local_tangent`'s `C = C_position + lambda_v * C_velocity` has always used the
same global scalar `lambda_v` for every point regardless of confidence. Confirmed this is a genuinely
new, not-previously-implemented mechanism before touching anything.

**Implemented in `scripts/velocity_manifold_fitter.py`** (the core class every frozen result in this
whole plan depends on -- treated carefully): new `lambda_v_confidence_scaling` ("none"/"linear"/
"power", default "none") and `lambda_v_confidence_power` (default 1.0) constructor parameters. "none"
preserves the old behavior exactly (now implemented as a per-point array holding the same scalar
everywhere, mathematically identical to the old unconditional scalar use, including the original
`lambda_v==0` fast-path optimization, now gated on `scaling=="none" and lambda_v==0` specifically so
it still fires in the common case). "linear" uses `lambda_v * confidence_i`; "power" uses
`lambda_v * confidence_i ** power`. Also added `effective_lambda_v` to the tangent diagnostics dict
and both new fields to `fit()`'s returned `algorithm_settings`. `fit_scalar_gradient_manfit` updated
to pass both parameters through.

**Backward-compatibility verification, since this touches the frozen protocol's foundation**: full
`pytest -q simulation` initially failed one test (`test_settings_are_serializable_scalars`) that
asserted the exact key-set of `algorithm_settings` -- expected, since two new keys were legitimately
added; updated the test to include them rather than weakening the check. More importantly, re-ran
`fit_vmf_variant` on `circle`/seed=43000/manfitvelo with the frozen config and the new default
("none" scaling) and compared `clean_point_rmse` against the value already stored in
`results/manfitvelo_benchmark/final_seed_metrics.csv` (computed before this change): matched to 14
decimal places. Confirms zero impact on the already-frozen vector-field protocol.

**Tested on `scalar_saddle`, 15 final seeds, "power" scaling with power in {1,2,4,8,16}, added as a
sixth pipeline family to `run_p4_1_scalar_oracle_ablation.py`** (oracle pipeline not included -- oracle
confidence is uniformly 1.0 by construction, so scaling is a no-op there). **Effect is real, monotonic,
and substantial**: `clean_point_rmse` falls from the flat `lambda_v=1.0`'s 0.0511 down to 0.0224 at
power=16 -- more than halving the damage relative to the safe `lambda_v=0` baseline (0.0204), without
touching the oracle pipeline's own `lambda_v=1.0` advantage (confidence≡1 there, so scaling never
kicks in). Median local-regression confidence on this scenario is ~0.78, which is why low powers (1-2)
only partially help -- a fairly aggressive exponent is needed before "uncertain" points get pushed
close to zero effective `lambda_v`.

**Deliberately not done**: did not pick a specific power (or scaling mode) as a new frozen value --
same reasoning as `lambda_v` itself originally required a proper tuning-seed selection procedure
(Round 5), not an ad hoc pick from one exploratory validation run. What's now established: the
mechanism itself works, moves in the right direction, is monotonic in the tested range, and is
zero-risk to the existing frozen vector-field protocol. This becomes a real candidate to include in
the scope of the scalar-branch hyperparameter selection already flagged as a prerequisite for S1/S2
(now "which scaling mode + which power", not just "keep or drop a fixed lambda_v").

Full data (including the new power variants): `results/p4_1_scalar_oracle_ablation/`.

### Redesign: `"power"` wasn't the user's original intent -- replaced with `1/(1+relative_error)` (same day)

User pushback after seeing the `power` results above: the original suggestion was to define the
lambda_v discount directly as a decreasing function of the gradient-estimation error itself, with
nothing new to separately tune. `confidence` genuinely comes from that error, but the `power`
exponent doesn't -- it's an extra free shape parameter I introduced on top, not itself derived from
anything. A fair correction, acknowledged directly, with a concrete principled alternative proposed
and confirmed via AskUserQuestion: `lambda_v_effective_i = lambda_v / (1 + relative_error_i)`, where
`relative_error_i` is the local ridge regression's own `ss_res_i/ss_tot_i` -- already computed inside
`estimate_gradient_confidence_from_neighbors` on the way to `confidence`, just never returned. No new
estimation, no new tunable shape parameter; `1/(1+x)` is already a well-behaved decreasing map from
`[0, inf)` to `(0, 1]` with no extra normalization constant needed. Kept (did not delete) the
already-validated `"power"`/`"linear"` modes as documented alternatives -- this is an addition, not a
replacement.

**Implemented**: `estimate_gradient_confidence_from_neighbors` now returns a 3-tuple
`(gradients, confidence, relative_error)`. `VelocityManifoldFitter` gained a new constructor
parameter `lambda_v_relative_error=None` (defaults to all zeros, i.e. no discount, when not supplied)
and a new `lambda_v_confidence_scaling="inverse_error"` mode; `_effective_lambda_v()` computes
`lambda_v / (1 + self.lambda_v_relative_error)` in that branch, untouched by `velocity_confidence`/
`power`. `"none"` (the frozen-protocol default) is byte-for-byte unchanged. `fit_scalar_gradient_manfit`
threads `lambda_v_relative_error` through the same way it already threads `confidence`.

**Backward-compatibility verification, same method as every other core-class change this session**:
re-ran `fit_vmf_variant` on `circle`/seed=43000/manfitvelo with the frozen config and default
(`"none"`) scaling; `clean_point_rmse` matched the value stored in
`results/manfitvelo_benchmark/final_seed_metrics.csv` to 14 decimal places
(`0.01707085007914008` vs. the stored `0.017071`). Full `pytest -q simulation`: 20/20, after extending
(not weakening) `test_settings_are_serializable_scalars` to include the new
`lambda_v_relative_error_mean` key in `algorithm_settings`.

**Smoke test on `scalar_saddle`/seed=43000** before the full run: not NaN, not identical to the flat
`lambda_v=1.0` result (0.0541 vs. 0.0562 for `clean_point_rmse` at that single seed) -- confirmed the
mechanism actually does something before committing to a 15-seed run.

**Full 15-final-seed result, added as a 7th pipeline (`estimated_lambda1.0_inverse_error`) alongside
the existing `power` variants in `run_p4_1_scalar_oracle_ablation.py`, reported without presupposing a
winner**:

| pipeline | `clean_point_rmse` | `gradient_rmse` |
|---|---:|---:|
| oracle, λ_v=1.0 (upper-bound reference) | 0.0168 | 0.062 |
| estimated, λ_v=0 (safe baseline) | 0.0204 | 0.284 |
| estimated, λ_v=1.0, power=16 (best power variant) | 0.0224 | 0.301 |
| estimated, λ_v=1.0, power=8 | 0.0276 | 0.338 |
| estimated, λ_v=1.0, power=4 | 0.0361 | 0.419 |
| estimated, λ_v=1.0, power=2 | 0.0428 | 0.494 |
| estimated, λ_v=1.0, power=1 (= linear) | 0.0454 | 0.528 |
| **estimated, λ_v=1.0, inverse_error (new)** | **0.0491** | **0.597** |
| estimated, λ_v=1.0 (flat, original, worst) | 0.0511 | 0.623 |

`scalar_s_curve` (the degenerate z≡0 scenario): identical across every variant including
`inverse_error`, as expected -- another cross-check consistent with the earlier finding that lambda_v
has no effect at all in that geometry.

**Why the effect is real but weak**: checked the actual `relative_error` distribution on this
scenario/seed -- median `0.227`, giving a median `lambda_v_effective ≈ 1/(1+0.227) ≈ 0.815`, i.e. only
roughly an 18% discount for a "moderately trustworthy" point. `power16` at the same median confidence
(~0.78) computes `0.78**16 ≈ 0.02`, an almost total shutoff. `power`'s exponent can be dialed
arbitrarily steep; `1/(1+x)` is bounded to `(0,1]` and only drops well below 0.5 once
`relative_error` is well above 1 -- that fixed, non-adjustable steepness is the direct cost of having
no extra free parameter to tune.

**Honest conclusion**: `inverse_error` is closer to what the user actually asked for (a genuine
decreasing function of the error itself, nothing extra to separately select), but on the one real
curved scenario tested it clearly underperforms the `power` family and does not beat the safe
`lambda_v=0` baseline. This isn't a bug in the design -- it's a real trade-off between "no extra
hyperparameter" and "discount steep enough to matter." Both modes are kept in the code as documented,
validated alternatives; neither has been frozen as a new default. The scalar branch's own
lambda_v/scaling-mode selection is still deferred to the proper tuning-seed procedure flagged
earlier in this round.

Full data (including the `inverse_error` row): `results/p4_1_scalar_oracle_ablation/` (pre-change
snapshot archived to `archive/p4_1_scalar_oracle_ablation_pre_inverse_error_20260812/`).

### Third iteration: `"rank"`, a genuinely zero-free-parameter mode (same day, user pushback again)

User pushback, this time on the framing rather than the mechanism: `power=16` was displayed at the
top of the comparison table looking like a recommended answer, but it was never actually tuning-seed
selected -- it's just the lowest value in an exploratory grid `{1,2,4,8,16}` evaluated directly on the
15 **final seeds**. The doc already said "not frozen as a new default," but the table's framing still
implied a conclusion, and the selection process itself (picking the best-looking grid point on final
seeds) is a real protocol-spirit violation even without a formal freeze. Fixed the misleading wording
in both existing tables in `current_plan.md` first (neutral "lowest value in the grid, not tuning-seed
selected" instead of bold/starred "best"), then asked what direction to take: user confirmed adding
another zero-free-parameter mode as a genuine `power`-strength/`inverse_error`-principle middle ground,
rather than either fixing `power`'s selection process or stopping at `inverse_error`.

**Confirmed design**: `"rank"` -- `lambda_v_effective_i = lambda_v * (1 - percentile_rank_i)`, where
`percentile_rank_i` is point `i`'s 0-indexed ascending rank of `relative_error` within the current
batch, divided by `(n-1)`. Purely ordinal: no exponent to pick (unlike `power`), no normalizing
constant to compute or pick (unlike `inverse_error`, whose weakness traced directly to
`relative_error`'s raw numeric scale happening to be small -- `rank` only cares about each point's
*relative* standing within the batch, immune to that). Degenerate case (`lambda_v_relative_error` all
zero, i.e. not supplied, or an exact tie across every point): rather than let an arbitrary `argsort`
order manufacture a fake spread from full lambda_v down to zero, falls back to a uniform 0.5 discount
for everyone -- verified directly (`_effective_lambda_v()` on an unconfigured default returns exactly
`lambda_v * 0.5` for every point, not an arbitrary per-point ordering).

**Implemented**: new `"rank"` branch in `_effective_lambda_v()`, reusing the existing
`lambda_v_relative_error` field (no new constructor parameter needed), pure numpy `argsort`. Legal-
value set extended to `{"none","linear","power","inverse_error","rank"}`. `fit_scalar_gradient_manfit`
needed no changes -- the `lambda_v_relative_error` plumbing from the previous round already covers it.

**Verification, same pattern as every prior core-class change**: circle/seed=43000/manfitvelo
`clean_point_rmse` still matches the frozen stored value to 14 decimal places under default `"none"`
scaling. Full `pytest -q simulation`: 20/20 (no new `algorithm_settings` keys added this round, so no
test update needed). Smoke test on `scalar_saddle`/seed=43000: `rank` gives `clean_point_rmse=0.0510`,
between `inverse_error`'s 0.0541 and flat `lambda_v=1.0`'s 0.0562 -- stronger than `inverse_error`,
right direction, not NaN.

**Full 15-final-seed result on `scalar_saddle`, added as an 8th pipeline
(`estimated_lambda1.0_rank`), reported alongside everything else without presupposing a winner**:

| pipeline | `clean_point_rmse` | `gradient_rmse` |
|---|---:|---:|
| oracle, λ_v=1.0 | 0.0168 | 0.062 |
| estimated, λ_v=0 (safe baseline) | 0.0204 | 0.284 |
| estimated, λ_v=1.0, power=16 (lowest grid value, not tuning-seed selected) | 0.0224 | 0.301 |
| estimated, λ_v=1.0, power=8 | 0.0276 | 0.338 |
| estimated, λ_v=1.0, power=4 | 0.0361 | 0.419 |
| **estimated, λ_v=1.0, rank (new, zero free parameters)** | **0.0443** | **0.511** |
| estimated, λ_v=1.0, power=2 | 0.0428 | 0.494 |
| estimated, λ_v=1.0, power=1 (= linear) | 0.0454 | 0.528 |
| estimated, λ_v=1.0, inverse_error | 0.0491 | 0.597 |
| estimated, λ_v=1.0 (flat, worst) | 0.0511 | 0.623 |

`scalar_s_curve`: identical across every variant again, third cross-check of the same "λ_v has no
effect in this degenerate geometry" finding, not a new surprise.

**Why the ordering makes sense**: `rank`'s implied median discount is exactly 0.5 (a "typical" point
gets cut in half), noticeably steeper than `inverse_error`'s ~0.815, putting it in the same rough
strength class as `power=1`/`power=2` (whose median discounts, `confidence^1≈0.78` and
`confidence^2≈0.61`, bracket 0.5) -- but nowhere near `power=8`/`power=16`'s near-total shutoff
(`confidence^8≈0.15`, `confidence^16≈0.02`).

**Honest conclusion**: `rank` is the only mode so far that is both zero-free-parameter *and* clearly
better than `inverse_error` -- but it still can't reach the strength of high `power` values, and that's
not a bug, it's the inherent cost of having no exponent to dial arbitrarily steep. `power`'s best-looking
values remain unvalidated by any real selection procedure (per the framing fix above) and must not be
read as a recommendation. Three iterations (`power` → `inverse_error` → `rank`) have now mapped out the
real trade-off space rather than picking a winner -- the scalar branch's own lambda_v/scaling-mode
selection is still deferred to a proper tuning-seed procedure, as flagged repeatedly through this round.

Full data (including the `rank` row): `results/p4_1_scalar_oracle_ablation/` (pre-change snapshot
archived to `archive/p4_1_scalar_oracle_ablation_pre_rank_20260812/`).

### Closing the loop: freeze `"rank"`, then a real tuning-seed selection for lambda_v's magnitude (same day)

User decision: stop litigating scaling-mode choice and just fix `lambda_v_confidence_scaling="rank"`
-- defensible on its own terms (zero free parameters, not a tuned number), no further comparison
needed. Before actually freezing it, flagged one thing worth doing first: every `rank`/`power`/
`inverse_error` number so far used `lambda_v=1.0` unconditionally, copied from the vector-field M6
frozen value and never itself chosen for the scalar branch -- and it's already known that
`lambda_v=1.0 + rank` clearly underperforms the safe `lambda_v=0` baseline on `scalar_saddle` (0.0443
vs. 0.0204). Freezing that pair as-is would mean committing to a config already known to be worse
than doing nothing. User agreed to do one proper tuning-seed selection for lambda_v's magnitude
before calling this frozen, mirroring exactly how the vector-field's own `lambda_v` was chosen
(Round 5: grid on tuning seeds, pooled score, safeguard against regressing any single scenario below
its own `lambda_v=0` baseline) rather than reopening the mode question.

**New `simulation/run_scalar_lambda_v_selection.py`**: grid `lambda_v ∈ {0.0, 0.5, 1.0, 2.0, 4.0}`,
scaling fixed to `"rank"`, scenarios `scalar_s_curve` + `scalar_saddle`, **tuning seeds only
(42000-42002) -- never final seeds for the selection step itself**. Score: pooled mean
`log(clean_point_rmse)` across scenarios. Safeguard: a candidate must not regress either scenario's
`clean_point_rmse` below its own `lambda_v=0` baseline (same rule as Round 5's vector-field
selection, adapted from 9 scenarios to 2).

**Result, reported as found rather than steered toward an expected answer**:

| lambda_v | pooled log(clean_point_rmse) | safe on both scenarios? |
|---:|---:|:---:|
| **0.0** | **-3.393** | -- |
| 0.5 | -3.149 | no |
| 1.0 | -3.077 | no |
| 2.0 | -3.031 | no |
| 4.0 | -2.942 | no |

Score gets monotonically worse as `lambda_v` increases; every candidate above 0 fails the safeguard on
`scalar_saddle`. **The selection winner is `lambda_v=0.0`.** Even with a principled, zero-parameter
discount mechanism, blending the estimated gradient's covariance into the tangent estimate does not
currently help on the one real curved scalar scenario available, given today's local-regression
gradient-estimation quality -- a legitimate, honestly-reported outcome, not a sign the mechanism was
built wrong. `scalar_s_curve` again contributed no differentiating signal, as expected throughout P4.1.

**Frozen**: `fit_scalar_gradient_manfit`'s protocol default for the scalar branch is now
**`lambda_v=0.0`** (recorded in `simulation/parameter_rules.md` §3b, mirroring §3a's vector-field
writeup) -- `lambda_v_confidence_scaling` is moot at `lambda_v=0` regardless of which mode. Final-seed
confirmatory numbers (reporting only): `scalar_s_curve` 0.0495, `scalar_saddle` 0.0204 median
`clean_point_rmse` -- consistent with every "safe baseline" row reported earlier in this round.

**This closes the "scalar branch's own lambda_v/scaling-mode selection is deferred" line that's been
repeated since P4.0/P4.1** -- S1/S2 no longer have this as a blocking prerequisite.

Full data: `results/scalar_lambda_v_selection/` (`tuning_seed_grid.csv`,
`tuning_seed_selection_audit.csv`, `final_seed_confirmation.csv`).

### Experiment S1: same manifold, different scalar landscapes (same day, proceeding immediately after the freeze)

User confirmed the lambda_v=0.0 freeze with a one-line framing suggestion ("if it needs explaining, just
say the error source in v isn't trustworthy enough") and said to proceed -- matches exactly what's
already written up above, no further edit needed there. Moved straight to S1, the next unblocked item.

**New `simulation/run_s1_scalar_landscape_family.py`**, deliberately mirroring `run_v1_field_family.py`'s
exact flat-unit-disk embedding/noise convention (N=480, sigma_X=0.05, z=0 plane) for direct
cross-experiment comparability with V1. Four landscapes, all from current_plan.md's own spec: single_basin
(x^2+y^2), double_well ((x^2-a^2)^2+c*y^2, a=0.5, c=1.0), saddle (x^2-y^2), and nonlinear_multimodal (a
log-sum-exp soft two-well, centers at x=+-0.5, tau=0.15 -- genuinely nonlinear gradient, unlike the
three polynomial landscapes). Gradient and f are both rescaled by the same median-gradient-norm
constant (same convention as V1's field rescaling; linear scaling keeps grad(f) consistent with
rescaled f automatically). Four pipelines, exactly current_plan.md's list: `raw_local_regression` (no
fitting at all), `geometry_only` (Local PCA denoises position, then post-hoc gradient estimation on
the denoised positions -- the scalar analog of M4's `downstream_velocity`), `joint_scalar_aware`
(`fit_scalar_gradient_manfit` at the just-frozen protocol, lambda_v=0.0), `oracle_gradient_joint`
(same config, ground-truth gradient substituted, reusing P4.1's oracle-isolation logic).

Added a metric that's never existed before in this pipeline: scalar-value RMSE (current_plan.md's own
"Scalar" metric layer). P4.1 never needed this -- only gradient/position metrics. Implemented as a
deliberately simple uniform-weighted kNN average of the noisy scalar observations at the final fitted
positions (`local_scalar_smooth`), documented explicitly as a new, simple baseline rather than
silently invented.

**Smoke test** (one landscape, one seed): no NaN/Inf, and a clean, consistent ordering on gradient
metrics across all four landscapes (raw > geometry_only > joint_scalar_aware > oracle, worst to best)
-- confirmed sane before running the full sweep.

**Full 15-final-seed result** (`gradient_rmse` / `gradient_angle_mae`, same ordering holds without
exception across all four landscapes):

| landscape | raw | geometry_only | joint_scalar_aware | oracle |
|---|---:|---:|---:|---:|
| single_basin | 0.585 / 27.0° | 0.419 / 18.6° | 0.305 / 10.0° | 0.071 / 2.6° |
| double_well | 0.661 / 30.0° | 0.516 / 22.3° | 0.435 / 13.6° | 0.077 / 2.3° |
| saddle | 0.555 / 26.5° | 0.384 / 18.6° | 0.265 / 10.5° | 0.041 / 1.7° |
| nonlinear_multimodal | 0.617 / 29.0° | 0.458 / 21.2° | 0.353 / 13.2° | 0.064 / 2.1° |

**A clean, worth-flagging finding**: even at the just-frozen lambda_v=0.0 (covariance blend off),
`joint_scalar_aware` still clearly beats `geometry_only` on gradient recovery, on all four landscapes.
The gradient-recovery gain must therefore be coming from the velocity-aware neighbor reranking
mechanism itself (using the estimated gradient to rescore/direct neighbor selection), not from the
covariance-blend term this round's whole selection process just found unhelpful -- these two mechanisms
have always been independent in `VelocityManifoldFitter` (the same fact `run_lambda_sensitivity.py`
already documented for the vector-field side), now cross-validated on the scalar side across four
distinct landscapes.

`clean_point_rmse` itself shows no consistent direction between `geometry_only` and
`joint_scalar_aware` (saddle: joint better, 0.0116 vs. 0.0124; single_basin/double_well: joint
slightly worse, 0.0131/0.0127 vs. 0.0124) -- small effect sizes, consistent with lambda_v=0 meaning
the tangent-covariance estimate itself is untouched, so any position difference is only the weaker,
indirect effect of neighbor reranking. `scalar_rmse` barely differs across pipelines (0.073-0.087,
close to the raw sigma_S=0.08 noise floor) -- the simple kNN-average scalar denoiser just isn't very
sensitive to position-fitting quality; doesn't undercut the gradient finding, which is a direct,
much stronger signal.

**Known gap, flagged not fixed**: `fit_scalar_gradient_manfit`'s own T/eta_g/theta/kappa/
outer_iterations/gradient_n_neighbors are still just function defaults, never tier-3 selected the way
the vector-field's shared hyperparameters were (parameter_rules.md SS3). Only lambda_v/scaling has
gone through a real selection so far (SS3b).

Full data: `results/s1_scalar_landscape_family/` (`seed_metrics.csv`, `summary_metrics.csv`,
`provenance.json`, `s1_report.html`).

### Experiment S2: same scalar landscape, different manifolds (same day, immediately after S1)

User said to proceed. S2 is the scalar analog of V2 (S1 held geometry fixed and varied the landscape;
S2 holds the landscape fixed and varies geometry) -- next unblocked item, no further hyperparameter
gap.

**New `simulation/run_s2_manifold_landscape_family.py`**, reusing `run_v2_manifold_family.py`'s four
embeddings verbatim via import (`phi`, `dphi_du`, `unit_normal`, `DOMAINS`,
`curvature_aware_k_for_manifold` -- not reimplemented), so the two experiments are directly
comparable. Shared landscape: S1's `landscape_nonlinear_multimodal` (also imported, not
reimplemented) -- the most genuinely nonlinear of S1's four, chosen to make "does the same nonlinear
landscape get harder or easier to recover as curvature changes" as clean a question as possible.

Since the four manifolds' native (u,v) ranges don't share a scale (longitude/colatitude vs.
arc-length-like coordinates vs. plain xy -- same issue V2 faced for its own dynamics), "the same
f(u,v)" is operationalized as an affine remap of each manifold's own domain to a shared [-1,1]x[-1,1]
reference square, with the universal landscape evaluated there. Recovering the landscape's *ambient*
gradient at each point needed the local pullback metric (g_ij = <dphi/du_i, dphi/du_j>) rather than a
naive per-axis chain rule -- reused the exact g11/g22/g12 + 2x2-inverse construction the existing
`scalar_saddle` scenario already uses (`scripts/run_field_informed_manfit_benchmark.py::scalar_data`),
not re-derived. This matters concretely for `saddle_surface`, whose chart has a nonzero g12 cross term
(the other three are orthogonal charts) -- an isotropic rescale would have been silently wrong there.

Same four pipelines/three metric layers as S1. k(n,d): the full two-stage curvature-aware rule
(reused from V2, not S1's plain Stage-1 -- three of these four manifolds are genuinely curved).
Smoke test cross-check: recomputed k matched the canonical/V2 frozen values exactly (flat_plane=37,
sphere_patch=21=half_sphere_tangent, swiss_roll=16, saddle_surface=26) -- same validation signal V2's
own smoke test relied on.

**Smoke test surfaced a real, mechanistically-explained counterintuitive result**: on `sphere_patch`,
`geometry_only`'s `gradient_rmse` (0.7415) was *worse* than doing no manifold fitting at all
(`raw_local_regression`, 0.6762) -- the only manifold where "denoise first" backfires. Checked it
wasn't a bug: the local design-matrix condition number for the ambient gradient regression roughly
doubled after Local-PCA denoising (median 2.9 -> 6.8, p90 4.3 -> 10.3, max 6.6 -> 16.5). Mechanism:
Local PCA projects each point's neighborhood onto a locally-estimated 2D tangent plane, making the
neighborhood nearly coplanar -- `estimate_gradient_from_neighbors` then solves an ambient R^3 least
squares problem that becomes ill-conditioned in the normal direction, even though the denoised
*positions* themselves are quite accurate (clean_point_rmse=0.0297, ~3% of the sphere's radius).
Confirmed consistent across 4 seeds, not a one-off. This explains why the "denoise geometry, then
regress gradient" two-stage strategy can backfire specifically on curved manifolds -- while
`joint_scalar_aware` (which never separates the two stages) stayed clearly ahead on `sphere_patch`
regardless (0.485), suggesting the joint mechanism is naturally robust to this particular trap.

**Full 15-final-seed result** (`gradient_rmse` / `gradient_angle_mae`):

| manifold | raw | geometry_only | joint_scalar_aware | oracle |
|---|---:|---:|---:|---:|
| flat_plane | 0.610 / 29.4° | 0.435 / 20.5° | 0.335 / 12.0° | 0.056 / 1.9° |
| saddle_surface | 0.573 / 28.0° | 0.461 / 22.9° | 0.318 / 12.1° | 0.096 / 3.9° |
| sphere_patch | 0.660 / 33.2° | **0.743 / 35.8° (worse than raw)** | 0.485 / 13.3° | 0.159 / 6.9° |
| swiss_roll | 0.558 / 28.3° | 0.569 / 28.0° (roughly tied with raw) | 0.375 / 13.1° | 0.231 / 6.6° |

**Core finding**: `joint_scalar_aware` beats `geometry_only` on all four manifolds, including the two
(`sphere_patch`, `swiss_roll`) where `geometry_only` has already lost its edge over doing nothing --
same conclusion as S1 (the gain traces to velocity-aware neighbor reranking, not the covariance-blend
term already found unhelpful), now cross-validated across curved geometries too. New information S1
couldn't surface (it was flat-only): the "denoise-then-regress" two-stage strategy doesn't just lose
its benefit on curved manifolds, it can actively backfire, while joint fitting avoids that trap --
about as clean an answer as this controlled experiment could give to "how does curvature affect
landscape recovery." `clean_point_rmse` degrades monotonically with curvature (flat_plane 0.011 ->
saddle_surface 0.017 -> swiss_roll/sphere_patch 0.037-0.039), matching V2's own curvature-vs-difficulty
finding, now cross-checked on the scalar side. `scalar_rmse` stays close to the noise floor across
pipelines, same explanation as S1.

**Known gap** (same as S1): `fit_scalar_gradient_manfit`'s own T/eta_g/theta/kappa/outer_iterations/
gradient_n_neighbors are still function defaults, never tier-3 selected.

Full data: `results/s2_manifold_landscape_family/` (`seed_metrics.csv`, `summary_metrics.csv`,
`provenance.json`, `s2_report.html`).

### P5: scoping check before the "full final-freeze rerun" (same day)

User said to proceed with P5's full final-freeze rerun. Checked first whether that literally means
what it says before spending the compute: since P0 (parameter freeze, C=0.60) happened first in the
execution order this whole round, and every phase since (P1-P4) used that frozen protocol from the
start, nothing on disk should actually be stale. Verified rather than assumed: directory mtimes show
`results/manfitvelo_benchmark/`'s data files were written right after the C=0.60 freeze (not before);
independently re-ran `fit_vmf_variant` on circle/seed=43000/manfitvelo today and matched the stored
value to 14 decimal places (same check already done twice this round for unrelated reasons). Scan A/B/C,
ambient-D (P1.2), V1/V2 are all similarly post-freeze; nothing since (P0.2 diagnosis, P1.2's
`noise_mode` field) changed any value they depend on. S1/S2 are today's work, already current. Grepped
the repo for "wilcoxon" -- nothing exists. So "rerun everything" would reproduce identical numbers at
real compute cost; the two genuinely unfinished P5 items are the paired Wilcoxon test (never built) and
a truly unified report (`build_experiment_report.py` is "v2" from before almost all of today's work).
Presented this finding and asked whether to proceed on the scoped-down understanding rather than
silently deciding unilaterally; user confirmed skip-rerun, do the two real gaps.

**`simulation/run_wilcoxon_test.py`** (new): reads `results/manfitvelo_benchmark/final_seed_metrics.csv`
directly, no recomputation. Tests the scenario/metric pairs the plan named (circle G1/G2,
flat_rotation_annulus V3, swiss_roll G1) plus one addition: swiss_roll G2
(`clean_point_rmse_rel`) -- the metric that actually flipped to favor M5 by marginal median after the
C=0.60 rerun, more consequential than the originally-named G1. Paired by seed on the `_rel` metrics
(cancels per-seed noise-level variation), `scipy.stats.wilcoxon` with `zero_method="pratt"`, both
two-sided and one-sided (M6-better) p-values reported.

**All 5 pairs significant in M6's favor at p<0.05**:

| scenario | metric | M5 median | M6 median | M6 wins | p (two-sided) | p (one-sided, M6 better) |
|---|---|---:|---:|---:|---:|---:|
| circle | G1 | 0.3692 | 0.3573 | 15/15 | 0.00006 | 0.00003 |
| circle | G2 | 0.3769 | 0.3589 | 15/15 | 0.00006 | 0.00003 |
| flat_rotation_annulus | V3 | 0.8222 | 0.8206 | 11/15 | 0.0181 | 0.0090 |
| swiss_roll | G1 | 0.6719 | 0.6633 | 13/15 | 0.0015 | 0.0008 |
| swiss_roll | G2 | 0.7152 | 0.7269 | 11/15 | 0.0479 | 0.0240 |

**A genuine statistical subtlety worth recording**: swiss_roll G2's marginal medians (each method's
own median across the 15 seeds, computed independently) put M5 ahead (0.7152 vs. 0.7269) -- this is
what originally read as a "flip" after the C=0.60 rerun. But that comparison throws away the seed
pairing. The paired statistic (median of per-seed M6-M5 differences) is -0.027, clearly favoring M6,
and the signed-rank test on those same paired differences is (marginally) significant in M6's favor
(p=0.048). Marginal-median and paired comparisons can point in different directions with this much
seed-to-seed variance at n=15 -- a real statistical phenomenon, not an error in either number -- and
the paired test is the correct one for this seed-matched design. Conclusion: swiss_roll is no longer
"M6 loses to M5," it's "M6 still wins but by the thinnest, only-marginally-significant margin of the
five tested pairs," alongside flat_rotation_annulus V3.

**`build_experiment_report.py` extended from "v2" to "v3"** (`results/experiment_report/index.html`,
old version archived to `archive/experiment_report_pre_v3_20260812/`): worked through the plan's own
"Claim language review" checklist item by item rather than a blanket find-replace --
(1) Q2b's half-sphere caveat (pooled-hyperparameter cost, not a capability regression, 44.6% gap vs.
locally-optimal hyperparameters); (2) Q1/SS5.1's "9/9" claim replaced with the actual Wilcoxon results
table (new SS5.2.3) and the swiss_roll statistical-subtlety explanation, not just a blanket "8/9";
(3) confirmed the M1 GraphVelo wording fix from P2.1 was already correctly reflected; (4) added the
M5-vs-M6 two-mechanisms caveat (neighbor reranking vs. covariance blend, not separately isolated on
the vector-field side), cross-referencing S1/S2's own isolation of exactly this on the scalar side.
Also fixed Scan C's section text, which still described the old absolute sigma_V grid -- replaced with
the actual relative-grid + shuffle-control design and its 8/9-scenario crossover finding. Added a new
SS6 "Extensions" section indexing P1.2 (ambient-D), P3 (V1/V2), and P4 (scalar branch: P4.1 ablation,
the lambda_v=0.0 freeze, S1, S2) with headline numbers and pointers to each experiment's own full
report, rather than re-embedding every figure -- those reports already exist and are complete.
Verified: 31 images all embedded as base64 (self-contained, no external refs), syntax check,
`pytest -q simulation` 20/20, ~11.4MB (same order of magnitude as the old v2 report's ~14.4MB).

### Closing the last known gap: scalar branch reuses ManfitVelo's theta/kappa (same day)

User's call on the one remaining flagged gap (scalar branch's T/eta_g/theta/kappa never tier-3
selected): don't run a separate search, just reuse the vector-field M6's own frozen values for
consistency. Entered plan mode for this given the size of the cascade it implied; started implementing
(`parameter_rules.md` SS3c drafted, `run_scalar_lambda_v_selection.py` given a `SHARED_KWARGS =
dict(inner_T=3, eta_g=0.7, theta=0.02, kappa=0.0)` constant) -- user paused before anything actually
ran, asking to confirm what the vector-field values even were first. Answered directly (T=3, eta_g=0.7,
theta=0.02, kappa=0.0, from `results/manfitvelo_benchmark/selected_hyperparameters.json`, same across
all 9 scenarios) and confirmed nothing had executed yet. User then confirmed: go ahead, keep
consistent with ManfitVelo's settings.

**Caught a real problem before it propagated anywhere**: `fit_scalar_gradient_manfit` calls `fit()`
`outer_iterations=4` times, each with `inner_T` steps -- literally copying `T=3` into `inner_T=3` means
4x the total position-update budget of M6's own single `fit()` call, at the same aggressive `eta_g=0.7`
per step. Checked before cascading into S1/S2: on `scalar_saddle`/seed=43000 at `lambda_v=0` (the safe
baseline), raw noisy input's own error is 0.0518; the *old* scalar defaults reached 0.0250 (clearly
better than nothing); the literal T/eta_g copy reached **0.0587 -- worse than not fitting at all**, and
checked it wasn't a few outliers (median per-point error nearly doubled, 38% of points landed >3x their
old error). Surfaced this to the user with the numbers rather than silently cascading a degraded
baseline into every downstream experiment. Presented three options (reuse only theta/kappa; accept the
literal copy anyway; rescale eta_g to match the total iteration budget); user picked reusing only
theta/kappa, leaving inner_T/eta_g at their own already-used values -- theta/kappa are what the
"consistency" argument was actually about (neighbor-reranking strength, the exact mechanism S1/S2's own
headline finding traces the whole joint-vs-geometry-only gap to), not the step-size parameters.

**Verified the fix**: theta/kappa-only change moves the same seed/scenario's safe baseline from 0.0250
to 0.0249 -- negligible, no overshoot.

**Cascaded reruns for self-consistency** (theta/kappa changed, so anything computed under the old 0.2/2.0
needed rerunning, even though it barely moved the numbers in practice):
- `run_scalar_lambda_v_selection.py`: rerun, **`lambda_v=0.0` still wins** the tuning-seed
  selection -- every candidate above 0 still regresses `scalar_saddle` below its own safe baseline.
- `run_s1_scalar_landscape_family.py`: rerun, full 15-seed sweep. Pattern held exactly (raw >
  geometry_only > joint_scalar_aware > oracle on all four landscapes); `clean_point_rmse` actually
  became a cleaner story than before -- `joint_scalar_aware` now consistently beats `geometry_only` on
  all four landscapes (0.0105-0.0106 vs. 0.0124), where the old run had an inconsistent direction.
- `run_s2_manifold_landscape_family.py`: rerun, full 15-seed sweep. Same gradient-recovery pattern held
  on all four manifolds, including `sphere_patch`'s "geometry_only worse than raw" finding (unaffected
  by this change since neither of those two pipelines calls `fit_scalar_gradient_manfit` at all --
  confirmed the mechanism, ambient-regression ill-conditioning after Local-PCA denoising, still applies).
  `clean_point_rmse` shifted: `joint_scalar_aware` now noticeably worse than `geometry_only` on
  `sphere_patch`/`swiss_roll` (0.0441 vs. 0.0313; 0.0423 vs. 0.0378) while staying better on
  `flat_plane` -- reported as-is rather than smoothed over.
- **Not rerun**: P4.1's own oracle-gradient ablation and the three-round confidence-scaling exploration
  (`power`/`inverse_error`/`rank`) -- diagnostic, not part of the frozen protocol, qualitative
  conclusions don't depend on theta/kappa. Flagged in `current_plan.md` as computed under the prior values
  rather than silently left looking current.

Old results archived to `archive/{scalar_lambda_v_selection,s1_scalar_landscape_family,
s2_manifold_landscape_family}_pre_shared_hparams_20260812/`. `pytest -q simulation`: 20/20 throughout
(no core-class changes this round, pure orchestration-layer parameter changes).

`simulation/parameter_rules.md` SS3c documents the full decision, the overshoot finding, the correction,
and the rerun results; SS3b updated with a pointer to the rerun. `current_plan.md`'s S1/S2 sections updated
with new numbers and a "parameter update" note each; P4.1's section gets a flag noting its numbers
predate this change.

### Still open / not yet done

Nothing in this round has been committed to git (still deferred until the whole plan is frozen, per
the user's standing instruction). The scalar branch's `theta`/`kappa` gap is now closed (SS3c); the
remaining known gap is narrower than before -- only `inner_T`/`eta_g`/`outer_iterations`/
`gradient_n_neighbors` are still function defaults, never tier-3 selected. Otherwise P0-P5 as scoped
this round are essentially complete -- next step is presumably the user's call on whether anything else
needs doing before the whole plan is considered frozen and committed.

### Repo cleanup for GitHub delivery (same day, immediately after)

User asked for a full audit of `scripts/`/`simulation/`: what's used, what's dead, core method vs.
experiment scenarios vs. retired code, clean consistent naming, and README updated -- prep for pushing
to GitHub as a minimal delivery for collaborators. Given the scope and destructive potential (renames,
deletions, and a genuinely surprising discovery about git state), entered plan mode: 3 parallel
read-only Explore agents inventoried `scripts/`, `simulation/`, and the top-level repo structure before
any design decision.

**Biggest finding**: almost everything actively in use is git-untracked. 6 of 10 `scripts/` files
(including the two most heavily-imported "hub" modules) and 23 of 27 active `simulation/` files had
never been committed. `archive/` (303M) and `results/` (50M) were untracked and not gitignored.
`reports/` (47 files, the pre-`simulation/`-era pipeline) was still tracked in git but deleted from
disk, never `git rm`'d.

User confirmed (via AskUserQuestion): stage git to match the current working tree but don't commit yet
(same standing "not until frozen" rule); delete confirmed-dead code outright rather than archiving;
retire the whole superseded `scripts/run_simulation_benchmark_v2.py` chain (which was the only reason
three `simulation/*.py` "legacy" files were being kept at all); rename the two files whose names
misrepresented their role as libraries. Follow-up corrections after plan review: don't overwrite any
collaborator's GitHub history when this eventually gets pushed (rename-on-conflict, not force-push --
noted for that future step, doesn't change this round's local-only scope); rename `plan2.2.md` to
`current_plan.md`; delete (not archive) `run_simulation_benchmark_v2.py`/`run_position_only_manfit_
diagnostic.py` outright, with a one-time archive snapshot kept only as a safety net given they were
untracked (git history alone wouldn't have recovered them).

**Changes made**:
- `scripts/run_field_informed_manfit_benchmark.py` renamed to `scripts/benchmark_scenarios.py`
  (misleading `run_*` name for what's actually a shared library most of `simulation/` imports from).
  `position_only_trajectory` (M5's real implementation) migrated in as an actual definition, replacing
  the previous import-passthrough from the now-deleted `run_position_only_manfit_diagnostic.py` --
  verified bit-exact identical output against the original before deleting it.
- Deleted confirmed-dead code: `pca_denoisers.py::oracle_pca_rank_sweep` (zero call sites anywhere);
  six unused functions in `scalar_potential_manfit.py` (`s_curve_projection_distance`,
  `gradient_cosine_error`, `local_tangent_projectors`, `project_vectors_to_local_tangent`,
  `summarize_fit_errors`, `format_metric_table`, plus `normalize_rows` once it became dead too --
  double-checked `estimate_gradient_confidence_from_neighbors` was NOT on this list, it's core to
  `fit_scalar_gradient_manfit`); two of three generators in `ambiguity_simulations.py`
  (`folded_hairpin_opposite_flow`, `near_intersection_incompatible_flow`, and the now-pointless
  `GENERATORS` dict -- kept `y_branch_outward_flow`, still used).
- Deduped `graphvelo_official_adapter.py::graph_hash` against the identical
  `simulation_baselines.py::neighbor_graph_hash` (same SHA-256-over-neighbor-array implementation in
  two files) -- the former now just calls the latter.
- Retired `scripts/run_simulation_benchmark_v2.py` and `scripts/run_position_only_manfit_diagnostic.py`
  (backed up to `archive/scripts/` first, since both were untracked). This was the only thing keeping
  `simulation/flat_manifold_potential_fields.py`/`flat_manifold_vector_fields.py`/
  `manifold_velocity_flows.py` alive (plan2.2.md/current_plan.md's own words: "legacy code, kept only
  because it's indirectly imported") -- confirmed via grep that nothing else referenced them, retired
  all three to `archive/simulation/` alongside `run_dt_sensitivity.py` (orphaned, predates the
  2026-08-12 C=0.60 rewrite, absent from the plan's own "still current" audit) and
  `ManfitVelo_Simulation_Weekly_Plan_v1.1.md` (fully superseded by `current_plan.md` since 2026-08-11).
- Renamed `plan2.2.md` to `current_plan.md`; global grep+replace across every file that referenced it
  by name (about 20 files -- `log.md`'s and `current_plan.md`'s own historical narrative entries were
  deliberately left referencing the old names/paths where they describe past events, matching this
  log's existing append-only convention; only live technical references were updated).
- Synced `methods_config.yaml`'s `shared_graph_k` and `C_d`→`C` fields to the actual post-C=0.60
  values in `results/manfitvelo_benchmark/selected_hyperparameters.json` (previously stale, flagged in
  its own header comment as a known "transitional inconsistency"); fixed the one place
  `build_experiment_report.py` read the old `C_d` dict shape.
- `.gitignore`: added `/archive/`, `/results/`, `.pytest_cache/`, `.Rhistory` (deleted the one stray
  0-byte `docs/.Rhistory` file rather than tracking it). `requirements.txt` added (numpy, pandas,
  scikit-learn, matplotlib, scipy, PyYAML, pytest -- none were declared anywhere before).
- `README.md` rewritten: new top-of-file repository-structure map (`scripts/`/`simulation/`/`archive/`/
  `results/`/`notebooks/`/`data/`/`docs/`, plus a file-by-file table for `scripts/`), full formal
  entry-point list for `simulation/` (previously incomplete relative to everything this whole plan
  actually built), full `results/` output inventory, dependency-install step, and a License section
  flagging that one still needs to be added and decided by the user before any public push.

**Verification, same rigor as every other core-touching change this session**: every renamed/edited
file re-`ast.parse`d; full repo grep confirmed zero remaining references to the retired names/paths
outside `archive/` and deliberate historical narrative; `pytest -q simulation` 20/20 at multiple
checkpoints through the process; re-ran `fit_vmf_variant` on circle/seed=43000/manfitvelo through the
new `benchmark_scenarios` import path and matched the stored value to 14 decimal places, confirming the
rename touched zero computational logic. `build_experiment_report.py` regenerated once at the end
(picks up the `methods_config.yaml` C_d/C text fix; no numeric results changed).

`git add -A` staged the full set (38 additions, 69 deletions -- the old `reports/`/`scripts/`/
`simulation/` tracked-but-superseded files -- 5 modifications; confirmed `archive/`/`results/` excluded
by the new `.gitignore`) but **not committed**, per the standing instruction.

Full before/after snapshots: `archive/scripts/{run_simulation_benchmark_v2,
run_position_only_manfit_diagnostic}.py`, `archive/simulation/{flat_manifold_potential_fields,
flat_manifold_vector_fields,manifold_velocity_flows,run_dt_sensitivity}.py`,
`archive/simulation/ManfitVelo_Simulation_Weekly_Plan_v1.1.md`.

### Repo cleanup, round 2: README trims, a real bug caught, and one more consolidation (same day)

User feedback on the round-1 README: several sections were more narrative than a reference README
needs -- trimmed the `lambda_v_confidence_scaling` bullet from a full walkthrough of all four modes'
tradeoffs down to a short pointer at the code's own docstring; dropped "were archived"/specific-date
phrasing throughout (Notebooks, Position + Potential Experiments, Simulation Benchmark Suite) in favor
of just stating current fact; shortened the entry-point inline comments and collapsed the
per-directory "Generated outputs" breakdown into one sentence; removed every remaining explicit date
stamp from the README (log.md/current_plan.md already track those precisely, no need to duplicate).

User also asked to actually delete (not just shorten the README row for) `scripts/
ambiguity_simulations.py`, and separately floated removing `simulation/run_half_sphere_diagnosis.py`.
Checked both before acting rather than complying blindly:

- `ambiguity_simulations.py` still had one genuinely live export, `y_branch_outward_flow` (used by
  `benchmark_scenarios.py`'s `y_branch` scenario generator, one of the 9 canonical scenarios) -- so a
  literal delete would have broken the canonical benchmark. Inlined `y_branch_outward_flow` (and its
  `_finish` helper, renamed `_y_branch_finish` to avoid ambiguity) directly into `benchmark_scenarios.py`
  instead, verified bit-exact identical output against the archived original (one apparent mismatch on
  first check was just `NaN != NaN` in a naive equality test, not a real difference), then deleted the
  now-empty standalone file (archived first, since it was untracked).
- `run_half_sphere_diagnosis.py` is different in kind from the other retired scripts: its output
  (`results/half_sphere_diagnosis/p0_2_summary.json`, the 44.6% pooled-hyperparameter gap number) is
  directly cited as evidence in both `current_plan.md`'s P0.2 writeup and the consolidated report's
  §5.7 interpretation -- not dead weight, a citation. Flagged this distinction explicitly rather than
  deleting on the same reasoning as the dead files; user reconsidered and confirmed **keep**.

**Real bug caught while investigating the user's report that Scan A/B figures wouldn't render**: both
were rendering as completely blank plots (empty axes, no data) in the consolidated report.
Root-caused to `run_stress_scans.py::plot_scan`'s shuffle-control filter,
`sub[sub.velocity_shuffled == False]` -- `velocity_shuffled` is only ever set for Scan C (`NaN` for
Scan A/B rows, which never have a shuffle dimension at all), and `NaN == False` is `False` under
IEEE754 comparison semantics, so this line silently filtered every Scan A/B row down to nothing before
any plotting happened. Confirmed via `pd.Series` toy example before touching the fix. Corrected to
`!= True` (`NaN != True` is `True`, correctly keeping those rows while still excluding real
`shuffled=True` ones). Regenerated the figures directly from the already-computed
`results/stress_scans/summary_metrics.csv` via `build_report(..., regenerate_figures=True)` -- no need
to rerun the underlying 15-seed scan, only the plotting step was broken. Visually confirmed both Scan
A and Scan B now show real per-scenario curves (previously-blank `A_sample_size_clean_point_rmse_rel.png`
and `B_position_noise_clean_point_rmse_rel.png` spot-checked directly). This bug had been present since
the Scan C redesign (P1.1) introduced the `velocity_shuffled` column without accounting for Scan A/B
never setting it -- present in every report generated since, only surfaced now because nobody had
looked closely at the Scan A/B panels specifically until this delivery-prep pass.

Re-verified after all of the above: `pytest -q simulation` 20/20, full syntax check, grep confirmed no
stale references outside `archive/` and this file's own historical narrative.
`build_experiment_report.py` regenerated once more to pick up the fixed Scan A/B figures (11.4MB ->
15.9MB, the previously-blank plots compressed unusually well as PNGs). `git add -A` re-staged the
updated set; still not committed.

### V1/V2 reports were under-using their own data (same day)

User flagged that V1/V2's figures looked odd -- only 3 of the 7 computed methods (Local PCA/
Position-only MANFIT/ManfitVelo, the same ablation trio as the canonical benchmark's own SS5.2) were
ever plotted or tabled in `v1_report.html`/`v2_report.html`, even though all 7 (M0-M6) were already
computed and sitting in `summary_metrics.csv`/`seed_metrics.csv` -- the other 4 (ambient noisy,
GraphVelo, Cosine Kernel, Joint Low-Rank) were paid for but never shown. Asked which fix they wanted
rather than guessing: one combined 7-bar chart, or mirror the canonical report's own two-section split
(primary external-baseline comparison + separate M4-M5-M6 ablation, `build_experiment_report.py`
SS5.1/SS5.2). User picked the two-section mirror, for consistency with the rest of the report suite.

Refactored both `run_v1_field_family.py` and `run_v2_manifold_family.py` identically:
`COMPARISON_METHODS` split into `PRIMARY_METHOD_ORDER` (ambient_noisy, graphvelo, cosine_kernel,
joint_low_rank, manfitvelo) and `ABLATION_METHOD_ORDER` (local_pca, position_only_manfit, manfitvelo,
reordered to match the main report's M4-M5-M6 convention); `plot_fields`/`plot_manifolds` generalized
into a `plot_methods(summary, output, methods, filename)` callable for either view; `build_report`
now renders two bar-chart figures and two tables instead of one. Regenerated both reports directly
from the already-computed summary CSVs (no need to rerun either 15-seed experiment) and visually
spot-checked all four new figures:

- V1 primary: M1/M2 sit at ~1.0 on every field as expected (they never touch position, so G1/G2 equal
  M0 by construction); M3 tracks M0 closely except spiking to ~3.85x (worse than noise) on the
  `nonlinear` field; ManfitVelo (M6) clearly best on all five fields (~0.15-0.2).
- V2 primary: M3 fails badly on every curved manifold (sphere_patch ~6x, saddle_surface ~4x, swiss_roll
  ~1.3x -- all worse than noisy input) while winning outright on `flat_plane` (~0.12) -- exactly the
  "global linear low-rank subspace can't represent real ambient curvature" pattern already documented
  for the canonical 9-scenario benchmark, now cross-validated on V2's controlled manifold family too.

`pytest -q simulation` 20/20 after the refactor; `git add -A` re-staged. The main consolidated report
was not regenerated for this change -- it only links to `v1_report.html`/`v2_report.html` by path, it
doesn't embed their figures, so nothing in it was stale.

---

## 2026-08-11 — Round 5: lambda_v re-selection, Scan C, Δt check, consolidated report v2

### Context

Preparing the simulation section for a paper targeting Nature Machine Intelligence / Nature
Computational Science. Discussion (chat, 2026-08-11) surfaced that `lambda_v` — the coefficient
weighting the velocity second-moment matrix in ManfitVelo's tangent-covariance blend, i.e. the one
parameter that actually implements "use velocity to improve manifold recovery" — was fixed at 0.1
and never included in the shared tuning grid. Agreed scope for this round: (1) write up lambda_v's
provenance and, if warranted, formally re-validate/re-select it (not initially planned as a
re-selection, see below for how that changed); (2) expand the report's method descriptions; (3) keep
the RNA-velocity-baseline comparison as-is (GraphVelo/Cosine Kernel/Joint Low-Rank) plus a discussion
paragraph rather than new implementations; (4) restructure M4/M5 as an ablation section rather than
primary competitors; (5) Δt sensitivity check (explicitly lower priority); (6) add Scan C (velocity
noise σ_V).

### lambda_v: from "validate only" to "re-select" (a methodological correction along the way)

Original plan: rerun the archived λ-grid study's sweep `{0,0.1,0.25,0.5,1,2}` under the current
protocol as **reporting only**, explicitly not re-selecting — the user's initial position was that
0.1 "makes sense" and re-tuning it risked looking circular for the paper's central claim.

That run (on **final seeds**, since it was framed as reporting-only) showed something unexpected: 8/9
scenarios improve *monotonically* out to the grid's ceiling (λ=2.0), only Swiss Roll shows the
U-shape the archived study warned about. This was surprising enough that the user decided to actually
change the default rather than just report the curve — which immediately created a methodological
problem: **the curve that motivated the change had used final seeds**, and this pipeline's one
inviolable rule is that final seeds never participate in any selection
(`selection_uses_final_seeds`/`final_seeds_used_for_selection` asserted `False` everywhere). Caught
this before acting on it; user agreed to redo the selection properly.

**Proper re-selection** (`run_lambda_sensitivity.py --seeds tuning`): swept the same grid on the 3
tuning seeds only, pooling via the same `tuning_score`-style mechanism `tune_shared_vmf` already uses
for T/eta_g/theta/kappa/theta_schedule. First pass (naive pooled mean of the *identity-anchored*
`tuning_score`) picked λ=2.0 — but manual inspection showed this made Swiss Roll's
`clean_point_rmse_rel` worse than its own λ=0 baseline (0.783 vs 0.741), i.e. "using velocity here is
worse than not using it at all" on one scenario, exactly the failure mode the archived study's own
multi-criterion safeguard was designed to catch. Added a proper automated safeguard (candidate must
not regress any single scenario below its own λ=0 baseline) — but the *first* safeguard
implementation used the same identity-anchored `tuning_score`, under which λ=2.0 turned out to
satisfy the safeguard after all (its aggregate score, which blends in large velocity-metric gains,
never dips below the λ=0 baseline even though position accuracy alone does). Caught this too: the
right metric for a "did this hurt geometry recovery" safeguard should be the report's own headline
metrics (G1/G2 position + *location*-anchored V3, not identity-anchored), not the T/eta_g selection
criterion. Recomputed with `headline_score = mean(log(clean_point_rmse_rel) + log(distance_to_
manifold_rel) + log(velocity_rmse_loc_rel) + log(joint_euler_state_rmse_rel))`: under this metric
even λ=2.0 is technically "safe" (Swiss Roll's blended score still beats its own baseline), but scores
only ~1% better than λ=1.0 in aggregate while gaining almost nothing further on `velocity_rmse_loc`
over λ=1.0 (0.916 vs 0.917) and giving up meaningfully more `clean_point_rmse`/`distance_to_manifold`
there — a Pareto-inefficient trade, not a genuine improvement. User confirmed: **λ_v=1.0**, a
Pareto-efficiency argument rather than a strict safeguard technicality.

Three passes were needed to reach a defensible number (see `select_lambda_v()` in
`run_lambda_sensitivity.py` for the final, headline-metric-based safeguarded selection) — recorded
here in full rather than only the final version, since the false starts (final-seed selection, then
identity-anchored safeguard) are exactly the kind of thing that should be visible in an audit trail
for a paper's central parameter.

**Updated**: `run_manfitvelo_benchmark.shared_vmf_grid()`'s hardcoded `lambda_v` (0.1→1.0),
`run_lambda_sensitivity.FROZEN_DEFAULT_LAMBDA_V`, `methods_config.yaml`, `parameter_rules.md` §3a (new).

### Full recompute with λ_v=1.0

Everything downstream of the shared VMF config had to be rerun: canonical benchmark (9 scenarios),
`run_sphere_scalability.py`, Scan A/B/C. Pre-update snapshots archived:
`archive/manfitvelo_benchmark_pre_lambda1.0_20260811/`, `archive/sphere_scalability_pre_lambda1.0_20260811/`,
`archive/stress_scans_pre_lambda1.0_20260811/`. All reruns `all_checks_pass: true`.

**Effect on Q1 (the headline finding)**: at λ_v=0.1, ManfitVelo (M6) beat Position-only MANFIT (M5) on
`clean_point_rmse_rel` in 5/9 scenarios. At λ_v=1.0, **M6 beats M5 on all 9/9 scenarios**, with
meaningfully larger margins on Curved Hairpin (0.385→0.321), Near Intersection (0.408→0.319), Saddle
(0.321→0.267), S-curve (0.296→0.263). A substantially stronger and more consistent result for the
paper's central claim.

### Scan C — velocity noise σ_V

Added `velocity_noise` override to `scripts/run_field_informed_manfit_benchmark.py::vector_data`
(mirrors the existing `position_noise` override; backward-compatible, default `None` preserves old
behavior). Grid: `σ_V ∈ {0.05, 0.10, 0.15, 0.20, 0.30}` (absolute, not relative — unlike Scan B, σ_V
is already uniform 0.10 across every scenario). `evaluate_condition()` still computes k(n,d) +
curvature-aware refinement explicitly for every Scan C point rather than special-casing it away, even
though it's provably a no-op here (k only reads position observations, and Scan C holds n/σ_X at
canonical — confirmed empirically: identical k at σ_V=0.05 and σ_V=0.30). Deliberate choice: keeps
"recompute k fresh at every scan point" a rule with no silent exceptions, verifiable rather than
assumed. Also confirmed as a free correctness check: Position-only MANFIT's `clean_point_rmse_rel` is
*exactly* invariant to σ_V (it never uses velocity for position updates), while ManfitVelo's varies
slightly (it does).

### Δt sensitivity check (`run_dt_sensitivity.py`)

One-time confirmatory check (Weekly Plan §7) on Curved Hairpin + Near Intersection: does the E_flow
method ranking stay stable across `τ ∈ {0.5, 1.0, 2.0} × τ0`? Reuses already-frozen canonical config,
final seeds, reporting only. Result: the *strict* all-pairs rank stability check fails (`False`) — but
only because of a near-tied swap between Local PCA and Position-only MANFIT on Near Intersection at
2×τ0 (0.497 vs 0.504, essentially noise). The practically relevant check —
**ManfitVelo remains the best (lowest E_flow) method at every tested τ, on both scenarios** — holds
(`manfitvelo_always_best_all_scenarios: true`). Added both checks to the script's output rather than
only the strict one, since the strict version alone would misleadingly read as "unstable" for what is
actually a robust headline finding.

### Report v2 (`build_experiment_report.py`, `results/experiment_report/index.html`, ~14.4MB, 29 figures)

- **Methods**: added a step-by-step ManfitVelo algorithm description (neighbor selection → kernel
  weights → joint tangent estimation → velocity projection → position update), plus fuller
  descriptions of GraphVelo's TSP objective, Cosine Kernel's formula, M3/M4/M5's mechanics.
- **RNA-velocity-literature discussion** (text only, no new experiments, per the confirmed scope):
  clarifies GraphVelo/Cosine Kernel/Joint Low-Rank are the directly-comparable competitors for this
  specific task (denoising an already-observed (position, velocity) pair via manifold structure);
  most of the wider literature (scVelo, veloVI, DeepVelo, cellDancer, UniTVelo, …) solves a different
  problem (estimating velocity from counts) and isn't comparable without a different simulation
  design; Dynamo's sparseVFC is methodologically closest in spirit but a separate framework, not
  implemented head-to-head.
- **Ablation restructuring**: primary headline table (§5.1) now shows only M0/M1/M2/M3/M6 (the
  fairly-rankable-on-G1/G2 set); M4/M5 moved to a new §5.2 "Ablation: M4 → M5 → M6", explicitly
  labeled a three-different-implementations pipeline-capability comparison, distinct from the
  single-parameter-isolated λ_v sweep (§5.2.2, both the tuning-seed selection curve and the
  final-seed confirmatory curve, with the safeguard audit table).
- Added Scan C figures (§5.6) alongside Scan A/B.

### Still open / deferred

- Everything from Rounds 1–4's deferred lists not touched this round: paired Wilcoxon test, phase-
  diagram experiment, real single-cell validation, comparison to more RNA-velocity baselines beyond
  discussion text, Near-Intersection reach audit, stale top-level `README.md`, extending the M3/Scan
  machinery to `scripts/simulation_baselines.py` standalone helpers.
- The near-tied Local-PCA/Position-only-MANFIT rank swap on Near Intersection at 2×τ0 (Δt check) is
  noted but not investigated further — doesn't affect the headline ManfitVelo finding.

---

## 2026-08-11 — Round 4: Scan A (sample size), Scan B (position noise), consolidated report

### Context

Asked to implement Scan A (sample size n) and Scan B (position noise σ_X — the Weekly Plan's
highest-priority stress test, direct test of Q1), across all 9 scenarios, then produce a single
consolidated report (methods / metrics / scenarios / parameters / results). Confirmed before
implementation (chat, 2026-08-11): Scan B only (not Scan C velocity-noise), all 9 scenarios (not the
plan's suggested representative subset), HTML report.

### Implementation: `simulation/run_stress_scans.py`

Key design decision, following the hard constraint written into `parameter_rules.md` §7 back in
Round 3: **k(n,d) and its curvature-aware refinement are recomputed fresh at every scan point** from
that point's own development-seed draws (cheap — one local-PCA probe sweep), while the **shared
VMF/Position-only hyperparameters (T, eta_g, theta, kappa, theta_schedule) stay frozen** at their
canonical pooled-search values (tier-3 parameters, not tier-2 — re-tuning the 162-candidate grid at
every scan point would both be enormously more expensive and wrong: that grid search is supposed to
answer "what's the one shared setting," not "what's best at this particular n/noise level").
This made the implementation much cheaper than a naive full-retune-per-point design would have been:
`fit_final_states` (already parameterized by an explicit `data`/`selected` pair, not hardcoded to the
canonical setting) is reused directly — a per-condition `selected` dict is built by deep-copying the
frozen canonical config and overriding only that scenario's `k`.

Grids: Scan A `n ∈ {200,400,800,1600}` (Weekly Plan §10); Scan B `σ_X ∈ {0.5,1.0,1.5,2.0,3.0} ×`
each scenario's own canonical σ_X (canonical σ_X already varies 0.02–0.05 across scenarios, so a
relative grid keeps the *relative* stress comparable). Both scans run on all 9 scenarios × all 15
final seeds. Total 9 scenarios × 9 scan points × 15 seeds × 7 methods = 8505 fits, ~11.3 minutes
end-to-end (background run, `sanity_checks.json`: `self_contained_html: true`,
`embedded_figure_count: 8` = `expected_figure_count`).

Sanity-checked before the full run: n↑ ⇒ error↓ monotonically and k scales up with n as the formula
predicts (Circle: n=200→k=18, n=1600→k=132); σ_X↑ ⇒ error trends up (not perfectly monotonic seed to
seed, as expected); at the 1.0× canonical point every recomputed k matched the already-frozen
canonical `shared_graph_k` value exactly — a good implicit consistency check that the scan machinery
reproduces the canonical run rather than silently diverging from it.

### Consolidated report: `simulation/build_experiment_report.py` → `results/experiment_report/index.html`

Pulls together `methods_config.yaml` (§1 Methods, §4 Parameters), `scenario_config.yaml` (§3
Scenarios), the metric prose from `metric_definitions.md` (§2 Metrics), the canonical run's headline
table + all 9 representative state figures, and both scans' curve plots (§5 Results) into one
self-contained HTML file (~10.6MB, all figures base64-embedded, 17 images total). Headline table
bolds the best of the four fairly-rankable geometry-fitting methods (M3–M6) per scenario/metric; M0
reference and M1/M2 (which never touch position) shown but excluded from "best" highlighting on
G1/G2 since their G1/G2 always equal M0 by construction.

**Bug caught before shipping**: the results table's method columns initially rendered in tuple-literal
order (M0, M2, M1, M3...) instead of numeric order — cosmetic only (highlighting logic itself was
correct: M3 legitimately wins the V3 velocity metric on Circle/S-curve per its known position/velocity
tradeoff, matching Round 3's finding), but confusing to read. Fixed `METHOD_ORDER` and rebuilt.

### Results summary

See `results/experiment_report/index.html` §5 for full tables and curves. Headline pattern: M4/M5/M6
cluster closely and clearly beat noisy input on every scenario (canonical single-point comparison);
Scan A shows the expected sample-complexity improvement (error falls as n grows, roughly consistent
across scenarios); Scan B is the most informative new result — for most scenarios M5 (position-only)
and M6 (ManfitVelo) track each other closely and diverge only mildly as σ_X grows, i.e. **the
canonical single-point Q1 finding (M6 ⪆ M5) is not an artifact of one specific noise level** — full
per-scenario curves in the report rather than reproduced here.

### Still open / deferred

- Scan C (velocity noise σ_V) — explicitly out of scope this round.
- Phase-diagram experiment (Δ over a (σ_X, σ_V) grid) — still not started.
- Everything else from Rounds 1–3's deferred lists (Δt sensitivity, Wilcoxon test, M3/Scan
  generalization to `scripts/simulation_baselines.py` standalone helpers, stale top-level
  `README.md`, Near-Intersection reach audit) unchanged.

---

## 2026-08-11 — Round 3: M3 baseline, Swiss Roll / Saddle Surface, Weekend deliverables

### Context

With the curvature-aware k(n,d) rule settled (Round 2), the user asked to move toward "freezing" the
protocol, in this priority order: (1) implement M3 Joint Low-Rank and remove the old Global PCA
baseline it was always meant to replace; (2) add the two Weekly-Plan Group-A scenarios that were
never implemented (Swiss Roll, Saddle Surface) and rerun; (3) either produce the plan's Weekend
"freeze" deliverables (`methods_config.yaml`, `scenario_config.yaml`, `metric_definitions.md`,
`parameter_rules.md`, `simulation_protocol.md`) or explicitly document that they're merged elsewhere.
Δt sensitivity check and the paired Wilcoxon test stay explicitly out of scope. Also asked for an
opinion (not implementation) on compute-budget documentation and numerical-failure reporting, with
the note that GraphVelo's own failure modes don't need dedicated space (vendored official method, not
the object of this study). All of the above confirmed before implementation (see chat, 2026-08-11).

Also flagged for the *next* round after this one: the next stress-test scan should probably be Scan B
(position noise σ_X) over Scan A (sample size n), since the Weekly Plan itself calls Scan B out as one
of the most important tests (it's the direct test of Q1). Whichever scan comes next, k(n,d) and its
curvature-aware refinement **must be recomputed fresh at every scan point** from that point's own
development-seed draws, never reused from the canonical setting — written into
`parameter_rules.md` §7 as a hard constraint for that future work.

### M3 Joint Low-Rank Denoising (replaces Global PCA)

Implemented `scripts/simulation_baselines.joint_low_rank_state` exactly per Weekly Plan §4: center and
block-normalize `X`/`V` by their own Frobenius norms, concatenate, truncate the joint SVD at
cumulative explained variance ≥ 0.90 (fixed, no scenario/ground-truth-based tuning), exact affine
unscale back to original units. Wired into both formal entry points (`BASE_METHODS`/`METHODS`,
`selected["joint_low_rank"]`, sanity-check keys renamed `global_pca_present` →
`joint_low_rank_present`, `run_sphere_scalability.py`'s per-point `tangent_projector_error` diagnostic
falls back to the generic post-hoc local-PCA path since M3 has no pointwise tangent projector, unlike
Global PCA's single global one).

**Debugging note, not a bug**: on `flat_rotation_annulus` M3 initially looked bad on
`clean_point_rmse_rel` (1.4) despite the plan predicting it should do well on a truly-linear velocity
field. Traced to a real, expected property: block-normalizing X and V to *equal* total weight means M3
trades position accuracy for velocity accuracy when V happens to be more redundant/lower-rank than X
(here V is an exact linear function of X, so the joint SVD spends its rank budget capturing V almost
perfectly — `velocity_rmse_id` dropped from 0.176→0.072 — at a mild cost to X: 0.050→0.070). This is
inherent to the specified equal-block-weighting design, not an implementation error.

**Result pattern**: M3 does badly (`clean_point_rmse_rel` 1.5–7.6, i.e. worse than noisy input) on
every scenario with real ambient/extrinsic curvature (Circle, S-curve, Half-sphere, Y-branch,
Near-Intersection, Swiss Roll, Saddle) — expected, since a global *linear* low-rank subspace can't
represent a manifold that genuinely curves through all 3 ambient dimensions. It's roughly break-even
on Curved Hairpin (0.86) and clearly helps specific *location*/*velocity* metrics (not
identity-position) on the exactly-flat Flat Rotation Annulus. This matches M3's documented role
("representative of global algebraic/low-rank structure") — it's supposed to be the baseline that
*can* fail on curved data, motivating the local/manifold-aware methods.

### Swiss Roll and Saddle Surface

**Noise model correction first**: before designing the new generators, discovered (by reading
`vector_data`/`add_noise` closely) that position noise in this codebase is **not** full isotropic
Gaussian noise — it's a single scalar draw per point pushed along the manifold's own analytic *normal*
direction (`N` in `vector_data`; e.g. Circle's `N=X` is the in-plane radial direction, Half-sphere's
`N=X` is the true sphere normal). Both new scenarios needed their own analytic unit normal (computed
via the cross product of the parametrization's Jacobian columns) to match this convention, not
arbitrary 3D noise.

**Saddle Surface**: `X(u,v)=(u,v,0.45(u²−v²))`, `u,v∈[-1,1]`, flow along `+u`, `a=0.45` reused from the
legacy `scalar_saddle` scenario's curvature scale. Worked correctly on the first try — negative/mixed
Gaussian curvature, contrasting the positive-curvature Half-sphere; `local_pca`/`position_only_manfit`/
`manfitvelo` all land around 0.30–0.35 (`clean_point_rmse_rel`), comparable to the other regular
scenarios.

**Swiss Roll — required a design fix**: the initial design (`t∈[1.5π,4.5π]`, the classic 1.5-turn
sklearn `make_swiss_roll` range) produced a scenario where *every* method, including ManfitVelo, was
**worse than noisy input at every tested neighborhood size down to k=5** (`clean_point_rmse_rel` never
dropped below ≈0.72). Root-caused with the same log-log-slope diagnostic used for the curvature-aware
k rule: the local-PCA normal-residual curve was already steeply *increasing* (log-log slope 1.0–1.7)
at the smallest probed k=8, with no low-bias regime visible at all — the classic Swiss-roll pathology
where Euclidean k-NN bridges across adjacent windings of the spiral even for very small
neighborhoods, since consecutive windings are close in ambient space despite being far apart along the
manifold. This is a well-known limitation of Euclidean-neighborhood methods on tightly-coiled swiss
rolls (motivating geodesic-distance approaches like Isomap in the literature) — informative as a
*stress test*, but the Weekly Plan classifies Swiss Roll as a **Group-A regular benchmark**, where
methods should have a fair chance. Tested 0.5/0.75/1.0-turn variants; **one full winding**
(`t∈[1.5π,3.5π]`) was the gentlest choice that (a) still looks like a genuinely rolled/curved sheet and
(b) gives every method a real shot — confirmed both by direct performance sweep (sweet spot ≈k=12–16,
`clean_point_rmse_rel`≈0.55–0.66) and by checking the curvature-aware detector lands close to that
sweet spot on the new geometry (`k=15`, vs. true optimum ≈12–16) without any per-scenario tuning of
the detector itself — the *generator* was fixed, not the detection rule.

### Lightweight numerical-failure audit

Added `nan_inf_count` (per scenario/seed/method) to both entry points' final metrics — counts
non-finite values in a method's `(X̂,V̂)` output, near-zero cost, output sanitized to 0 before metrics
are computed so one bad fit can't silently poison a median. Not a full failure-rate protocol (would
matter more once high-noise stress sweeps are implemented — deferred, see `simulation_protocol.md`).
GraphVelo's own numerical behavior intentionally isn't given more space than this shared column, per
the user's explicit note that it's a vendored official method, not the object of this study.

### Weekend deliverables

Generated all 5 requested files (not just a "merged into X" pointer note — see the confirmed rationale
in chat, 2026-08-11): `methods_config.yaml`, `scenario_config.yaml`, `metric_definitions.md`,
`parameter_rules.md`, `simulation_protocol.md`, all in `simulation/`. `methods_config.yaml` is an
explicit human-readable **snapshot** of `results/manfitvelo_benchmark/selected_hyperparameters.json`
(the latter regenerates fresh every run and is authoritative if they ever disagree).

### Final results (9 scenarios, full reruns, 15 final seeds)

`all_checks_pass: true` for both entry points. Frozen shared config shifted from the Round-2 numbers
now that 9 scenarios are pooled instead of 7 (expected — pooling more scenarios into the same
once-for-all grid search can shift which single candidate wins):

```
ManfitVelo (M6):        T=3, eta_g=0.7, theta=0.02, kappa=0.0, theta_schedule=flat (no longer "ramp"),
                         bandwidth_mode=variable, lambda_v=0.1
Position-only MANFIT:   T=3, eta_g=0.7
shared_graph_k:          circle=31, s_curve=40, curved_hairpin=14, flat_rotation_annulus=40,
                         half_sphere_tangent=20, y_branch=33, near_intersection=12,
                         swiss_roll=15, saddle_surface=26
```

Q1 headline (`clean_point_rmse_rel`, M5 Position-only MANFIT vs. M6 ManfitVelo, <1 = better than
noisy): ManfitVelo wins or ties in 7/9 scenarios (Circle 0.390→0.382, S-curve 0.296→0.283,
Flat-rotation 0.213→0.208, Half-sphere 0.803→0.779, Swiss-roll 0.690→0.667, Saddle 0.321→0.303,
Y-branch ≈tied 0.233/0.236); Curved Hairpin and Near-Intersection are now essentially tied or slightly
favor M5 (0.385/0.391 and 0.408/0.423) — a less dramatic gap than the 7-scenario Round-2 run, which is
an expected consequence of the pooled hyperparameter search shifting once 2 more scenarios joined the
pool (κ dropped from 2.0→0.0, `theta_schedule` from ramp→flat), not a regression in the underlying
method. Full numbers: `results/manfitvelo_benchmark/summary_metrics.csv`; pre-this-round snapshot for
comparison: `archive/manfitvelo_benchmark_pre_m3_scenarios_20260811/`.

`run_sphere_scalability.py` rerun for consistency (M3 replaces Global PCA there too); pre-round
snapshot: `archive/sphere_scalability_pre_m3_20260811/`; `all_checks_pass: true`.

### Still open / deferred

- All of Round 1's original deferred list minus M3/new-scenarios (now done): Δt sensitivity check,
  paired Wilcoxon test, Near-Intersection reach audit, stale top-level `README.md`.
- Scan A/B/C stress-test sweeps — not started; Scan B recommended next (see Context above).
- The residual Half-sphere-tangent/Circle gap vs. the original hand-tuned numbers (Round 2) is
  unchanged by this round — this round didn't touch the curvature-aware k rule itself.

### Test suite

`simulation/test_manfitvelo_benchmark.py` hardcoded the old 7-scenario counts (`735` final-metric
rows, `7*15*2` scale-audit rows, `7` embedded figures, `global_pca` in the expected method set) —
updated to the new 9-scenario numbers (`945`, `9*15*2`, `9`, `joint_low_rank`). Full kept test suite
(`test_global_pca`, `test_graphvelo_baselines`, `test_manfitvelo_benchmark`, `test_sphere_scalability`,
`test_velocity_augmented_tangent`) passes: 20/20.

---

## 2026-08-11 — Round 2: curvature-aware k(n,d) refinement

### Context

Round 1 (below) fixed the fairness violation but, in the process, exposed a real limitation: a
neighborhood rule that only sees `(n, d)` can't tell a flat scenario from a curved one, so it
overshoots the bias/variance-optimal `k` on every curved scenario — most visibly on
Half-sphere-tangent, where ManfitVelo regressed to *worse than noisy input*. Root-caused (see Round 1
"Results" section) to a classic local-linear bias/variance tradeoff: bigger `k` reduces
noise-averaging variance but also reaches further into the manifold's curvature, biasing the local
tangent/normal estimate. The user asked to design a curvature-aware rule to fix `k` selection
properly rather than leave it as a documented limitation.

### Design

Requirements: (1) **no ground truth** — must work from the noisy observations alone, so it stays
usable on real data later, not just this synthetic benchmark; (2) **data-adaptive, not
scenario-specific** — one rule, evaluated per scenario from its own data, not a hand override; (3)
**development seeds only** — never touch final seeds for selection, matching every other rule in this
pipeline; (4) fit the existing architecture (one scalar `k` per scenario, reused by Cosine Kernel /
Local PCA / M5 / M6) rather than a larger refactor to per-point adaptive neighborhoods.

**Method** (`benchmark_core.curvature_aware_neighbor_count`, `curvature_probe_k_grid`,
`local_pca_normal_residual`): sweep `k` from a small floor (`max(2d+2, 8)`) up to the existing
`neighbor_count(n, d)` ceiling (14-point geometric grid). At each `k`, fit `local_pca_denoise` and
read its already-computed `mean_local_spectrum` — take the sum of the smallest `ambient_dim − d`
eigenvalues as a population-mean "normal-direction residual" (how much local spread a rank-`d`
tangent plane fails to explain). Track this residual on a log-log(residual) vs. log(k) plot: its
*slope* first decreases (finite-sample eigenvalue-estimation bias shrinking as k grows) then, only
for genuinely curved geometry, turns around and increases again (curvature bias taking over). The
chosen `k` is the grid point right after that slope's minimum. On a flat manifold the slope never
turns back up, so this naturally reduces to the unchanged `neighbor_count(n, d)` ceiling — confirmed
empirically on `flat_rotation_annulus`/`s_curve` below. Averaged over the 3 `TUNING_SEEDS` per
scenario before computing the slope (reduces per-draw noise); `k` stays capped at the existing
formula's ceiling, so this can only shrink `k`, never grow it beyond the sample-complexity bound.

This is the same idea as the existing Hairpin `hairpin_reach_diagnostics()` (sweep a diagnostic vs.
`k`, stop once it crosses a threshold) generalized to not need branch labels or synthetic ground
truth — it works on the local-PCA eigenvalue spectrum alone, so it would carry over to real data.

**Rejected alternative**: an earlier version added a forward-tolerance extension past the slope
minimum (to rescue Near-Intersection, whose minimum was reached very early). Validated both variants
against `clean_point_rmse_rel` on all 7 scenarios' dev seeds — the plain "stop at the minimum, no
extension" rule scored better in aggregate (the tolerance variant fixed Near-Intersection but
reverted Circle's fix), so the simpler, parameter-light version was kept.

### Validation (dev seeds, before touching final seeds)

| scenario | `neighbor_count(n,d)` (old) | `clean_point_rmse_rel` @ old k | curvature-aware `k` | `clean_point_rmse_rel` @ new k |
|---|---|---|---|---|
| circle | 40 | 0.492 | 31 | **0.378** |
| s_curve | 40 | 0.323 | 40 | 0.323 (unchanged — correctly finds no curvature penalty) |
| curved_hairpin | 51 | 0.729 | 14 | **0.438** |
| flat_rotation_annulus | 40 | 0.278 | 40 | 0.278 (unchanged — correctly finds no curvature penalty) |
| half_sphere_tangent | 44 | 1.092 | 20 | **0.582** |
| y_branch | 51 | 0.265 | 33 | 0.283 (negligibly worse, <7%) |
| near_intersection | 51 | 0.450 | 12 | 0.475 (negligibly worse, <6%) |

4 clear wins, 2 exact ties (correctly detects the two truly-flat/near-flat scenarios and leaves them
at the formula's ceiling), 2 negligible losses. Net clearly better than the plain `(n,d)`-only rule.

### Integration

- `simulation/benchmark_core.py`: added `curvature_probe_k_grid`, `local_pca_normal_residual`,
  `curvature_aware_neighbor_count` (pure functions, no scenario coupling — reusable on any point
  cloud).
- `simulation/run_manfitvelo_benchmark.py`: added `curvature_aware_scenario_k()`, which loops the
  above over all 7 scenarios × `TUNING_SEEDS`, replacing the plain `neighbor_count(n,d)` call in
  `main()`. Saves per-scenario `k`-grid/residual/slope curves to
  `results/manfitvelo_benchmark/curvature_aware_k_diagnostics.csv` for audit. `selected["neighbor_
  count_rule"]` in `selected_hyperparameters.json` documents both the base formula and the
  refinement.
- `simulation/run_sphere_scalability.py`: `K` (used for Cosine Kernel / Local PCA / diagnostics) now
  reused directly from the main benchmark's frozen `half_sphere_tangent` value (same `n=480, d=2`)
  via `load_frozen_config()`, instead of recomputing the plain formula independently — keeps the two
  formal entry points consistent by construction rather than by coincidence.
- Verified end-to-end with a reduced-scope smoke test (3 scenarios, 2 dev + 2 final seeds,
  `all_checks_pass: true`) before committing to the full run.

### Final results (full reruns, 15 final seeds, curvature-aware k)

`shared_graph_k` (final, frozen): `circle=31, s_curve=40, curved_hairpin=14,
flat_rotation_annulus=40, half_sphere_tangent=20, y_branch=33, near_intersection=12` — matches the
dev-seed validation above exactly.

`clean_point_rmse_rel` (median over 15 final seeds), three-way comparison — original hand-tuned
(Round 0, pre-fairness-fix) vs. plain `(n,d)` formula (Round 1) vs. curvature-aware (Round 2):

| scenario | method | Round 0 (hand-tuned) | Round 1 (plain formula) | Round 2 (curvature-aware) |
|---|---|---|---|---|
| curved_hairpin | local_pca | 0.76 | 2.15 | **0.42** |
| curved_hairpin | position_only_manfit | 0.84 | 1.12 | **0.39** |
| curved_hairpin | manfitvelo | 0.55 | 0.83 | **0.37** |
| half_sphere_tangent | position_only_manfit | 0.56 | 0.88 | 0.80 |
| half_sphere_tangent | manfitvelo | 0.61 | 1.14 | **0.89** |
| near_intersection | local_pca | 0.35 | 1.15 | **0.41** |
| near_intersection | position_only_manfit | 0.27 | 0.71 | **0.41** |
| near_intersection | manfitvelo | 0.26 | 0.39 | 0.39 |
| circle | local_pca / manfitvelo | 0.30 / 0.31 | 0.48 / 0.47 | 0.35 / 0.44 |

Curved Hairpin is now *better than the original hand-tuned numbers* for every method — the
curvature-aware `k=14` genuinely beats the old ad-hoc "reach-safe" `k=4`, not just recovers from the
Round 1 regression. Half-sphere-tangent's ManfitVelo is back under 1.0 (`0.89`, better-than-noisy
again, vs. Round 1's `1.14`), though it doesn't fully return to Round 0's `0.61` — some residual gap
remains, which is expected: Round 2 fixes the *neighborhood-size* half of the curvature problem, not
every other now-shared hyperparameter (the pooled `theta_schedule=ramp`/`kappa=2.0` VMF default is
itself a compromise across all 7 scenarios, not re-optimized here). Near-Intersection's `local_pca`/
`position_only_manfit` are pulled back close to Round-0 levels; ManfitVelo was already fine and stays
fine. `run_sphere_scalability.py` corroborates the same pattern independently: ManfitVelo's
`clean_point_rmse` across D∈{3,5,10,20,50} moved from Round-1's `0.104–0.122` to Round-2's
`0.087–0.103` (Round-0 was `0.069–0.100`).

Full numbers: `results/manfitvelo_benchmark/summary_metrics.csv` /
`results/sphere_scalability/summary_metrics.csv` (current) vs.
`archive/manfitvelo_benchmark_pre_curvature_fix_20260811/` and
`archive/sphere_scalability_pre_curvature_fix_20260811/` (Round 1, pre-this-fix) vs.
`archive/manfitvelo_benchmark_pre_fairness_fix_20260811/` and
`archive/sphere_scalability_pre_k_fix_20260811/` (Round 0, original).

### Still open

- The residual gap on Half-sphere-tangent (and the mild Circle/Y-branch/Near-Intersection
  differences) suggests the *other* shared VMF hyperparameters (`theta_schedule`, `kappa`) might
  also benefit from a similar curvature-aware or per-scenario-class refinement, not just `k` — not
  attempted this round to keep the change scoped to neighborhood size.
- This curvature probe is only wired into the two formal `simulation/` entry points; it isn't used
  by `scripts/simulation_baselines.py`'s standalone helpers.
- All of Round 1's "Deferred to a later round" items (M3, Swiss Roll/Saddle, Δt check, Wilcoxon,
  Near-Intersection reach audit, freeze deliverables) are still deferred — this round only closed the
  curvature-aware-k item.

---

## 2026-08-11 — Round 1: fairness-principle fix + repo cleanup

### Context

A review of `ManfitVelo_Simulation_Weekly_Plan_v1.1.md` against the code found that the two formal
entry points (`simulation/run_manfitvelo_benchmark.py`, `simulation/run_sphere_scalability.py`) did
not yet satisfy the plan's core §4 fairness principle ("each method has one fixed
parameter-selection rule across all scenarios; scenario-specific tuning is forbidden in final
experiments"). `results/manfitvelo_benchmark/selected_hyperparameters.json` had ManfitVelo's
`k`/`T`/`eta_g`/`kappa`/`theta` hand-tuned to a *different* value for every one of the 7 scenarios
(e.g. `k`: 4 for Curved Hairpin vs. 120 for Flat Rotation Annulus), and Position-only MANFIT (M5)
similarly. Several other plan items (M3 Joint Low-Rank baseline, Swiss Roll / Saddle scenarios, Δt
sensitivity check, paired Wilcoxon test) were also not yet implemented.

Scope agreed with the user for this round (see chat, 2026-08-11):
1. **Fix the fairness violation first**, then run the existing 6-baseline + ManfitVelo comparison
   (M3, new scenarios, Δt check, Wilcoxon deferred to a later round).
2. **Archive, don't delete**, historical/unused files (many are `git`-untracked and unrecoverable if
   deleted outright).
3. Keep only the two formal `simulation/` entry points and their **direct dependencies**; archive
   everything else.

### What changed

**`simulation/benchmark_core.py`** — added `neighbor_count(n, d)`, the shared k(n,d) neighborhood
rule from Weekly Plan §4:

```
k(n, d) = clip(ceil(C_d * n**(4/(d+4))), 10, 200)
```

`C_d` is calibrated analytically (not by a performance sweep — the plan only requires `k(n0,d)` to
land in a reasonable band) so that a representative Group-A development scenario per intrinsic
dimension gives `k ≈ 40`:
- `d=1` anchor: Circle, `n0=360` → `C_1 = 40/360^0.8 ≈ 0.3606`
- `d=2` anchor: Flat Rotation Annulus, `n0=420` → `C_2 = 40/420^(2/3) ≈ 0.7132`

Resulting frozen `k` per scenario (all inside the target [20,60] band):

| scenario | n | d | k(n,d) | old hand-tuned k |
|---|---|---|---|---|
| circle | 360 | 1 | 40 | 20 |
| s_curve | 360 | 1 | 40 | 30 |
| curved_hairpin | 480 | 1 | 51 | 4 |
| flat_rotation_annulus | 420 | 2 | 40 | 120 |
| half_sphere_tangent | 480 | 2 | 44 | 20 |
| y_branch | 480 | 1 | 51 | 50 |
| near_intersection | 480 | 1 | 51 | 20 |

Applies uniformly to Cosine Kernel, Local PCA, Position-only MANFIT (M5), and ManfitVelo (M6) — see
"Curved Hairpin exception" decision below for why Hairpin gets no special case.

**`simulation/run_manfitvelo_benchmark.py`** — replaced the old `hairpin_vmf_grid` /
`tune_hairpin_vmf` (a grid search scoped to Curved Hairpin only, re-run every invocation) with:
- `shared_vmf_grid()` / `tune_shared_vmf()`: once-for-all grid search over
  `T ∈ {3,5,8} × eta_g ∈ {0.35,0.5,0.7} × kappa ∈ {0.0,1.0,2.0} × theta ∈ {0.02,0.05,0.1} ×
  theta_schedule ∈ {flat, ramp}` (162 candidates), scored by mean `tuning_score` pooled over **all 7
  scenarios × 3 tuning seeds** (42000–42002), picking a *single* config frozen for every final
  scenario. `k` is excluded from the grid (supplied per scenario by `neighbor_count`).
  `lambda_v=0.1`, `velocity_covariance_mode="uncentered"`, `velocity_trace_normalization=
  "match_position_trace"`, `bandwidth_mode="variable"` stay fixed — they were already constant
  across the old per-scenario configs, so re-searching them wasn't necessary.
- `shared_position_only_grid()` / `tune_shared_position_only()`: same idea for Position-only
  MANFIT's `(T, eta_g)`, 9 candidates.
- `hairpin_reach_diagnostics()` is kept, but only as a **geometry-validity audit** (it still checks
  that the frozen Hairpin separation satisfies the reach / cross-arm conditions); its own `k` output
  is no longer propagated into any method's operational neighbor count.
- `selected_hyperparameters.json` is now built fresh from `neighbor_count` + the two shared grid
  searches every run, instead of being loaded from a pre-existing, partly hand-curated JSON file
  (`load_frozen_config()` is no longer called here).

**`simulation/run_sphere_scalability.py`** — the hardcoded `K = 20` neighbor count (used for Cosine
Kernel / Local PCA / diagnostics on the S² scalability experiment) is now
`K = neighbor_count(N, INTRINSIC_DIMENSION)` = 44, matching the Half-sphere-tangent scenario since
both use `n=480, d=2`. This script already pulls its VMF / Position-only MANFIT configs from
`results/manfitvelo_benchmark/selected_hyperparameters.json` via `load_frozen_config()`, so it
automatically inherits the new shared, cross-scenario-frozen hyperparameters without further code
changes.

### Decision: Curved Hairpin gets no k(n,d) exception

`hairpin_reach_diagnostics()` picks a "reach-safe" `k` purely from geometry (cross-arm contamination
< 5%, no method results involved) — a legitimate, non-circular rule, but a *different* rule from the
general k(n,d) formula, and the plan text doesn't carve out an exception for it. Asked the user
whether to (a) keep the reach-safe k for Hairpin specifically, or (b) apply k(n,d) uniformly with no
exception, accepting that Hairpin metrics for every method would likely look much worse (heavy
cross-arm contamination at k≈51 vs. the previous k=4). **Decision: (b), no exception** — this is
what Weekly Plan §4 literally specifies, and the resulting degradation is itself the point of the
Group-B geometry stress test rather than a bug. Confirmed via a 2-seed dev smoke test: under k=51,
Local PCA and Position-only MANFIT both regress to ≈ no-improvement over noisy input on Hairpin,
while ManfitVelo still improves on most metrics (`clean_point_rmse_rel≈0.63`) — informative evidence
for Q1 (velocity information helps most exactly when position-only geometry is degraded).

### Repo cleanup (archived, not deleted)

Moved to `archive/` (mirrors the pre-move layout under `archive/scripts/`, `archive/simulation/`,
`archive/reports_legacy/`, `archive/results/`), keeping only the two formal entry points and their
load-bearing dependency closure (verified by importing both entry points and inspecting
`sys.modules` for every `scripts.*` / `simulation.*` module actually loaded — not just by reading
imports, since a few "legacy-looking" `run_*.py` scripts turned out to be load-bearing):

- **`scripts/` kept** (imported, directly or transitively, by the two formal entry points):
  `velocity_manifold_fitter.py`, `pca_denoisers.py`, `graphvelo_official_adapter.py`,
  `simulation_baselines.py`, `ambiguity_simulations.py`, `scalar_potential_manfit.py`,
  `html_report_utils.py`, `run_field_informed_manfit_benchmark.py`,
  `run_position_only_manfit_diagnostic.py`, `run_simulation_benchmark_v2.py`. The last three are
  named like one-off diagnostic scripts but actually *host* shared utility functions
  (`vector_data`, `hairpin`, `fit_vmf_variant`, `position_only_trajectory`, `Config`/`make_data`)
  that the formal entry points import — they could not be archived. **Follow-up worth doing later**:
  extract those shared utilities into a proper `scripts/simulation_shared.py`-style module so the
  formal pipeline no longer depends on modules named/shaped like legacy one-off scripts.
- **`scripts/` archived**: `build_portable_low_noise_report.py`, `build_vmf_benchmark_v2_report.py`,
  `check_benchmark_integrity.py`, `fitness_landscape_pca_report_panels.py`,
  `geometry_velocity_metrics.py`, `manfit.py`, `manfit_ours.py`,
  `palantir_gradient_field_before_after.py`, `prepare_protein_latent_paper_data.py`,
  `run_application_geometry_report.py`, `run_focused_vmf_benchmark.py`,
  `run_local_pca_bandwidth_diagnostic.py`, `run_low_noise_ambiguity_vmf_benchmark.py`,
  `run_parameter_sweep.py`, `run_simulation_benchmark.py` (the non-`_v2` legacy version),
  `run_vmf_benchmark_v2_concise.py`, `run_vmf_benchmark_v2_local.py`, `train_test_evaluation.py`,
  `reference_implementations/`.
- **`simulation/` archived**: `evaluate_sasaki_joint_metric.py`, `sasaki_joint_metric.py`,
  `test_sasaki_joint_metric.py`, `run_velocity_augmented_main_benchmark.py`,
  `test_velocity_tangent_scaling.py`, `generate_flat_manifold_potential_fields.py`,
  `generate_flat_manifold_vector_fields.py`, `generate_manifold_velocity_flows.py`,
  `serve_website.py`, `website/`, `data/` (cached generator output), `results/` → moved to
  `archive/simulation/results_legacy/` (Sasaki-metric / velocity-augmented / tangent-scaling study
  outputs — distinct from top-level `results/`, which holds only the two formal entry points'
  outputs). `flat_manifold_potential_fields.py`, `flat_manifold_vector_fields.py`,
  `manifold_velocity_flows.py` were **kept**: they're imported transitively via
  `run_simulation_benchmark_v2.py`.
- **`reports/` archived in full** → `archive/reports_legacy/` (per `code_cleanup_manifest.md`, this
  whole tree was already legacy/superseded before this round).
- **`results/` archived**: `field_informed_manfit_benchmark/`, `graphvelo_cosine_benchmark/`,
  `unified_manfitvelo_benchmark/`, `unified_manfitvelo_benchmark_legacy_20260803/`,
  `vmf_benchmark_v2/`, `vmf_benchmark_v2_concise/`. **Kept**: `manfitvelo_benchmark/`,
  `sphere_scalability/` (the two formal outputs).
- A snapshot of the **pre-fix** `results/manfitvelo_benchmark/` (the version with per-scenario
  hand-tuned parameters) was copied to `archive/manfitvelo_benchmark_pre_fairness_fix_20260811/`
  before rerunning, for before/after comparison.
- `__pycache__/` directories removed (regenerable build artifacts, not experiment records).
- All moves used `mv`, not `git mv`/`git rm` — nothing has been committed; the working tree just
  reflects the new layout. Not staged or committed since the user hasn't asked for that yet.

Verified after every archiving step that `import simulation.run_manfitvelo_benchmark` and
`import simulation.run_sphere_scalability` still succeed.

### Known follow-up: stale top-level README.md

The project-level `README.md` (one directory up from `simulation/`) still lists paths under
`reports/` (e.g. `reports/application_geometry/`, `reports/parameter_sweep/`) and
`scripts/geometry_velocity_metrics.py` that this round moved into `archive/`. It documents the whole
project, not just this simulation suite, and reconciling it fully was out of scope for this round —
flagging here rather than silently leaving it stale.

### Deferred to a later round (not in scope this time)

- M3 Joint Low-Rank Denoising baseline (block-normalized joint `[X,V]` SVD) — not implemented;
  `global_pca` is still used as its stand-in.
- Swiss Roll / Saddle Surface scenarios (mentioned in the plan's §3/§13 scenario lists, no generator
  exists yet).
- Δt sensitivity check on Curved Hairpin (§7).
- Paired Wilcoxon signed-rank secondary check for the M5-vs-M6 ablation (§12).
- `methods_config.yaml/json`, `scenario_config.yaml/json`, `metric_definitions.md`,
  `parameter_rules.md`, `simulation_protocol.md` — the plan's Weekend "freeze" deliverables; this
  log + the code comments in `benchmark_core.py`/`run_manfitvelo_benchmark.py` serve as the interim
  record.
- Near Intersection's separation (`0.13`, hardcoded) has no reach-based geometry audit analogous to
  Hairpin's — worth a similar diagnostic later.
- ~~**Curvature-aware neighborhood rule.**~~ **RESOLVED same day — see "Round 2" section above.**

### Results

Full run completed: `results/manfitvelo_benchmark/` (15 final seeds 43000–43014, 7 scenarios, all
`sanity_checks.json` invariants pass — `all_checks_pass: true`, including
`final_seeds_used_for_selection: false` and `hairpin_selection_uses_method_results: false`).
Report: `results/manfitvelo_benchmark/final_report.html`. Frozen shared config (identical across all
7 scenarios except `k`, confirmed programmatically):

```
ManfitVelo (M6):        T=3, eta_g=0.5, theta=0.02, kappa=2.0, theta_schedule=ramp,
                         bandwidth_mode=variable, lambda_v=0.1
Position-only MANFIT:   T=3, eta_g=0.35
k(n,d):                 circle=40, s_curve=40, curved_hairpin=51, flat_rotation_annulus=40,
                         half_sphere_tangent=44, y_branch=51, near_intersection=51
```

(Both grid searches picked `theta_schedule=ramp` as the single best default — previously this was
hand-set only for Curved Hairpin; it turns out to generalize well across scenarios on average.)

**Headline comparison, old (per-scenario-tuned) vs. new (fairness-fixed) — `clean_point_rmse_rel`
(median over 15 final seeds; <1 means better than noisy input; primary Q1 ablation is
Position-only MANFIT [M5] vs. ManfitVelo [M6]):**

| scenario | M5 old→new | M6 old→new | note |
|---|---|---|---|
| circle | 0.31 → 0.43 | 0.31 → 0.47 | mild regression, k 20/20→40/40 (larger, more averaging bias) |
| s_curve | 0.28 → 0.38 | 0.29 → 0.32 | mild regression |
| curved_hairpin | 0.84 → **1.12** | 0.55 → 0.83 | M5 crosses to *worse than noisy*; M6 degrades but stays clearly better than noisy — see decision above |
| flat_rotation_annulus | 0.13 → 0.35 | 0.11 → 0.27 | k 120→40 removes a lot of the old (scenario-specific) noise-averaging advantage |
| half_sphere_tangent | 0.56 → 0.88 | 0.61 → **1.14** | M6 crosses to *worse than noisy* — new, notable finding, see below |
| y_branch | 0.23 → 0.35 | 0.20 → 0.25 | mild regression, k basically unchanged (50→51); driven by the now-shared T/eta_g/theta/kappa |
| near_intersection | 0.27 → 0.71 | 0.26 → 0.39 | M5 close to no-improvement; M6 still clearly better than noisy |

**Interpretation.** Every scenario got somewhat worse for every geometry-fitting method once the
single frozen rule replaced per-scenario tuning — expected, since the old numbers partly reflected
tuning to each scenario's own test set rather than a genuinely shared rule. Two results stand out:

1. **Curved Hairpin and Near Intersection (Group B, the small-reach/ambiguous-neighborhood stress
   tests) now show a clear gap opening up between M5 and M6**: Position-only MANFIT loses essentially
   all its advantage over noisy input (Hairpin: 1.12, i.e. worse than doing nothing), while ManfitVelo
   keeps a real improvement (Hairpin: 0.83, Near Intersection: 0.39). This is exactly the kind of
   evidence Q1 ("does velocity information improve manifold recovery?") is designed to surface — it
   only shows up now because the neighborhood is no longer secretly tuned to be small/safe for M5.
2. **Half-sphere tangent field is a new concern**: ManfitVelo now performs *worse than the noisy
   input* on `clean_point_rmse_rel` (1.14) and `distance_to_manifold_rel` (1.13), a reversal from
   before (0.61 / 0.60). **Root-caused on 2026-08-11 (see below): this is entirely `k`, not
   `theta_schedule`/`kappa`.**

   #### Root-cause diagnosis (2026-08-11)

   A 2×2×2×2 factorial ablation on `half_sphere_tangent` (dev seeds, 3 reps), varying
   `k∈{20,44}, theta∈{0.02,0.05}, kappa∈{0.0,2.0}, theta_schedule∈{None,ramp}` while holding
   `T=3, eta_g=0.5` fixed, cleanly separates by `k` alone — every `k=20` combination scores
   `clean_point_rmse_rel ≈ 0.58`, every `k=44` combination scores `≈ 1.10–1.16`, regardless of the
   other three knobs (max spread within each `k` group < 3%). `theta`/`kappa`/`theta_schedule` are
   not the cause.

   A finer `k` sweep on `half_sphere_tangent` (dev seeds) shows a smooth, monotonically *increasing*
   error from `k=12` (0.52) through `k=100`+ — no sweet spot in the tested range; every `k` above the
   scenario's old hand-tuned value (20) is already past the point of diminishing returns. Repeating
   the same sweep on `flat_rotation_annulus` (literally flat — embedded as `z=0`, zero curvature)
   shows the *opposite*: error falls monotonically from `k=12` (0.57) to `k=100` (0.21) — larger
   neighborhoods purely help there, which is exactly why the old (unfair) hand-tuning had picked
   `k=120` for it. `circle`, `curved_hairpin`, `s_curve`, `y_branch`, `near_intersection` (all
   genuinely curved or branch-proximate) each show a U-shaped or monotonically-increasing curve with
   an optimum at or below their old hand-tuned `k`, well below what `neighbor_count(n,d)` now assigns
   them.

   **Mechanism**: this is the standard local-linear bias/variance tradeoff, just not one
   `k(n,d) = C_d·n^{4/(d+4)}` — a function of sample size and *intrinsic dimension only* — can
   capture. A larger neighborhood averages out more position noise (variance ↓) but also reaches
   further along the manifold's curvature, biasing the local tangent/normal estimate by an amount
   that grows with the neighborhood's geodesic radius (bias ↑). On a genuinely flat patch
   (`flat_rotation_annulus`) that bias term is ≈0, so bigger `k` is strictly better. On curved
   patches it isn't, and `half_sphere_tangent` is simply the most curved of the 7 scenarios (true
   curvature in both intrinsic directions, vs. 1-D curvature for Circle/S-curve/Hairpin, or exactly
   zero for the piecewise-linear Y-branch/Near-Intersection branches away from the branch point) — so
   the same formula-driven `k` increase (20→44, +120%) hurts it the most in absolute terms.

   **Conclusion**: not a bug in this round's implementation, and not unique to Half-sphere — it's a
   known, now-confirmed limitation of a neighborhood rule that only sees `(n, d)` and not manifold
   curvature. It quietly affects every curved scenario (contributes to the mild circle/s_curve/
   y_branch/near_intersection regressions in the table above too); Half-sphere just makes it most
   visible. **Not fixed this round** — re-adding a per-scenario `k` override would undo the fairness
   fix by definition. Left as a precisely-scoped follow-up (see below) rather than a vague "open
   question."

Full per-method, per-metric numbers: `results/manfitvelo_benchmark/summary_metrics.csv` (new) vs.
`archive/manfitvelo_benchmark_pre_fairness_fix_20260811/summary_metrics.csv` (old, for reference).

**`run_sphere_scalability.py` rerun** (K=20→44 fix; pre-fix snapshot in
`archive/sphere_scalability_pre_k_fix_20260811/`, `sanity_checks.json` all pass): ManfitVelo's
`clean_point_rmse` increased consistently across every ambient dimension D∈{3,5,10,20,50} (e.g.
D=3: 0.069→0.104, D=50: 0.100→0.122) — same direction and similar relative size as the
Half-sphere-tangent regression above, and using a fully independent script/dataset. This is
corroborating evidence that the shared VMF config (in particular the larger `k` and the
`theta_schedule=ramp`/`kappa=2.0` combination inherited from the pooled grid search) genuinely
underperforms on curved 2-D surfaces specifically — not a fluke of one run. Reinforces that this is
worth a dedicated follow-up rather than a one-off tweak.

## Pushed to GitHub: merged with collaborator's velocity_tangent_weight

`git fetch` before pushing showed `origin/main` one commit ahead — Jingyuan Hu's "Add
velocity-augmented tangent fitting" (Jul 14), adding `build_figure2_html_report.py`,
`figure2_geometric_knn_metrics.py`, `figure2_manifold_projection_metrics.py`,
`figure2_reconstruction_metrics.py`, `plot_figure2_vector_fields_1x4.py`,
`potential_from_gradient.py`, `notebooks/simulations/figure2_all_in_one.ipynb`, and a change to
`scripts/velocity_manifold_fitter.py` adding `velocity_tangent_weight` — an independently-built
mechanism for the same idea as `lambda_v` (blend a velocity-derived covariance into the tangent
estimate), but numerically different: unit-normalized velocity *directions* rather than raw
vectors, per-neighbor `velocity_confidence` discount, and `trace(C_position)`-direct scaling
rather than `lambda_v`'s exact trace-matching.

Resolution (user confirmed: keep both, do a real merge): kept `velocity_tangent_weight` as an
independent, additive, keyword-only parameter (default `0.0`) applied on top of whatever `C`
`lambda_v` produces, right before final symmetrization — it never touches the `lambda_v` code
path. Verified bit-exact against the stored `circle`/seed=43000/manfitvelo regression value
(`0.01707085007914008`) with `velocity_tangent_weight=0.0`, and machine-epsilon equivalent
(`np.allclose(..., rtol=1e-10, atol=1e-12)`) to the collaborator's original formula when
`lambda_v=0.0, velocity_tangent_weight>0`. Added 3 dedicated tests
(`simulation/test_velocity_augmented_tangent.py`) covering the no-op-at-zero case, additivity/
independence from `lambda_v`, and the negative-value validation error — full suite 23/23 passing.

The other 6 files + notebook had zero overlap with this session's work and were brought in
unchanged (`git show origin/main:<path>`, verified byte-identical via `cmp`).

Merged via `git merge origin/main`, resolving the single conflicting file
(`scripts/velocity_manifold_fitter.py`) by keeping the already-reconciled version (confirmed via
`git status` that no other file conflicted). Pushed: `54a967e..c07986a main -> main`.

## Restored 9 collaborator-authored files removed by the repo cleanup

The 2026-08-11/12 repo cleanup (see above) removed 9 files originally authored by Jingyuan Hu on
2026-06-08 (`flat_manifold_potential_fields.py`, `flat_manifold_vector_fields.py`,
`manifold_velocity_flows.py`, `generate_flat_manifold_potential_fields.py`,
`generate_flat_manifold_vector_fields.py`, `generate_manifold_velocity_flows.py`,
`serve_website.py`, `website/index.html`, `data/.gitignore`) as orphaned dead code once their last
importer (`run_simulation_benchmark_v2.py`) was retired. The reasoning was sound (nothing in the
frozen P0-P5 protocol references them), but the user flagged after the push that this removed a
collaborator's own contribution from the repo's current tree, not just this session's scratch
code. Restored all 9 verbatim from their last committed state (`dcfb890`, byte-identical,
confirmed via `cmp`) at the user's request, with no deprecation annotation -- kept exactly as they
were before the cleanup touched them.
