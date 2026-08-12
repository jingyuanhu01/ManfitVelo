# Metric Definitions

Reference for every metric computed by `run_manfitvelo_benchmark.py` and `run_sphere_scalability.py`.
Companion to `parameter_rules.md` (how methods are configured) and `scenario_config.yaml` (what
geometry each metric is evaluated against). Matches Weekly Plan v1.1 §5–§9.

All metrics compare a method's output `(x̂, v̂)` against either the original clean generating point
`(x*, v(x*))` ("identity" anchoring) or the point on the true manifold nearest the method's own
denoised location `(x_proj, v(x_proj))` ("location" anchoring, `x_proj = Π_M(x̂)`). Every metric is
reported as **both** an absolute value and a value **relative to the Ambient Noisy Input baseline**
(`metric(x̂,v̂) / metric(x_noisy, v_noisy)`); relative values below 1 mean "better than doing nothing."

## A. Geometry recovery

| Metric | Column | Formula | Anchoring | Reads as |
|---|---|---|---|---|
| G1 — Distance to true manifold | `distance_to_manifold` | `d(x̂, 𝓜)` | — | Did the denoised point actually land back on the manifold? Primary geometry metric. |
| G2 — Distance to original clean point | `clean_point_rmse` | `‖x̂ − x*‖` | identity | Did we recover the *specific* generating point, not just *some* point on the manifold? Penalizes tangential sliding; interpret alongside G1. |

**Per-method caveat**: for M1 (GraphVelo) and M2 (Cosine Kernel), `x̂ ≡ x_noisy` by construction (they
never touch position) — their G1/G2 are therefore identical to the M0 baseline (relative value = 1
exactly) and are not meaningful method comparisons; the report does not rank them on G1/G2.

## B. Velocity recovery

Two different questions, kept separate:

### B1. Recovery relative to the original generating cell (identity-anchored)

| Metric | Column | Formula | Reads as |
|---|---|---|---|
| V1 — Velocity MAE (identity) | `velocity_rmse_id` | `‖v̂ − v(x*)‖` | Did we recover *this cell's* velocity? |
| V2 — Velocity angle (identity) | `velocity_angle_mae_id` | `∠(v̂, v(x*))` | Same question, direction-only. |

Secondary diagnostic — a method can legitimately reconstruct the *field* correctly at its own
(possibly shifted) output location while still scoring poorly here.

### B2. Recovery relative to the method's own denoised location (location-anchored)

| Metric | Column | Formula | Reads as |
|---|---|---|---|
| V3 — Projection-aware velocity MAE | `velocity_rmse_loc` | `‖v̂ − v(x_proj)‖`, `x_proj = Π_M(x̂)` | Is the velocity *consistent with where the method says the cell is*? |
| V4 — Projection-aware velocity angle | — (folded into `velocity_rmse_loc` reporting; see `joint_euler_state_rmse` for the paired joint metric) | `∠(v̂, v(x_proj))` | Same, direction-only. |

V3/V4 are the **primary velocity metrics** — they answer the actually-relevant question ("is the
joint manifold+field recovery self-consistent") rather than forcing every method to hit one fixed
target point.

## C. Joint dynamical metric

| Metric | Column | Formula | Reads as |
|---|---|---|---|
| One-Step Flow Error | `joint_euler_state_rmse` | `‖(x̂ + τv̂) − (x* + τv(x*))‖` | If we forecast one small Euler step from the denoised state, how far off is the predicted next state from the true one? |

`τ` (`tau`) is derived once per (scenario, seed) from the **observed noisy data only**
(`benchmark_core.observed_tau`: `0.5 × median-noisy-kNN-edge-distance / median-noisy-speed`) — never
tuned per method, never uses ground truth. Same `τ` is used for every method on that draw so the
comparison is fair; it is not selected to favor any method's timescale.

## D. Projection ambiguity handling

`x_proj = Π_M(x̂)` (nearest point on the manifold) is not always well-defined by ordinary nearest-Euclidean-point projection:

- **Curved Hairpin / Near Intersection**: nearest-Euclidean projection can jump to the *other* arm/branch
  even when the method's estimate is reasonable. Evaluation uses an **oracle branch-aware projection**
  (`benchmark_core.project_location_truth(..., branch_aware=True)`) that restricts the nearest-point
  search to the same labeled branch as the point's origin. The ordinary (branch-unaware) projection
  distance is also recorded as `unrestricted_branch_switch_fraction` / `distance_to_manifold` — a
  diagnostic for how often the two disagree, not part of the primary ranking.
- **Y-branch**: branch point itself has no unique tangent (non-smooth). Points within radius 0.05 of
  the branch point are excluded (`location_valid=False`) from velocity-angle-type metrics; geometry
  metrics (G1/G2) still cover them.
- **Flat Rotation Annulus / Half-sphere-tangent**: `x_proj` has a closed-form expression (radial clip
  to `[0.35,1]`; unit-sphere renormalization respectively) instead of dense-sample nearest-neighbor
  search — exact rather than approximate, same semantics.
- All other scenarios (Circle, S-curve, Swiss Roll, Saddle Surface): unambiguous single-sheet
  manifolds, plain dense-sample nearest-Euclidean-point projection.

## E. Mechanism diagnostics (not part of the primary ranking)

Recorded in `graphvelo_mechanism_diagnostics.csv` for GraphVelo and ManfitVelo only (the two methods
where a tangent/normal decomposition is most informative): `tangential_component_rmse` /
`normal_component_rmse` (decompose velocity error into the true tangent-plane and true-normal
directions — see `run_manfitvelo_benchmark.true_projector`), `graph_cross_branch_edge_fraction`
(kNN-graph contamination on the branch/stress scenarios), `velocity_speed_rmse` /
`velocity_direction_angle_error` (magnitude vs. direction error, separated). GraphVelo's raw-scale
sensitivity variant and oracle-rescaled velocity (`graphvelo_scale_audit.csv`) are provenance-only —
never enter the primary ranking (`oracle_enters_primary_ranking: false`, checked by
`validate()` at report-build time).

`nan_inf_count` (per scenario/seed/method) is a lightweight numerical-failure audit — count of
non-finite values in a method's output, sanitized to 0 before metrics are computed so one failed
fit can't silently poison a median. Not a full failure-rate protocol (deferred; see
`simulation_protocol.md` §Limitations).

## F. Headline vs. supplementary

Per Weekly Plan §9, the report's headline table shows only `distance_to_manifold_rel`,
`clean_point_rmse_rel`, the projection-aware velocity angle, and `joint_euler_state_rmse_rel`. Full
per-metric numbers live in `summary_metrics.csv` / `final_seed_metrics.csv` and the report's
"Show every frozen per-scenario algorithm parameter" detail section.
