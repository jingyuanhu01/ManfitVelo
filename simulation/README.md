# ManfitVelo simulation benchmark

## Research question

This benchmark asks whether a method can recover the clean manifold state
`(X, V)` from ambient noisy observations `(X_noisy, V_noisy)`. It therefore
evaluates geometry and velocity in one pipeline comparison; it is not only a
velocity-smoothing benchmark. The seven main scenarios use only their frozen
default noise settings. Y-branch is intentionally retained as a branching,
non-manifold stress test.

## Algorithms

| Method | Input | Updates X | Updates V | Neighborhood/tangent rule | Hyperparameters / selection | Implementation and limitation |
|---|---|---:|---:|---|---|---|
| Ambient noisy input | X_noisy, V_noisy | No | No | None | None | Normalization reference for every relative metric. |
| Cosine kernel | X_noisy, V_noisy | No | Yes | Frozen noisy-position kNN; cosine direction, official-style density correction, observed noisy speed restoration | Previously frozen k; no final-seed selection | Local directional smoother; cannot repair geometry. |
| GraphVelo | Continuous X_noisy, V_noisy | No | Yes | Truth-free unit standardization, then official analytical-manifold 15-NN graph (including query point), cosine kernel, density correction, tangent-space projection, graph reconstruction | Fixed `a=1, b=0, r=1, loss_func=linear`; not tuned | Primary row standardizes by median noisy 15-NN distance and median noisy velocity norm, then maps velocity back. Raw-scale official output is a sensitivity diagnostic. Vendored from GraphVelo 0.1.11 / commit `0d2bb4e69b3632fe075963753efa913c51930d71`. No log, PCA, count preprocessing, labels, or clean graph. |
| Global PCA | X_noisy, V_noisy, known d | Yes | Yes | One centered global rank-d projector: `X_hat=mean+(X-mean)P_d`, `V_hat=V P_d` | No tuning; rank fixed to known intrinsic dimension d | Global PCA (rank fixed to the known intrinsic dimension d). A linear low-rank diagnostic, not a general nonlinear manifold estimator. |
| Local PCA | X_noisy, V_noisy, known d | Yes | Yes | Local affine PCA reconstructs X; neighborhoods and rank-d tangents are rebuilt at X_hat; V_noisy is projected there | Frozen scenario k | Full position–velocity pipeline. |
| Position-only MANFIT | X_noisy and V_noisy only for the downstream velocity step | Yes | Yes | Geometry fit excludes velocity feedback; frozen downstream rank-d local tangent rule reconstructs V | Frozen k, step size, iterations | Separates geometry-only fitting from the final velocity reconstruction. |
| ManfitVelo | X_noisy, V_noisy | Yes | Yes | Velocity-aware local neighborhoods and final fitted tangent projectors | Frozen on tuning seeds | Joint local manifold/vector-field estimator; can share kNN failure modes near branches or close sheets. |

The GraphVelo adapter vendors only the official numerical functions required by
the analytical simulation notebook. The fixed-ridge objective is not invariant
to independent changes of position and velocity units. The primary adapter
therefore uses `X*=(X_noisy-mean(X_noisy))/s_X` and `V*=V_noisy/s_V`, with `s_X`
the median positive official-15-NN displacement and `s_V` the median noisy
velocity norm, then returns `s_V V_hat*`. This rule is global, truth-free, and
not performance-selected. `simulation/test_graphvelo_baselines.py` checks both
the raw wrapper against the direct vendored notebook path and scale equivariance
of the standardized wrapper. Oracle nonnegative global rescaling uses clean
velocity only after fitting and appears solely in `graphvelo_scale_audit.csv`.

## Main scenarios

| Scenario | d | D | n | Position / velocity noise | Geometry and vector field | Primary stress | Smooth-manifold assumption |
|---|---:|---:|---:|---|---|---|---|
| Circle | 1 | 3 | 360 | 0.05 / 0.10 | Closed circle, unit tangential flow | Curvature and global-PCA bias | Yes |
| S-curve | 1 | 3 | 360 | 0.05 / 0.10 | Curved open S, varying tangent | Curvature | Yes |
| Curved hairpin | 1 | 3 | 480 | 0.025 / 0.10 | Reach-audited separation 0.22, bend radius 0.11, continuous flow | Nearby arms and branch-aware projection | Yes, with finite-sample close-sheet stress |
| Flat rotation annulus | 2 | 3 | 420 | 0.05 / 0.10 | Planar annulus, rotational field | Global linear sanity check | Yes |
| Half-sphere tangent | 2 | 3 | 480 | 0.04 / 0.10 | Upper sphere, projected tangent field | Curvature and low-speed regions | Yes |
| Y-branch | 1 | 3 | 480 | 0.02 / 0.10 | Three branches, outward flow | Branch point | **No; explicit stress test** |
| Near-intersection | 1 | 3 | 480 | 0.02 / 0.10 | Two labeled nonintersecting curves with opposing flow | Cross-sheet neighbor contamination | Locally yes, but reach is small |

## Tuning and seeds

The frozen tuning seeds are 42000–42002 and final evaluation seeds are
43000–43014. Local PCA, Position-only MANFIT, and ManfitVelo reuse their prior
frozen settings and objectives. The revised hairpin geometry was selected only
from reach and noisy-neighborhood diagnostics; method results were not used.
GraphVelo has no tuning stage, and Global PCA has no fitted hyperparameter.
Final seeds never enter geometry selection, model selection, or parameter
selection. The representative figures use pilot seed 41000.

## Main metrics

All six errors are lower-is-better and divided by the matching Ambient noisy
input error, making its row exactly 1.

1. `clean_point_rmse_rel` compares reconstructed positions with the original
   clean cell positions.
2. `distance_to_manifold_rel` measures distance to the true support.
3. `velocity_rmse_id_rel` compares `V_hat_i` with the clean velocity of the
   original generating cell.
4. `velocity_angle_mae_id_rel` compares their directions.
5. `velocity_rmse_loc_rel` projects `X_hat_i` to the true manifold and compares
   `V_hat_i` with the field at that location. Hairpin, near-intersection, and
   Y-branch use branch-aware projection. This metric does not penalize sliding
   along a manifold and must be read with clean-point RMSE.
6. `joint_euler_state_rmse_rel`, displayed as **Short-step Euler forecast RMSE
   rel**, compares `X_hat + tau V_hat` with `X_clean + tau V_true`, where
   `tau = 0.5 × median noisy kNN distance / median noisy velocity norm`. It is a
   joint short-step state metric, not a standalone measure of velocity recovery.

## Sphere scalability module

The independent module embeds a uniformly sampled `S²` into `R^D` for
`D={3,5,10,20,50}` with a deterministic orthonormal `D×3` matrix; it never uses
zero padding. The intrinsic dimension stays 2 and `n=480`. Clean rotation
velocities are tangent and normalized to a comparable median scale.

The formal experiment holds each coordinate's noise variance constant across
ambient dimensions: position and velocity noise have standard deviations
`tau_x/sqrt(3)` and `tau_v/sqrt(3)`, respectively, so total ambient noise grows
as `sqrt(D)`. Angle errors use the method-independent clean-speed threshold 0.20
and report the retained fraction. Runtime covers only each method call after
one untimed warm-up; metric evaluation, plotting, and I/O are excluded.
Per-method peak RSS is left missing because Python/process memory tools cannot
reliably isolate native BLAS and SciPy allocations.

Besides state errors, the module reports clean-vs-reconstructed kNN recall,
tangent-projector error, local covariance eigengap, and median kNN radius. The
formal Global PCA remains rank 2 even though a sphere has a three-dimensional
linear span.

## Reproduction

The two unique formal entry points are:

```bash
python simulation/run_manfitvelo_benchmark.py
python simulation/run_manfitvelo_benchmark.py --report-only

python simulation/run_sphere_scalability.py
python simulation/run_sphere_scalability.py --report-only
```

Install the versions recorded in each output directory's
`environment_provenance.json`; core requirements are Python, NumPy, SciPy,
pandas, scikit-learn, and Matplotlib. GraphVelo is not imported as a count-data
package: its pinned BSD-licensed analytical numerical path is vendored in
`scripts/graphvelo_official_adapter.py`.

Main outputs are under `results/manfitvelo_benchmark/`; scalability outputs are
under `results/sphere_scalability/`. Both HTML files embed all figures as data
URIs and open without external assets. Main GraphVelo scale results are in
`graphvelo_scale_audit.csv` (seed level) and
`graphvelo_scale_audit_summary.csv` (scenario medians). The oracle columns in
those files are diagnostics and are excluded from `final_seed_metrics.csv` and
the primary table. Run tests with:

```bash
python -m pytest -q simulation
```

On the recorded local macOS/Python environment, the main run takes under a
minute and the full scalability run takes roughly two minutes; actual time is
hardware- and BLAS-dependent. Exact platform, dependency, thread, and GraphVelo
provenance are stored with the results.
