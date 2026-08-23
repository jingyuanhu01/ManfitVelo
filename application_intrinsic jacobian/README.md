# Intrinsic-Jacobian downstream analysis for ManfitVelo

This directory extends ManfitVelo from manifold and velocity estimation to
local differential analysis of the fitted velocity field. It estimates an
intrinsic Jacobian at every sufficiently supported cell, optionally smooths
the matrix field using the learned tangent connection, and derives downstream
quantities describing local expansion, contraction, shear, rotation, manifold
bending, and candidate fixed points.

## Inputs from ManfitVelo

The notebooks first run
`scripts.velocity_manifold_fitter.VelocityManifoldFitter`. The downstream
modules consume the resulting arrays:

| Input | Meaning |
|---|---|
| `X_hat` | Denoised cell states in the ambient expression/PCA space. |
| `V_hat` | Fitted RNA velocities in the same coordinates. |
| `U_hat` | Local orthonormal tangent bases, one $D\times d$ frame per cell. |
| `P_hat` | Local tangent projectors $U_iU_i^T$. |
| `neighbors` | ManfitVelo's local geometry graph. |
| `weights` | Local kernel weights on that graph. |

The FlowMap or PCA coordinates shown in the notebooks are used only for
visualization. Jacobian estimation is performed using ManfitVelo's fitted
state, velocity, tangent, projector, neighbor, and weight outputs.

## 1. Intrinsic and ambient Jacobian estimation

Implemented in [`intrinsic_jacobian.py`](intrinsic_jacobian.py).

### Intrinsic regression

For cell $i$, neighbor displacements are expressed in the local tangent frame:

$$z_{ij}=U_i^T(\hat X_j-\hat X_i).$$

Because tangent coordinates at different cells use different bases, the
neighbor frame is aligned to the frame at $i$ by an orthogonal Procrustes
transport $O_{ij}$ computed from $U_i^TU_j$. With
$a_j=U_j^T\hat V_j$, the local Jacobian is estimated by weighted ridge
regression:

$$
(b_i,J_i)=\arg\min_{b,J}\sum_j w_{ij}
\left\|O_{ij}a_j-b-Jz_{ij}\right\|^2+\rho_i\|J\|_F^2.
$$

The intercept $b_i$ is a locally denoised tangent velocity. The fitted
$d\times d$ matrix $J_i$ approximates $\nabla^M v(x_i)$ and transforms
consistently when the arbitrary local tangent basis is rotated.

### Ambient regression and Gauss split

A second weighted ridge regression is performed directly in the shared
ambient coordinates:

$$
B_i=\arg\min_B\sum_j w_{ij}
\left\|(\hat V_j-\hat V_i)-B(\hat X_j-\hat X_i)\right\|^2
+\rho_i^{\mathrm{amb}}\|B\|_F^2.
$$

Using ManfitVelo's projector $P_i$, this derivative is separated into

$$J_i^{\mathrm{amb}}=P_iB_iP_i,\qquad
N_i=(I-P_i)B_iP_i.$$

$J_i^{\mathrm{amb}}$ is the tangential component of the ambient derivative,
whereas $N_i$ captures the normal component associated with bending of the
learned manifold.

### Reliability diagnostics

Each local estimate reports effective neighbor count, design conditioning,
ridge scale, regression residual, tangent-transport consistency, and a local
boundary/branch diagnostic. Cells with insufficient or inconsistent local
information are marked unreliable or abstained from rather than interpreted as
equally trustworthy estimates.

## 2. Connection-aware smoothing

Implemented in [`connection_smoothing.py`](connection_smoothing.py).

Neighboring Jacobians cannot be averaged entry by entry because they are
represented in different tangent frames. The smoother first transports
$J_j$ into the frame at $i$ and solves

$$
\min_{\{J_i\}}
\sum_i\alpha_i\|J_i-\hat J_i\|_F^2+
\lambda_J\sum_{(i,j)}w_{ij}
\|J_i-O_{ij}J_jO_{ij}^T\|_F^2.
$$

The fidelity weight

$$
\alpha_i\propto\frac{n_{\mathrm{eff},i}}
{\max(\kappa_i,1)(1+r_i)}
$$

trusts local fits more when they have stronger effective support, better
conditioning, and smaller residuals. Abstained points receive zero direct
fidelity and can only be interpolated from connected neighbors. The quadratic
system is solved with a matrix-free conjugate-gradient method; no dense global
$(nd^2)\times(nd^2)$ matrix is assembled.

## 3. Quantities derived from the Jacobian

Implemented in [`downstream_geometry.py`](downstream_geometry.py).

### Local flow geometry

The smoothed intrinsic Jacobian is decomposed into symmetric and antisymmetric
parts,

$$S=\tfrac12(J+J^T),\qquad A=\tfrac12(J-J^T).$$

- `divergence = tr(J)` measures local expansion or contraction of nearby
  state-space trajectories.
- `shear` is the Frobenius norm of the trace-free part of $S$ and measures
  direction-dependent stretching.
- `vorticity_mag = ||A||_F` measures the magnitude of the locally cyclic
  component.
- Eigenvalues of $S$ summarize instantaneous strain as stable, unstable, or
  saddle-like.

These are properties of the inferred expression-state flow. They are not
direct measurements of proliferation, cell death, physical rotation, or
causal gene regulation.

### Gauss acceleration decomposition

For the tangential fitted velocity $v_{\mathrm{tan}}=Pv$:

$$
a_{\mathrm{int}}=J^{\mathrm{amb}}v_{\mathrm{tan}},\qquad
a_{\mathrm{normal}}=Nv_{\mathrm{tan}},\qquad
a_{\mathrm{amb}}=a_{\mathrm{int}}+a_{\mathrm{normal}}.
$$

$a_{\mathrm{int}}$ describes change of the velocity within the fitted
manifold. $a_{\mathrm{normal}}$ describes ambient direction change associated
with bending of that manifold. Their relative contribution is summarized by

$$R_{\mathrm{bend}}=
\frac{\|a_{\mathrm{normal}}\|}{\|a_{\mathrm{int}}\|+\epsilon}.$$

### Schur cross-sections and divergence transitions

A real Schur decomposition can select the invariant subspace associated with
the most expanding, most contracting, or largest-magnitude eigenvalue block of
a nonsymmetric Jacobian. The module also identifies reliable graph edges on
which divergence changes sign, providing a graph-native transition detector
that does not require a predefined one-dimensional ordering.

### Candidate fixed points

Reliable local minima of $\|\hat V\|$ are used as candidate fixed points. A
single ridge-stabilized Newton correction is computed in tangent coordinates
and capped to a fraction of the local neighborhood scale; a capped step is
reported as a warning that the extrapolated zero is not locally trustworthy.

Classification uses eigenvalues of the full nonsymmetric intrinsic Jacobian:

- negative real parts: attractor;
- positive real parts: repeller;
- mixed real-part signs: saddle;
- nonzero imaginary parts: spiral rather than node-like motion.

These remain candidate local dynamical structures rather than experimentally
confirmed biological fixed points.

## Application notebooks

- [`fucci_intrinsic_jacobian.ipynb`](fucci_intrinsic_jacobian.ipynb) applies
  the workflow to independently FACS-labelled FUCCI cell-cycle data. It
  includes `d=1`/`d=2` geometry, Gauss decomposition, candidate fixed points,
  divergence along FUCCI ordering, and phase differential expression.
- [`cell_cycle_intrinsic_jacobian.ipynb`](cell_cycle_intrinsic_jacobian.ipynb)
  applies the same workflow to a 33-gene cell-cycle panel. Its independently
  measured protein-derived continuous cell-cycle state is used for descriptive
  external ordering without imposing categorical phase boundaries.

The categorical FUCCI DEG analysis uses independent FACS labels with
Wilcoxon/BH-FDR testing. Gene–Jacobian correlations in the FUCCI notebook and
the geometry-stratified gene ranking in the 33-gene notebook are descriptive,
not causal analyses.

## Data and preprocessing

Analysis-ready arrays are included under [`datasets/`](datasets/README.md).
The FUCCI raw loom and FACS exports are not duplicated here;
[`prepare_fucci_data.py`](prepare_fucci_data.py) documents and implements the
conversion from the upstream Mahdessian et al. data to the included state,
velocity, expression, and independently measured phase arrays.

## Run the notebooks

From the ManfitVelo repository root:

```bash
python -m pip install -r "application_intrinsic jacobian/requirements.txt"
cd "application_intrinsic jacobian"

jupyter nbconvert --to notebook --execute --inplace \
  fucci_intrinsic_jacobian.ipynb \
  --ExecutePreprocessor.timeout=600

jupyter nbconvert --to notebook --execute --inplace \
  cell_cycle_intrinsic_jacobian.ipynb \
  --ExecutePreprocessor.timeout=600
```

The notebooks should be launched from this directory so they can resolve both
the three downstream modules and the parent ManfitVelo implementation.
