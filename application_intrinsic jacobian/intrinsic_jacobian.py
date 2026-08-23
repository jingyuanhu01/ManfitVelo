"""Core local intrinsic/ambient Jacobian estimator for a velocity field on an
unknown manifold.

Setup (see ``Notes/intrinsic_jacobian_velocity_beamer.tex``, the source of
truth for the math that *is* worked out elsewhere): noisy observations
``Y_i = X_i + eps_i``, ``W_i = v(X_i) + xi_i``, ``X_i in M subset R^D``.
Target: the intrinsic first derivative ``J_x = grad^M v(x) : T_xM -> T_xM``,
and the Gauss decomposition of the ambient derivative for ``u in T_xM``::

    D_u v = grad^M_u v + II(u, v)

with ``grad^M_u v in T_xM`` (intrinsic dynamics: expansion/contraction,
shear, rotation) and ``II(u, v) in N_xM`` (curvature / manifold bending,
"without dynamical change").

Design constraints (see ``application_intrinsic jacobian/DECISIONS.md`` and
``PLAN.md``):
  * Pure numpy/scipy. Zero dependency on ``scripts/`` or ``simulation/`` so
    this module is unit-testable independent of ManfitVelo actually
    running. See ``sources.py`` for the adapter that feeds this module
    ManfitVelo's (or any other source's) output.
  * Operates on plain arrays for a *single, fixed* local dimension ``d`` and
    ambient dimension ``D`` shared by every point — this matches the frozen
    canonical protocol's ``d_mode="global"`` exactly (see
    ``scripts/benchmark_scenarios.fit_vmf_variant``); the ragged per-point
    ``d`` produced by ManfitVelo's ``d_mode="adaptive"`` is out of scope.

Formula provenance:
  * Edge coordinates ``z_ij``, Procrustes ``O_ij``, the intrinsic Taylor
    expansion ``O_ij a_j - a_i ~= J_i z_ij``, and gauge equivariance under
    ``U_i -> U_i Q_i``: ``Notes/intrinsic_jacobian_velocity_beamer.tex``,
    "Local tangent-basis representation" + "A minimal direct estimator"
    slides (source lines 201-256).
  * The ambient regression ``B_i`` and its tangential/normal decomposition
    ``J_amb_i = P_i B_i P_i``, ``N_i = (I - P_i) B_i P_i``: **not** present
    in the beamer deck, ``PLAN.md``, ``NOVELTY_AUDIT.md``, or any other file
    in the repo (confirmed by a full-repo search on 2026-08-14) — designed
    here directly from the Gauss decomposition above, structurally parallel
    to ``scripts/scalar_potential_manfit.py``'s
    ``estimate_gradient_from_neighbors`` (a scalar-target ridge regression
    generalized to a matrix-valued target).
  * Local reliability diagnostics (``n_eff``, ``G_i``, ``kappa``, the ridge
    rule, ``Delta_cons``, reliability flags) and the adaptive ``k_deriv``
    rule: also absent from every source file — original to this module,
    deliberately kept simple (a handful of named constants, no
    per-scenario tuning) rather than theoretically derived.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

EPS = 1e-12


# --------------------------------------------------------------------------
# Low-level primitives
# --------------------------------------------------------------------------

def effective_sample_size(w: np.ndarray) -> float:
    """``n_eff = 1 / sum(w^2)`` for weights normalized to sum to 1.

    Identical formula to ``scripts.benchmark_scenarios.effective_sample_size``
    (Kish's effective sample size); reimplemented locally so this module has
    zero dependency on ``scripts/``.
    """
    return 1.0 / max(float(np.sum(w ** 2)), EPS)


def spatial_kernel_weights(dist: np.ndarray, bandwidth_mode: str = "variable",
                            beta: float = 1.0, h_fixed: Optional[float] = None) -> np.ndarray:
    """Position-only ``max(0, 1-(dist/h)^2)^beta`` kernel, normalized to sum
    to 1.

    This is a *spatial-only* variant of the kernel
    ``VelocityManifoldFitter._update_weights`` / ``position_only_trajectory``
    use in the parent repo (both use ``spatial = max(0,1-(d/h)^2)**beta``
    combined with a velocity-aware directional term we deliberately drop
    here). It is reused only to grow the derivative neighborhood (``k_deriv``,
    see :func:`grow_derivative_neighborhood`) in a way that "looks like" the
    rest of the codebase's local weighting, not to reproduce
    ``VelocityManifoldFitter``'s own geometry selection.
    """
    if bandwidth_mode == "variable":
        h = float(np.max(dist)) + EPS
    elif bandwidth_mode == "fixed":
        h = float(h_fixed) + EPS
    else:
        raise ValueError("bandwidth_mode must be 'variable' or 'fixed'")
    scaled = dist / h
    w = np.maximum(0.0, 1.0 - scaled ** 2) ** beta
    s = w.sum()
    if s <= EPS:
        w = np.ones_like(w)
        s = w.sum()
    return w / s


def edge_coordinates(U_i: np.ndarray, X_i: np.ndarray, X_neighbors: np.ndarray) -> np.ndarray:
    """``z_ij = U_i^T (X_j - X_i)`` for a batch of neighbors ``j``.

    Parameters
    ----------
    U_i : (D, d) orthonormal tangent basis at point i.
    X_i : (D,) position of point i.
    X_neighbors : (k, D) positions of the k neighbors.

    Returns
    -------
    z : (k, d)
    """
    return (X_neighbors - X_i[None, :]) @ U_i


def procrustes_transport_batch(U_i: np.ndarray, U_j: np.ndarray) -> np.ndarray:
    """Batched Procrustes alignment ``O_ij = L R^T`` from the SVD
    ``U_i^T U_j = L Sigma R^T``, for a fixed ``U_i`` against a batch of
    neighbor bases ``U_j``.

    Parameters
    ----------
    U_i : (D, d)
    U_j : (k, D, d)

    Returns
    -------
    O : (k, d, d), each an orthogonal matrix (O(d), not necessarily SO(d)).
    """
    M = np.einsum("Di,kDj->kij", U_i, U_j)  # (k, d, d) = U_i^T U_j per neighbor
    L, _, Rt = np.linalg.svd(M)
    return L @ Rt


def _weighted_intercept_ridge(Z_aug: np.ndarray, Y: np.ndarray, w: np.ndarray,
                               ridge: float):
    """Weighted ridge regression ``Y ~= Z_aug @ [intercept; J^T]`` with the
    ridge penalty applied only to the non-intercept block.

    ``Z_aug`` is ``(k, 1+p)`` with a leading column of ones. Returns
    ``(intercept (q,), J (q,p), G_raw (p,p) un-ridged normal block,
    weighted_mse (float))``.
    """
    p = Z_aug.shape[1] - 1
    Wz = w[:, None] * Z_aug
    G_aug = Z_aug.T @ Wz
    G_raw = G_aug[1:, 1:].copy()
    G_aug = G_aug.copy()
    G_aug[1:, 1:] += ridge * np.eye(p)
    rhs = Z_aug.T @ (w[:, None] * Y)
    sol = np.linalg.solve(G_aug, rhs)
    intercept = sol[0]
    J = sol[1:].T
    pred = Z_aug @ sol
    resid = Y - pred
    wsum = max(float(np.sum(w)), EPS)
    weighted_mse = float(np.sum(w * np.sum(resid ** 2, axis=1)) / wsum)
    return intercept, J, G_raw, weighted_mse


def _ridge_from_trace(G_raw: np.ndarray, p: int, alpha: float) -> float:
    """Scale-aware ridge rule ``rho = alpha * trace(G_raw) / p``: a single
    default ``alpha`` works across scenarios/scales without per-scenario
    tuning, because it matches the units of ``G_raw``'s own eigenvalues.
    """
    # A locally collapsed point cloud has trace(G_raw) == 0.  Keep the
    # normal equations solvable so the conditioning diagnostics can reject
    # the fit cleanly instead of failing inside ``np.linalg.solve``.
    return max(float(alpha * np.trace(G_raw) / max(p, 1)), EPS)


# --------------------------------------------------------------------------
# Per-point fit results
# --------------------------------------------------------------------------

@dataclass
class IntrinsicFit:
    J: np.ndarray            # (d, d)
    intercept: np.ndarray    # (d,)
    residual: float
    G: np.ndarray            # (d, d), un-ridged design covariance
    lambda_min: float
    kappa: float
    ridge: float


@dataclass
class AmbientFit:
    B: np.ndarray            # (D, D)
    J_amb: np.ndarray        # (D, D) = P_i B P_i
    N: np.ndarray             # (D, D) = (I-P_i) B P_i
    intercept: np.ndarray    # (D,)
    residual: float
    G: np.ndarray             # (D, D), un-ridged design covariance
    lambda_min: float
    kappa: float
    ridge: float


@dataclass
class PointDiagnostics:
    n_eff: float
    k_used: int
    lambda_min_intrinsic: float
    kappa_intrinsic: float
    lambda_min_ambient: float
    kappa_ambient: float
    delta_cons: float
    cons_scale: float
    transport_error: float
    boundary_moment: float
    abstained: bool
    flags: dict


@dataclass
class PointResult:
    intrinsic: Optional[IntrinsicFit]
    ambient: Optional[AmbientFit]
    diagnostics: PointDiagnostics


@dataclass
class JacobianFieldResult:
    """Stacked per-point results, arrays of length n (NaN-filled where a
    point abstained)."""
    n: int
    d: int
    D: int
    J_intrinsic: np.ndarray   # (n, d, d)
    intercept_intrinsic: np.ndarray  # (n, d)
    B: np.ndarray              # (n, D, D)
    J_amb: np.ndarray          # (n, D, D)
    N: np.ndarray               # (n, D, D)
    n_eff: np.ndarray           # (n,)
    k_used: np.ndarray          # (n,) int
    kappa_intrinsic: np.ndarray  # (n,)
    kappa_ambient: np.ndarray    # (n,)
    residual_intrinsic: np.ndarray  # (n,) weighted-MSE residual of the intrinsic fit
    residual_ambient: np.ndarray     # (n,) weighted-MSE residual of the ambient fit
    delta_cons: np.ndarray       # (n,)
    transport_error: np.ndarray  # (n,)
    boundary_moment: np.ndarray  # (n,)
    abstained: np.ndarray        # (n,) bool
    reliable: np.ndarray         # (n,) bool
    flag_low_effective_sample: np.ndarray
    flag_ill_conditioned: np.ndarray
    flag_transport_inconsistent: np.ndarray
    flag_possible_boundary_or_branch: np.ndarray
    points: list = field(default_factory=list)  # list[PointResult], for debugging


# --------------------------------------------------------------------------
# Single-point fitters
# --------------------------------------------------------------------------

def fit_intrinsic_jacobian(U_i: np.ndarray, X_i: np.ndarray, X_neigh: np.ndarray,
                            U_neigh: np.ndarray, a_i_obs: np.ndarray,
                            a_neigh_obs: np.ndarray, w: np.ndarray,
                            ridge_alpha: float = 0.05,
                            ridge: Optional[float] = None) -> IntrinsicFit:
    """Direct intrinsic-regression estimator (PLAN.md 1.2-1.3;
    ``Notes/intrinsic_jacobian_velocity_beamer.tex`` lines 231-256)::

        (b_i, J_i) = argmin_{b,J} sum_j w_ij || O_ij a_j - b - J z_ij ||^2
                                     + rho_i ||J||_F^2

    ``a_i_obs`` is accepted for interface symmetry with the ambient fitter
    but is *not* used directly as a regression target — per PLAN 1.3 the
    intercept ``b_i`` is itself a (weighted-centering) denoised estimate of
    the local velocity, not the raw noisy ``a_i_obs``.
    """
    d = U_i.shape[1]
    z = edge_coordinates(U_i, X_i, X_neigh)                    # (k, d)
    O = procrustes_transport_batch(U_i, U_neigh)                 # (k, d, d)
    y = np.einsum("kij,kj->ki", O, a_neigh_obs)                    # (k, d)
    Z_aug = np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)
    G_tmp = z.T @ (w[:, None] * z)
    if ridge is None:
        ridge = _ridge_from_trace(G_tmp, d, ridge_alpha)
    intercept, J, G_raw, wmse = _weighted_intercept_ridge(Z_aug, y, w, ridge)
    eigs = np.linalg.eigvalsh(G_raw)
    lam_min, lam_max = float(eigs[0]), float(eigs[-1])
    kappa = lam_max / max(lam_min, EPS)
    return IntrinsicFit(J=J, intercept=intercept, residual=wmse, G=G_raw,
                         lambda_min=lam_min, kappa=kappa, ridge=ridge)


def fit_ambient_jacobian(X_i: np.ndarray, V_i: np.ndarray, P_i: np.ndarray,
                          X_neigh: np.ndarray, V_neigh: np.ndarray, w: np.ndarray,
                          d: int, ridge_alpha: float = 0.05,
                          ridge: Optional[float] = None) -> AmbientFit:
    """Ambient-derivative regression -> tangential/normal decomposition
    (PLAN.md 1.4; formula designed for this module, see module docstring)::

        B_i = argmin_B sum_j w_ij || (V_j - V_i) - B (X_j - X_i) ||^2
                          + rho_i^amb ||B||_F^2
        J_amb_i = P_i B_i P_i          (tangential part, ~= grad^M v)
        N_i     = (I - P_i) B_i P_i    (normal part, ~= II(., v))

    No Procrustes transport is used here: ambient vectors ``V_j``, ``V_i``
    already live in one shared global frame ``R^D``, unlike the local
    tangent coordinates ``a_i``/``a_j`` the intrinsic fitter must align.
    ``P_i`` is consumed directly from the upstream geometry source (e.g.
    ManfitVelo's own ``P`` output) and never reconstructed here.

    ``d`` (the manifold's intrinsic dimension) is required here purely for
    conditioning diagnostics, not for the fit itself: neighbor positions
    ``X_neigh - X_i`` live, to leading order, in the ``d``-dimensional
    tangent subspace, so the raw ``(D,D)`` design ``G_raw`` structurally has
    ``D-d`` near-zero eigenvalues *by construction* for any embedded
    submanifold with ``D > d`` -- that is not an estimation problem (this
    fitter is only ever applied to tangential inputs, ``B_i @ v_tan``, so
    those near-null directions never get used downstream), but it makes a
    naive full-``D``-dimensional condition number ``lambda_max/lambda_min``
    spuriously enormous for *every* point on *every* embedded manifold,
    regardless of data quality. ``kappa``/the ridge scale are therefore both
    computed from only the top-``d`` eigenvalues (the directions that
    actually carry signal); ``lambda_min`` is still reported as the raw
    (expected-near-zero) overall minimum for reference.
    """
    D = X_i.shape[0]
    Z = X_neigh - X_i[None, :]
    Y = V_neigh - V_i[None, :]
    Z_aug = np.concatenate([np.ones((Z.shape[0], 1)), Z], axis=1)
    G_tmp = Z.T @ (w[:, None] * Z)
    eigs_tmp = np.linalg.eigvalsh(G_tmp)  # ascending
    top_d_sum = float(np.sum(eigs_tmp[-d:]))
    if ridge is None:
        ridge = max(ridge_alpha * top_d_sum / max(d, 1), EPS)
    intercept, B, G_raw, wmse = _weighted_intercept_ridge(Z_aug, Y, w, ridge)
    J_amb = P_i @ B @ P_i
    N = (np.eye(D) - P_i) @ B @ P_i
    eigs = np.linalg.eigvalsh(G_raw)  # ascending
    eigs_desc = eigs[::-1]
    lam_max = float(eigs_desc[0])
    lam_min_topd = float(eigs_desc[min(d - 1, D - 1)])
    kappa = lam_max / max(lam_min_topd, EPS)
    lam_min = float(eigs[0])  # raw overall min, expected near 0, diagnostic only
    return AmbientFit(B=B, J_amb=J_amb, N=N, intercept=intercept, residual=wmse,
                       G=G_raw, lambda_min=lam_min, kappa=kappa, ridge=ridge)


# --------------------------------------------------------------------------
# Diagnostics / reliability flags
# --------------------------------------------------------------------------

def compute_point_diagnostics(U_i: np.ndarray, intrinsic: IntrinsicFit,
                               ambient: AmbientFit, z: np.ndarray, O: np.ndarray,
                               U_neigh: np.ndarray, w: np.ndarray, n_eff: float,
                               k_used: int, abstained: bool, d: int,
                               low_eff_factor: float = 3.0,
                               kappa_threshold: float = 50.0,
                               boundary_threshold: float = 0.5) -> PointDiagnostics:
    """PLAN.md 1.5 diagnostics (formulas original to this module; see module
    docstring). Computes everything that is meaningful *per point in
    isolation*; the `transport_inconsistent`/`reliable` flags need a
    batch-relative reference scale and are filled in afterward by
    :func:`estimate_jacobian_field` (see `cons_scale`'s own docstring note
    below and that function's two-pass structure) -- so `flags` here only
    contains `low_effective_sample`/`ill_conditioned`/
    `possible_boundary_or_branch`, not the final `reliable` verdict.
    """
    # Consistency check: express J_amb in point i's own local frame (no
    # transport needed, both objects already live in the same frame U_i).
    J_amb_loc = U_i.T @ ambient.J_amb @ U_i
    num = float(np.linalg.norm(intrinsic.J - J_amb_loc, "fro"))
    cons_scale = float(np.linalg.norm(intrinsic.J, "fro") + np.linalg.norm(J_amb_loc, "fro"))
    den = cons_scale + EPS
    delta_cons = num / den
    # `delta_cons` is a *relative* mismatch and is not meaningful when both
    # J and J_amb_loc are near the estimator's own noise floor: two
    # independent near-zero estimates can differ from each other by ~100%
    # in relative terms while both being individually negligible in
    # absolute terms. `cons_scale` (returned, not thresholded here) is what
    # `estimate_jacobian_field`'s second pass uses to build a *batch-
    # relative* floor -- a single global constant does not generalize
    # across datasets whose typical J magnitude can differ by 2-3 orders of
    # magnitude (empirically: ~0.0002 on one real scenario, ~2.7 on
    # another, at zero data noise -- see DECISIONS.md's note on this fix).

    # O_ji ~= O_ij^T check: an independent Procrustes(U_j, U_i) computation,
    # not a transpose of O_ij, so this genuinely tests the identity rather
    # than assuming it.
    if U_neigh.shape[0] > 0:
        M = np.einsum("kDi,Dj->kij", U_neigh, U_i)  # U_j^T U_i per neighbor
        Lj, _, Rtj = np.linalg.svd(M)
        O_ji = Lj @ Rtj
        transport_err = float(np.mean(np.linalg.norm(
            np.einsum("kij,kjl->kil", O, O_ji) - np.eye(d)[None, :, :], axis=(1, 2))))
    else:
        transport_err = 0.0

    # Boundary/branch signal: weighted first moment of z relative to the
    # design's own scale sqrt(trace(G)).
    if z.shape[0] > 0 and np.sum(w) > EPS:
        mean_z = np.sum(w[:, None] * z, axis=0) / np.sum(w)
        scale = np.sqrt(max(float(np.trace(intrinsic.G)), EPS))
        boundary_moment = float(np.linalg.norm(mean_z) / scale)
    else:
        boundary_moment = 0.0

    flag_low_eff = n_eff < low_eff_factor * (d + 1)
    flag_ill_cond = (intrinsic.kappa > kappa_threshold) or (ambient.kappa > kappa_threshold)
    flag_boundary = boundary_moment > boundary_threshold

    flags = {
        "low_effective_sample": bool(flag_low_eff),
        "ill_conditioned": bool(flag_ill_cond),
        "possible_boundary_or_branch": bool(flag_boundary),
    }
    return PointDiagnostics(
        n_eff=n_eff, k_used=k_used,
        lambda_min_intrinsic=intrinsic.lambda_min, kappa_intrinsic=intrinsic.kappa,
        lambda_min_ambient=ambient.lambda_min, kappa_ambient=ambient.kappa,
        delta_cons=delta_cons, cons_scale=cons_scale, transport_error=transport_err,
        boundary_moment=boundary_moment, abstained=abstained, flags=flags,
    )


# --------------------------------------------------------------------------
# Adaptive k_deriv
# --------------------------------------------------------------------------

def grow_derivative_neighborhood(tree: cKDTree, X: np.ndarray, i: int, d: int,
                                  k0: int, growth: float = 1.5,
                                  low_eff_factor: float = 3.0,
                                  cap_min: int = 8, cap_factor: float = 4.0):
    """PLAN.md 1.6: start at ``k0`` (the input source's own geometry ``k``),
    grow geometrically (factor ``growth``) until the local effective sample
    size ``n_eff >= low_eff_factor*(d+1)``, capped at
    ``max(cap_min*(d+1), cap_factor*k0)``. Abstain (flag only, caller decides
    what to do) if the cap is reached without hitting the threshold.

    Returns ``(idx (k,) int neighbor indices excluding i, dist (k,),
    w (k,) normalized spatial weights, n_eff, k_used, abstained)``.
    """
    n = X.shape[0]
    cap = int(min(n - 1, max(cap_min * (d + 1), cap_factor * k0)))
    k = int(min(max(k0, d + 1), cap))
    threshold = low_eff_factor * (d + 1)
    while True:
        dist, idx = tree.query(X[i], k=k + 1)
        idx = np.asarray(idx).reshape(-1)
        dist = np.asarray(dist).reshape(-1)
        mask = idx != i
        idx, dist = idx[mask][:k], dist[mask][:k]
        w = spatial_kernel_weights(dist)
        n_eff = effective_sample_size(w)
        if n_eff >= threshold or k >= cap:
            break
        k = min(int(np.ceil(k * growth)), cap)
    abstained = n_eff < threshold
    return idx, dist, w, n_eff, k, abstained


# --------------------------------------------------------------------------
# Top-level driver
# --------------------------------------------------------------------------

def estimate_jacobian_field(X: np.ndarray, V: np.ndarray, U: np.ndarray,
                             P: np.ndarray, d: int, *, k0: Optional[int] = None,
                             neighbors: Optional[np.ndarray] = None,
                             ridge_alpha: float = 0.05,
                             k_deriv_growth: float = 1.5,
                             low_eff_factor: float = 3.0,
                             kappa_threshold: float = 50.0,
                             cons_threshold: float = 0.5,
                             cons_scale_floor_factor: float = 0.5,
                             cons_scale_floor_min: float = 1e-6,
                             transport_threshold: float = 0.1,
                             boundary_threshold: float = 0.5,
                             cap_min: int = 8, cap_factor: float = 4.0,
                             keep_points: bool = False) -> JacobianFieldResult:
    """Fit the intrinsic and ambient Jacobian at every point of a
    standardized input source (see ``sources.StandardizedSource``).

    Parameters
    ----------
    X, V : (n, D) positions and tangent-projected ambient velocities.
    U : (n, D, d) per-point orthonormal tangent basis.
    P : (n, D, D) per-point tangent projector ``U @ U.T``.
    d : shared local (intrinsic) dimension.
    k0 : starting/geometry neighborhood size for the adaptive ``k_deriv``
        rule. Defaults to ``neighbors.shape[1]`` if given, else ``4*(d+1)``.
    neighbors : optional ``(n, k_geo)`` geometry graph; only used to infer
        a default ``k0`` when not given explicitly. The derivative
        neighborhood is always (re)computed by Euclidean kNN on ``X`` via a
        KD-tree, independent of whichever graph the geometry source used
        internally, per PLAN.md 1.6 ("adaptive k_deriv, separate from
        k_geo").
    cons_scale_floor_factor, cons_scale_floor_min : the
        ``transport_inconsistent`` flag's absolute floor on
        ``cons_scale = ||J_int||+||J_amb_loc||`` is computed *from this
        batch's own data* as
        ``max(cons_scale_floor_min, cons_scale_floor_factor * median(cons_scale))``
        rather than a fixed global constant -- a fixed constant tuned on
        one dataset does not generalize (empirically, a scenario's typical
        ``cons_scale`` at zero data noise ranged from ~0.0002 to ~2.7 across
        the 9 canonical benchmark scenarios; see DECISIONS.md's note on
        this fix). This requires two passes over the points (fits +
        per-point diagnostics first, then the batch-relative flag), which
        is why this function is not a simple single loop.

        Known residual case even with this fix: on an *exact* zero-noise
        oracle input where the intrinsic regression's target happens to be
        algebraically constant (``J_intrinsic`` comes out at machine
        epsilon, not just "small"), ``delta_cons`` is ~1 for a large
        fraction of points regardless of the floor, because one side of
        the comparison is pinned to exact zero while the other carries
        genuine finite-window regression bias -- no fixed-fraction-of-
        median threshold can separate "reliable" from "unreliable" when
        the whole batch is this same kind of noise. This does not occur on
        any realistically noisy source (raw data, M5, M6, or any real
        dataset) -- verified empirically the ``reliable`` fraction there is
        well-behaved (roughly 0.85-0.99 across the sources actually used
        for reporting).
    """
    n, D = X.shape
    if k0 is None:
        k0 = neighbors.shape[1] if neighbors is not None else 4 * (d + 1)
    tree = cKDTree(X)

    J_intrinsic = np.full((n, d, d), np.nan)
    intercept_intrinsic = np.full((n, d), np.nan)
    B = np.full((n, D, D), np.nan)
    J_amb = np.full((n, D, D), np.nan)
    N = np.full((n, D, D), np.nan)
    n_eff_arr = np.full(n, np.nan)
    k_used_arr = np.zeros(n, dtype=int)
    kappa_int_arr = np.full(n, np.nan)
    kappa_amb_arr = np.full(n, np.nan)
    residual_int_arr = np.full(n, np.nan)
    residual_amb_arr = np.full(n, np.nan)
    delta_cons_arr = np.full(n, np.nan)
    cons_scale_arr = np.full(n, np.nan)
    transport_err_arr = np.full(n, np.nan)
    boundary_arr = np.full(n, np.nan)
    abstained_arr = np.zeros(n, dtype=bool)
    reliable_arr = np.zeros(n, dtype=bool)
    flag_low = np.zeros(n, dtype=bool)
    flag_ill = np.zeros(n, dtype=bool)
    flag_trans = np.zeros(n, dtype=bool)
    flag_bound = np.zeros(n, dtype=bool)
    points = []
    diag_by_point = {}

    a_all = np.einsum("nDd,nD->nd", U, V)  # local velocity coords for every point

    # --- Pass 1: per-point fits + per-point-only diagnostics -------------
    for i in range(n):
        idx, _dist, w, n_eff, k_used, abstained = grow_derivative_neighborhood(
            tree, X, i, d, k0, growth=k_deriv_growth, low_eff_factor=low_eff_factor,
            cap_min=cap_min, cap_factor=cap_factor)
        n_eff_arr[i] = n_eff
        k_used_arr[i] = k_used
        if abstained or idx.shape[0] < d + 1:
            # idx.shape[0] < d+1 can only happen for a pathologically tiny
            # dataset (cap smaller than d+1); treat it the same as an
            # explicit abstention rather than relying on downstream NaN
            # comparisons (always False in numpy) to fall through to the
            # same "unreliable" verdict.
            abstained_arr[i] = True
            flag_low[i] = True
            if keep_points:
                points.append(None)
            continue
        abstained_arr[i] = abstained

        U_i, X_i, V_i, P_i = U[i], X[i], V[i], P[i]
        X_neigh, U_neigh, V_neigh = X[idx], U[idx], V[idx]
        a_i_obs, a_neigh_obs = a_all[i], a_all[idx]

        intrinsic = fit_intrinsic_jacobian(U_i, X_i, X_neigh, U_neigh, a_i_obs,
                                            a_neigh_obs, w, ridge_alpha=ridge_alpha)
        ambient = fit_ambient_jacobian(X_i, V_i, P_i, X_neigh, V_neigh, w, d,
                                        ridge_alpha=ridge_alpha)
        z = edge_coordinates(U_i, X_i, X_neigh)
        O = procrustes_transport_batch(U_i, U_neigh)
        diag = compute_point_diagnostics(
            U_i, intrinsic, ambient, z, O, U_neigh, w, n_eff, k_used, abstained, d,
            low_eff_factor=low_eff_factor, kappa_threshold=kappa_threshold,
            boundary_threshold=boundary_threshold)

        J_intrinsic[i] = intrinsic.J
        intercept_intrinsic[i] = intrinsic.intercept
        B[i] = ambient.B
        J_amb[i] = ambient.J_amb
        N[i] = ambient.N
        kappa_int_arr[i] = intrinsic.kappa
        kappa_amb_arr[i] = ambient.kappa
        residual_int_arr[i] = intrinsic.residual
        residual_amb_arr[i] = ambient.residual
        delta_cons_arr[i] = diag.delta_cons
        cons_scale_arr[i] = diag.cons_scale
        transport_err_arr[i] = diag.transport_error
        boundary_arr[i] = diag.boundary_moment
        flag_low[i] = diag.flags["low_effective_sample"]
        flag_ill[i] = diag.flags["ill_conditioned"]
        flag_bound[i] = diag.flags["possible_boundary_or_branch"]
        diag_by_point[i] = diag
        if keep_points:
            points.append(PointResult(intrinsic=intrinsic, ambient=ambient, diagnostics=diag))

    # --- Pass 2: batch-relative transport_inconsistent flag + reliable ---
    valid = ~abstained_arr
    if np.any(valid):
        cons_scale_floor = max(cons_scale_floor_min,
                                cons_scale_floor_factor * float(np.nanmedian(cons_scale_arr[valid])))
    else:
        cons_scale_floor = cons_scale_floor_min
    for i in range(n):
        if abstained_arr[i]:
            reliable_arr[i] = False
            continue
        flag_trans[i] = ((delta_cons_arr[i] > cons_threshold) and (cons_scale_arr[i] > cons_scale_floor)) \
            or (transport_err_arr[i] > transport_threshold)
        if i in diag_by_point:
            diag_by_point[i].flags["transport_inconsistent"] = bool(flag_trans[i])
            diag_by_point[i].flags["reliable"] = not (flag_low[i] or flag_ill[i] or flag_trans[i] or flag_bound[i])
        reliable_arr[i] = not (flag_low[i] or flag_ill[i] or flag_trans[i] or flag_bound[i])

    return JacobianFieldResult(
        n=n, d=d, D=D, J_intrinsic=J_intrinsic, intercept_intrinsic=intercept_intrinsic,
        B=B, J_amb=J_amb, N=N, n_eff=n_eff_arr, k_used=k_used_arr,
        kappa_intrinsic=kappa_int_arr, kappa_ambient=kappa_amb_arr,
        residual_intrinsic=residual_int_arr, residual_ambient=residual_amb_arr,
        delta_cons=delta_cons_arr, transport_error=transport_err_arr,
        boundary_moment=boundary_arr, abstained=abstained_arr, reliable=reliable_arr,
        flag_low_effective_sample=flag_low, flag_ill_conditioned=flag_ill,
        flag_transport_inconsistent=flag_trans,
        flag_possible_boundary_or_branch=flag_bound, points=points,
    )
