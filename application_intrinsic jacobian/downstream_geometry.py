"""Tier-A downstream geometric quantities computed from a fitted Jacobian
field (``intrinsic_jacobian.JacobianFieldResult``).

Zero dependency on ``scripts/``/``simulation/`` (same rule as
``intrinsic_jacobian.py``) — everything here is plain numpy over the arrays
``estimate_jacobian_field`` already produced.

A1 (acceleration decomposition) and A2 (strain-rotation decomposition) are
the two Tier-A candidates in scope this sprint (PLAN.md's downstream
workstream, Tier A1/A2; see ``DECISIONS.md``). Gauge-invariance of every
scalar produced here follows directly from the estimator's own equivariance
property ``J_i -> Q_i^T J_i Q_i`` under a tangent-frame rotation
``U_i -> U_i Q_i`` (``Notes/intrinsic_jacobian_velocity_beamer.tex`` lines
220-228): trace, Frobenius norms of the symmetric/antisymmetric parts, and
eigenvalues of the symmetric part are all invariant under orthogonal
similarity ``Q^T (.) Q``, because that's exactly what those operations are
built to be invariant to. This is asserted as a direct unit test
(``test_downstream_geometry.py``, T10), not argued informally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass
class AccelerationDecomposition:
    a_int: np.ndarray       # (n, D) tangential ("intrinsic") acceleration
    a_normal: np.ndarray    # (n, D) curvature-induced ("normal") acceleration
    a_amb: np.ndarray       # (n, D) total ambient acceleration, == a_int + a_normal exactly
    R_bend: np.ndarray      # (n,) ||a_normal|| / (||a_int|| + eps)


def acceleration_decomposition(J_amb: np.ndarray, N: np.ndarray, V: np.ndarray,
                                P: np.ndarray) -> AccelerationDecomposition:
    """A1: intrinsic/normal/ambient acceleration and the curvature-flow
    ratio ``R_bend``, built entirely from the *ambient*-regression
    decomposition (never mixed with the intrinsic-regression ``J``, whose
    frame/scale differs and would break the exact identity below).

    The velocity is first projected onto the tangent space,
    ``v_tan = P @ v``, before any of the three accelerations are computed.
    This makes ``a_amb = a_int + a_normal`` an *exact* algebraic identity
    for every input (not just tangential-by-construction ones): with
    ``J_amb = P B P`` and ``N = (I-P) B P``, and using ``P v_tan = v_tan``
    (idempotence), ``J_amb v_tan + N v_tan = B P v_tan = B v_tan``. For the
    M5/M6/oracle sources ``v`` is already tangential so this changes
    nothing; for ``raw_noisy`` it means the three accelerations describe
    the dynamics of the velocity field's tangential part only, which is the
    physically meaningful quantity here.
    """
    v_tan = np.einsum("nij,nj->ni", P, V)
    a_int = np.einsum("nij,nj->ni", J_amb, v_tan)
    a_normal = np.einsum("nij,nj->ni", N, v_tan)
    a_amb = np.einsum("nij,nj->ni", J_amb + N, v_tan)  # == B @ v_tan, see docstring
    R_bend = np.linalg.norm(a_normal, axis=1) / (np.linalg.norm(a_int, axis=1) + EPS)
    return AccelerationDecomposition(a_int=a_int, a_normal=a_normal, a_amb=a_amb, R_bend=R_bend)


@dataclass
class StrainRotationDecomposition:
    S: np.ndarray               # (n, d, d) symmetric part
    A: np.ndarray                # (n, d, d) antisymmetric part
    divergence: np.ndarray       # (n,) = trace(S) = trace(J)
    shear: np.ndarray            # (n,) ||deviatoric(S)||_F
    vorticity_mag: np.ndarray    # (n,) ||A||_F  (magnitude only, see note below)
    eigvals_S: np.ndarray        # (n, d) ascending eigenvalues of S
    stability: np.ndarray        # (n,) object array of "stable"/"unstable"/"saddle"/"nan"


def strain_rotation_decomposition(J: np.ndarray, stability_tol: float = 1e-9) -> StrainRotationDecomposition:
    """A2: symmetric/antisymmetric split of a ``(n, d, d)`` Jacobian field.

    ``vorticity_mag`` reports ``||A||_F`` only — a *signed* rotation angle
    is not gauge-invariant under the full ``O(d)`` gauge group (a
    reflection ``det(Q)=-1`` flips its sign); this is documented, not
    "fixed" (T10 checks the sign-flip explicitly rather than treating it as
    a bug).

    Exact degeneracy for ``d=1`` (5 of the 9 canonical scenarios): ``A``
    and ``shear`` are identically zero — pure linear algebra, no rotation
    or shear is possible in one dimension. A2 only carries real signal on
    the four ``d=2`` scenarios; ``d=1`` scenarios contribute divergence-only
    diagnostics.
    """
    n, d, _ = J.shape
    Jt = np.transpose(J, (0, 2, 1))
    S = 0.5 * (J + Jt)
    A = 0.5 * (J - Jt)
    divergence = np.einsum("nii->n", S)
    dev = S - (divergence / d)[:, None, None] * np.eye(d)[None, :, :]
    shear = np.linalg.norm(dev, axis=(1, 2))
    vorticity_mag = np.linalg.norm(A, axis=(1, 2))

    eigvals_S = np.full((n, d), np.nan)
    stability = np.empty(n, dtype=object)
    for i in range(n):
        if np.any(np.isnan(S[i])):
            stability[i] = "nan"
            continue
        ev = np.linalg.eigvalsh(S[i])
        eigvals_S[i] = ev
        if np.all(ev <= stability_tol):
            stability[i] = "stable"
        elif np.all(ev >= -stability_tol):
            stability[i] = "unstable"
        else:
            stability[i] = "saddle"

    return StrainRotationDecomposition(S=S, A=A, divergence=divergence, shear=shear,
                                        vorticity_mag=vorticity_mag, eigvals_S=eigvals_S,
                                        stability=stability)


# --------------------------------------------------------------------------
# A5: Schur-decomposition cross-section (design from the ddHodge primary-
# source read, see NOVELTY_AUDIT.md/FUTURE_DIRECTIONS.md A5). Not part of
# PLAN.md's original Tier-A list -- added 2026-08-15 as a better-precedented
# replacement for this project's own failed marker-crossover validation.
# --------------------------------------------------------------------------

@dataclass
class SchurCrossSection:
    vectors: np.ndarray        # (d, k) orthonormal columns spanning the selected invariant subspace
    eigenvalues: np.ndarray     # (k,) complex eigenvalues of J restricted to this subspace
    block_start: int
    block_size: int


def schur_cross_section(J: np.ndarray, mode: str = "extreme") -> SchurCrossSection:
    """Select the invariant subspace of a (possibly non-symmetric) Jacobian
    associated with its most extreme eigenvalue, via the **real Schur
    decomposition** -- ddHodge's own design (Maehara & Ohkawa, *Nat Commun*
    2025) for picking a "cross-section" of local dynamics to project
    velocities/expression onto (their Figs. 3G, 5B, 5D). Schur, not
    eigendecomposition, is used specifically because a general ``J`` can
    have complex eigenvalues/eigenvectors (e.g. a spiral); the real Schur
    form (``scipy.linalg.schur(J, output="real")``) keeps everything in
    real arithmetic via a 2x2 diagonal block for each complex-conjugate
    eigenvalue pair, and that block's own 2 Schur-vector columns already
    span a genuine real-valued 2D invariant subspace (the rotation/spiral
    plane) without ever constructing a complex eigenvector by hand.

    For the common ``d=2`` case this is almost trivial (there is only one
    block, covering the whole tangent space, so ``mode`` doesn't matter);
    the block-selection logic below only starts doing real work at ``d>2``.

    ``mode``: ``"extreme"`` (largest ``|eigenvalue|``), ``"most_negative"``
    (most contracting/stabilizing direction), ``"most_positive"`` (most
    expanding/destabilizing) -- selects which diagonal block of the Schur
    form to report, ranked by each block's eigenvalue real part(s).
    """
    from scipy.linalg import schur as _schur
    d = J.shape[0]
    T, Z = _schur(J, output="real")
    blocks = []  # (start, size, eigenvalues)
    i = 0
    while i < d:
        is_2x2_block = i + 1 < d and abs(T[i + 1, i]) > 1e-9 * (abs(T[i, i]) + abs(T[i + 1, i + 1]) + EPS)
        if is_2x2_block:
            a, b, c, dd = T[i, i], T[i, i + 1], T[i + 1, i], T[i + 1, i + 1]
            tr, det = a + dd, a * dd - b * c
            disc = tr ** 2 - 4 * det
            if disc < 0:
                re, im = tr / 2.0, np.sqrt(-disc) / 2.0
                eigs = np.array([re + 1j * im, re - 1j * im])
            else:  # numerically real pair that landed in a 2x2 block; report both real roots
                sq = np.sqrt(disc)
                eigs = np.array([(tr + sq) / 2.0, (tr - sq) / 2.0], dtype=complex)
            blocks.append((i, 2, eigs))
            i += 2
        else:
            blocks.append((i, 1, np.array([T[i, i]], dtype=complex)))
            i += 1
    if mode == "most_negative":
        chosen = min(blocks, key=lambda b: float(np.min(b[2].real)))
    elif mode == "most_positive":
        chosen = max(blocks, key=lambda b: float(np.max(b[2].real)))
    elif mode == "extreme":
        chosen = max(blocks, key=lambda b: float(np.max(np.abs(b[2]))))
    else:
        raise ValueError("mode must be 'extreme', 'most_negative', or 'most_positive'")
    start, size, eigs = chosen
    return SchurCrossSection(vectors=Z[:, start:start + size], eigenvalues=eigs,
                              block_start=start, block_size=size)


# --------------------------------------------------------------------------
# Generalized zero-crossing detection + fixed-point finding/classification.
# Added 2026-08-15 at the user's direct request, reframing the Jacobian
# module as an organic extension of ManfitVelo's own geometric-discovery
# theme rather than a head-to-head competitor to any one prior method (see
# the chat / plan file for that framing decision). Not part of PLAN.md's
# original Tier-A list.
# --------------------------------------------------------------------------

@dataclass
class DivergenceTransitionEdges:
    edge_i: np.ndarray   # (m,) int
    edge_j: np.ndarray    # (m,) int


def divergence_sign_transition_edges(divergence: np.ndarray, neighbors: np.ndarray,
                                      reliable: np.ndarray,
                                      min_abs_value: float = 0.0) -> DivergenceTransitionEdges:
    """Geometry-graph generalization of the "does divergence change sign
    between neighboring points" check that was previously only implemented
    ad hoc, over a 1D phase-binned ordering, in
    ``run_cell_cycle_case_study.find_critical_points``. Operating directly
    on the geometry graph's own edges instead means this works for any
    topology (loops, branches, trees) the data manifold happens to have,
    not just a 1D cyclic ordering with a known phase variable.

    A transition edge ``(i,j)`` is one where both endpoints are ``reliable``
    and ``sign(divergence[i]) != sign(divergence[j])`` -- these edges are a
    graph-native analogue of ``downstream_geometry.py``'s A2 divergence
    (``tr(Dv)``, expansion/contraction), marking where the local intrinsic
    dynamics changes character from contracting to expanding (or vice
    versa) as you move along the manifold's own connectivity.

    ``min_abs_value`` (default 0.0, i.e. off) additionally requires
    ``abs(divergence[i]) >= min_abs_value`` and ``abs(divergence[j]) >=
    min_abs_value`` before counting a sign flip as a transition edge. This
    is a robustness filter: with pure sign comparison, a divergence field
    that is noisy near zero (e.g. real single-cell data -- see
    `cell_cycle_validation.md`'s "too many spurious crossings" finding for
    the earlier 1D version of this check) produces a flip at almost every
    edge, since noise on either side of zero flips sign trivially. Raising
    the threshold restricts transitions to edges where both endpoints have
    a confidently non-trivial divergence magnitude of opposite sign -- a
    parallel fix to the bin-count reduction that was found (not assumed) to
    reduce spurious crossings for the marker-based check.
    """
    n, k = neighbors.shape
    src = np.repeat(np.arange(n), k)
    dst = neighbors.reshape(-1)
    valid = (dst != src) & reliable[src] & reliable[dst]
    src, dst = src[valid], dst[valid]
    div_src, div_dst = divergence[src], divergence[dst]
    sign_src, sign_dst = np.sign(div_src), np.sign(div_dst)
    flip = (sign_src != sign_dst) & (sign_src != 0) & (sign_dst != 0)
    if min_abs_value > 0.0:
        flip &= (np.abs(div_src) >= min_abs_value) & (np.abs(div_dst) >= min_abs_value)
    return DivergenceTransitionEdges(edge_i=src[flip], edge_j=dst[flip])


def find_fixed_point_candidates(V: np.ndarray, neighbors: np.ndarray, reliable: np.ndarray,
                                 local_min_only: bool = True) -> np.ndarray:
    """Candidate genuine fixed points (``v ~= 0``) of the fitted flow, as
    distinct from A5's "extreme divergence" points (which need not have
    small velocity at all -- a point can be rapidly moving through a
    strongly expanding or contracting region without ever being near a
    fixed point of the dynamical system itself). Ranks ``reliable`` points
    by ambient speed ``||V||`` ascending; ``local_min_only=True``
    additionally requires the point's speed to be a strict local minimum
    among its own geometry-graph neighbors, so a single generically-slow
    region doesn't get flagged as many separate candidates.

    Returns indices into the original arrays, ordered from most to least
    promising (lowest speed first).
    """
    speed = np.linalg.norm(V, axis=1)
    idx = np.where(reliable)[0]
    if local_min_only:
        n, k = neighbors.shape
        is_local_min = np.ones(n, dtype=bool)
        for j in range(k):
            nbr = neighbors[:, j]
            valid = reliable & reliable[nbr] & (nbr != np.arange(n))
            worse = np.ones(n, dtype=bool)
            worse[valid] = speed[valid] < speed[nbr[valid]]
            is_local_min &= worse
        idx = idx[is_local_min[idx]]
    return idx[np.argsort(speed[idx])]


@dataclass
class FixedPointRefinement:
    x_refined: np.ndarray     # (D,)
    delta_local: np.ndarray    # (d,) the (possibly capped) Newton step in local coordinates
    step_capped: bool


def refine_fixed_point(X_i: np.ndarray, V_i: np.ndarray, U_i: np.ndarray, J_i: np.ndarray,
                        local_scale: float, max_step_frac: float = 0.5,
                        ridge_alpha: float = 0.05) -> FixedPointRefinement:
    """One Newton-style correction step, in local tangent coordinates,
    moving a fixed-point candidate from a data point toward the location
    the *local linear model* actually predicts zero velocity at:
    ``a_i = U_i^T V_i`` (local velocity coordinates); solve
    ``(J_i + ridge*I) delta = -a_i``; ``x* = X_i + U_i @ delta``.

    ``local_scale`` (e.g. the mean/median distance to this point's own
    geometry neighbors -- supplied by the caller, since this function has
    zero dependency on ``intrinsic_jacobian.py``'s internals per this
    module's own design rule) caps ``||delta||`` at
    ``max_step_frac * local_scale``: a Newton step from a *linear*
    approximation is only trustworthy within the neighborhood the model
    was actually fit on, and ``step_capped=True`` signals the correction
    left that trust region -- the refined location should be treated
    skeptically (or the raw data point used instead) whenever this fires.
    """
    d = U_i.shape[1]
    a_i = U_i.T @ V_i
    ridge = ridge_alpha * float(np.trace(J_i @ J_i.T)) / max(d, 1) + 1e-8
    delta = np.linalg.solve(J_i + ridge * np.eye(d), -a_i)
    cap = max_step_frac * local_scale
    norm = float(np.linalg.norm(delta))
    capped = norm > cap
    if capped and norm > EPS:
        delta = delta * (cap / norm)
    x_refined = X_i + U_i @ delta
    return FixedPointRefinement(x_refined=x_refined, delta_local=delta, step_capped=capped)


@dataclass
class FixedPointClassification:
    kind: str                # "attractor" | "repeller" | "saddle"
    motion: str                # "node" | "spiral"
    eigenvalues: np.ndarray     # complex, eigenvalues of the FULL (non-symmetric) J


def classify_fixed_point(J: np.ndarray, tol: float = 1e-9) -> FixedPointClassification:
    """Textbook dynamical-systems classification of a fixed point of
    ``dx/dt = v(x)``, via the eigenvalues of the **full** (possibly non-
    symmetric, possibly complex-eigenvalued) local Jacobian ``J`` --
    **not** the same thing as ``StrainRotationDecomposition.stability``,
    which classifies via eigenvalues of only the *symmetric* part ``S``
    (a strain/continuum-mechanics concept: local expansion/contraction
    sign-definiteness). The two are related but genuinely different
    quantities. Asymptotic stability of a true fixed point depends on the
    real parts of ``J``'s own eigenvalues, whereas ``S`` describes
    instantaneous strain along individual directions.

    ``kind``: "attractor" (all eigenvalue real parts < -tol), "repeller"
    (all real parts > tol), "saddle" (mixed signs). ``motion``: "spiral"
    if any eigenvalue has nonzero imaginary part (rotation while
    approaching/leaving), else "node".
    """
    eigs = np.linalg.eigvals(J)
    re = eigs.real
    if np.all(re < -tol):
        kind = "attractor"
    elif np.all(re > tol):
        kind = "repeller"
    else:
        kind = "saddle"
    motion = "spiral" if np.any(np.abs(eigs.imag) > tol) else "node"
    return FixedPointClassification(kind=kind, motion=motion, eigenvalues=eigs)
