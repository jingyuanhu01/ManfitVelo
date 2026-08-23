"""Phase 3 (PLAN.md 3.1-3.2): connection-aware smoothing of a fitted
intrinsic-Jacobian field, over the graph's own discrete connection
(Procrustes transport `O_ij`), matrix-free.

Started 2026-08-15 on **partial** evidence for PLAN.md's 2.7 go/no-go (the
lightweight Phase 2 pass in `run_phase2_method_comparison.py`, not the full
2.5-2.7 checklist) -- proceeding was an explicit user decision, not a full
gate pass; see `DECISIONS.md`.

Zero dependency on ``scripts/``/``simulation/`` (same rule as
``intrinsic_jacobian.py``): operates on plain arrays.

## 3.1 Objective (design, not pre-existing anywhere -- see module docstring
of ``intrinsic_jacobian.py`` for the general pattern this follows)

The beamer deck (``Notes/intrinsic_jacobian_velocity_beamer.tex`` lines
258-280) states the *un-weighted* Singer-Wu-style connection-Laplacian
objective::

    min_{J_i} sum_i ||J_i - J_i_loc||_F^2
               + lambda_J sum_(i,j) w_ij ||J_i - O_ij J_j O_ij^T||_F^2

PLAN.md 3.1 additionally asks for a **two-stage** objective with a
per-point fidelity weight ``alpha_i`` "tied to local effective-sample/
conditioning/variance" -- no formula given anywhere. Designed here as::

    min_{J_i} sum_i alpha_i ||J_i - J_i_loc||_F^2
               + (lambda_J/2) sum_i sum_{j in N(i)} w_ij ||J_i - O_ij J_j O_ij^T||_F^2

    alpha_i = (n_eff_i / max(kappa_i, 1)) / (1 + residual_i)

``sum_i sum_{j in N(i)}`` ranges over the *directed* edge list built by
:func:`build_symmetrized_graph` -- both `(i,j)` and `(j,i)` are present as
separate entries (each with its own independently-computed `O_ij`, see that
function's docstring), so every undirected edge is walked from both
endpoints and would otherwise be counted twice. The explicit `1/2` keeps
that from doubling the effective smoothing strength, and is exactly what
makes the per-point stationarity condition in 3.2 fall out with a bare
(un-doubled) `lambda_J`, matching what :func:`connection_aware_smooth`
actually solves -- not an extra tuning knob.

i.e. points with more effective neighbors, a better-conditioned local
design, and a smaller regression residual are trusted more (kept closer to
their own local fit); points with weak local evidence get pulled more
toward their (transported) neighbors. `alpha_i` is normalized by its own
median across points before use, so its scale interacts predictably with
`lambda_J` regardless of the scenario's own units.

The graph used for smoothing is **symmetrized**: `(i,j)` edges come from
whichever ManfitVelo-style geometry graph (`neighbors`, `weights`) was
supplied, made symmetric via `W_sym = (W + W^T)/2` (both directions kept,
each with `O_ji = O_ij^T` by construction) -- this is what makes the
resulting normal equations an honest SPD (self-adjoint, positive-definite)
system rather than requiring separate bookkeeping for "i's own row" vs.
"i appearing as someone else's neighbor" edges (see the objective's
gradient derivation in this module's inline comments).

## 3.2 Matrix-free solve

Setting the gradient of the (quadratic, convex) objective to zero gives a
per-point linear system::

    [alpha_i + lambda_J * deg_i] J_i - lambda_J * sum_j w_ij O_ij J_j O_ij^T
        = alpha_i * J_i_loc

with `deg_i = sum_j w_ij` (the symmetrized weighted degree) -- the `1/2` in
3.1's objective cancels exactly against the two directed edges `(i,j)` and
`(j,i)` each contributing to this same stationarity condition (once as
`i`'s own row, once through `j`'s row via `O_ji = O_ij^T`), leaving a bare
`lambda_J` here. This is solved
via `scipy.sparse.linalg.cg` against a `LinearOperator` that only ever
evaluates `A @ vec(J)` through the edge list (`O_ij @ J_j @ O_ij^T` matvecs)
-- no `(n d^2) x (n d^2)` dense or sparse matrix is ever assembled, matching
PLAN.md 3.2's explicit no-dense-system requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg

EPS = 1e-12


@dataclass
class SymmetrizedGraph:
    edge_i: np.ndarray   # (m,) int
    edge_j: np.ndarray   # (m,) int
    O_ij: np.ndarray      # (m, d, d)
    w_edge: np.ndarray    # (m,) float, symmetric: edge (i,j) and (j,i) have equal weight
    degree: np.ndarray    # (n,) float, weighted degree per point


def build_symmetrized_graph(U: np.ndarray, neighbors: np.ndarray, weights: np.ndarray) -> SymmetrizedGraph:
    """Symmetrize a (possibly directed, e.g. kNN) geometry graph into an
    undirected edge list with Procrustes transports, both directions
    explicit (`O_ji = O_ij^T` computed independently per edge, not assumed).
    """
    n, k = neighbors.shape
    d = U.shape[2]
    src = np.repeat(np.arange(n), k)
    dst = neighbors.reshape(-1)
    w = weights.reshape(-1)
    valid = (dst != src) & (w > 0)
    src, dst, w = src[valid], dst[valid], w[valid]
    W_dir = sp.coo_matrix((w, (src, dst)), shape=(n, n)).tocsr()
    W_sym = ((W_dir + W_dir.T) * 0.5).tocoo()
    mask = (W_sym.row != W_sym.col) & (W_sym.data > EPS)
    edge_i, edge_j, w_edge = W_sym.row[mask], W_sym.col[mask], W_sym.data[mask]

    M = np.einsum("mDi,mDj->mij", U[edge_i], U[edge_j])  # U_i^T U_j per edge
    L, _, Rt = np.linalg.svd(M)
    O_ij = L @ Rt

    degree = np.zeros(n)
    np.add.at(degree, edge_i, w_edge)
    return SymmetrizedGraph(edge_i=edge_i, edge_j=edge_j, O_ij=O_ij, w_edge=w_edge, degree=degree)


def compute_fidelity_weights(n_eff: np.ndarray, kappa: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """`alpha_i` (PLAN.md 3.1's "tied to local effective-sample/
    conditioning/variance", formula designed here -- see module docstring).
    Normalized so `median(alpha) == 1`.
    """
    alpha = (n_eff / np.maximum(kappa, 1.0)) / (1.0 + np.maximum(residual, 0.0))
    med = np.median(alpha[np.isfinite(alpha)]) if np.any(np.isfinite(alpha)) else 1.0
    return alpha / max(med, EPS)


def _matvec_factory(graph: SymmetrizedGraph, alpha: np.ndarray, lambda_J: float, n: int, d: int):
    def matvec(x_flat: np.ndarray) -> np.ndarray:
        J = x_flat.reshape(n, d, d)
        out = alpha[:, None, None] * J + lambda_J * graph.degree[:, None, None] * J
        if graph.edge_i.size:
            transported = np.einsum("mab,mbc,mdc->mad", graph.O_ij, J[graph.edge_j], graph.O_ij)
            contrib = lambda_J * graph.w_edge[:, None, None] * transported
            np.add.at(out, graph.edge_i, -contrib)
        return out.reshape(-1)
    return matvec


@dataclass
class SmoothingResult:
    J_smoothed: np.ndarray   # (n, d, d)
    alpha: np.ndarray         # (n,)
    graph: SymmetrizedGraph
    cg_info: int               # scipy cg return code: 0 = converged
    n_iter: int                # actual number of CG iterations performed


def connection_aware_smooth(J_hat: np.ndarray, U: np.ndarray, neighbors: np.ndarray,
                             weights: np.ndarray, n_eff: np.ndarray, kappa: np.ndarray,
                             residual: np.ndarray, lambda_J: float = 1.0,
                             abstained: Optional[np.ndarray] = None,
                             rtol: float = 1e-10, maxiter: int = 2000) -> SmoothingResult:
    """Solve the 3.1 objective via the 3.2 matrix-free CG solve.

    Points with `abstained[i]=True` (no local fit at all, `J_hat[i]` is NaN)
    are given `alpha_i=0` (pure interpolation from neighbors, no fidelity
    pull) and their `J_hat` row is replaced by the (unweighted) neighbor
    mean before solving, purely so CG's `x0`/matvec never touch a NaN.
    """
    n, d, _ = J_hat.shape
    if abstained is None:
        abstained = np.zeros(n, dtype=bool)

    graph = build_symmetrized_graph(U, neighbors, weights)
    alpha = compute_fidelity_weights(n_eff, kappa, residual)
    alpha = np.where(abstained, 0.0, alpha)
    alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.any(alpha > 0):
        raise ValueError("connection smoothing needs at least one point with positive fidelity weight")

    J_fill = J_hat.copy()
    if np.any(abstained):
        fallback = np.nanmean(J_hat, axis=0)
        fallback = np.nan_to_num(fallback, nan=0.0)
        J_fill[abstained] = fallback

    b = (alpha[:, None, None] * J_fill).reshape(-1)
    matvec = _matvec_factory(graph, alpha, lambda_J, n, d)
    A = LinearOperator((n * d * d, n * d * d), matvec=matvec, dtype=float)
    x0 = J_fill.reshape(-1)
    n_iter = 0

    def count_iteration(_xk):
        nonlocal n_iter
        n_iter += 1

    sol, info = cg(A, b, x0=x0, rtol=rtol, maxiter=maxiter, callback=count_iteration)
    J_smoothed = sol.reshape(n, d, d)
    return SmoothingResult(J_smoothed=J_smoothed, alpha=alpha, graph=graph,
                           cg_info=info, n_iter=n_iter)
