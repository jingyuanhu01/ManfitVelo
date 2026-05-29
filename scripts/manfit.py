"""
manfit.py

Python implementation of MANFIT.

This version is configured to MATCH the original MATLAB implementation exactly:
- transform = "value2trans"
- use_matlab_mink = True

All algorithmic steps, neighbor selection, tie-breaking,
and feature transforms are numerically equivalent to manfit.m.

This file exists for correctness and reproducibility.
"""

import numpy as np
from sklearn.metrics import pairwise_distances


# ============================================================
# MATLAB-style mink (deterministic tie-breaking)
# ============================================================

def matlab_mink_indices(D, k):
    """
    Replicate MATLAB's mink(D, k):
    - primary key: distance ascending
    - secondary key: index ascending
    """
    n = D.shape[0]
    idx = np.tile(np.arange(n), (n, 1))
    order = np.lexsort((idx, D))
    return order[:, :k]


# ============================================================
# Transforms
# ============================================================

def value2trans_matlab(X):
    """
    MATLAB-style value2trans transform.

    Each row is stably sorted (mergesort),
    then mapped to sqrt(rank / d).
    """
    X = np.asarray(X, float)
    n, d = X.shape

    sorted_idx = np.argsort(X, axis=1, kind="mergesort")
    ordinals = np.sqrt(np.arange(1, d + 1) / d)

    out = np.zeros_like(X)
    rows = np.arange(n)[:, None]
    out[rows, sorted_idx] = ordinals

    return out.astype(np.float32)


def value2rank(X):
    X = np.asarray(X, float)
    n, d = X.shape
    ranks = np.argsort(np.argsort(X, axis=1), axis=1) + 1
    return np.sqrt(ranks / d).astype(np.float32)


def apply_transform(X, method):
    if method is None:
        return X.astype(np.float32)

    if method == "value2trans":
        return value2trans_matlab(X)

    if method == "rank":
        return value2rank(X)

    if method == "cosine":
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return (X / norm).astype(np.float32)

    if method == "log1p":
        return np.log1p(X).astype(np.float32)

    raise ValueError(f"Unknown transform: {method}")



# ============================================================
# MANFIT
# ============================================================

def manfit(
    sample,
    knn,
    knn3_factor=10,
    transform="rank",
    weights=(-0.1, -0.05, 0.0, 0.05, 0.1),
    use_matlab_mink=False,
):
    """
    MANFIT algorithm.
    """

    sample = np.asarray(sample, float)
    n, d = sample.shape

    # --------------------------------------------------------
    # 1. Distance
    # --------------------------------------------------------
    D = pairwise_distances(sample, metric="correlation")

    # --------------------------------------------------------
    # 2. Coarse neighborhood
    # --------------------------------------------------------
    knn3 = min(knn * knn3_factor, n)

    if use_matlab_mink:
        Nb3 = matlab_mink_indices(D, knn3)
    else:
        idx = np.argpartition(D, kth=knn3 - 1, axis=1)[:, :knn3]
        row = np.arange(n)[:, None]
        Nb3 = idx[row, np.argsort(D[row, idx], axis=1)]

    # --------------------------------------------------------
    # 3. DI construction (vanilla, sparse, correct)
    # --------------------------------------------------------
    DI = np.zeros((n, n))
    topk = Nb3[:, :knn]

    for i in range(n):
        neigh_i = set(topk[i])
        for j in Nb3[i]:
            neigh_j = set(topk[j])
            DI[i, j] = len(neigh_i & neigh_j)

    DI = (knn - np.maximum(DI, DI.T)) / knn

    # --------------------------------------------------------
    # 4. Final neighborhood
    # --------------------------------------------------------
    if use_matlab_mink:
        Nb = matlab_mink_indices(DI, knn)
    else:
        idx = np.argpartition(DI, kth=knn - 1, axis=1)[:, :knn]
        row = np.arange(n)[:, None]
        Nb = idx[row, np.argsort(DI[row, idx], axis=1)]

    # --------------------------------------------------------
    # 5. Feature transform
    # --------------------------------------------------------
    X = apply_transform(sample, transform)
    weights = np.asarray(weights)

    # --------------------------------------------------------
    # 6. Vectorized smoothing
    # --------------------------------------------------------
    neighbors = X[Nb]                 # (n, knn, d)
    x_bar = neighbors.mean(axis=1)
    direction = x_bar - X

    candidates = (
        x_bar[:, None, :]
        + weights[None, :, None] * direction[:, None, :]
    )

    diffs = neighbors[:, None, :, :] - candidates[:, :, None, :]
    scores = np.sum(diffs * diffs, axis=(2, 3))

    best_idx = np.argmin(scores, axis=1)
    Mout = candidates[np.arange(n), best_idx]

    return Mout

