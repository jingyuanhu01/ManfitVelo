"""Simple position-only MANFIT variant for S-curve experiments."""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


def manfit_ours(sample, sig, sample_init, op_average=1):
    """Fit a manifold using only positions.

    Parameters mirror the provided prototype. ``op_average`` is kept for API
    compatibility with the original snippet, but is not used.
    """
    del op_average

    sample = np.asarray(sample, dtype=float)
    Mout = np.copy(np.asarray(sample_init, dtype=float))
    N = Mout.shape[0]
    N0 = sample.shape[0]
    ns = np.arange(N0)

    r = 5 * sig / np.log10(N0)
    R = 10 * sig * np.sqrt(np.log(1 / sig)) / np.log10(N0)

    nbrs = NearestNeighbors(n_neighbors=5).fit(sample)

    for ii in range(N):
        x = Mout[ii, :]

        dists = squareform(pdist(np.vstack([x, sample])))[0, 1:]

        IDX1 = dists < 2 * r
        IDX1 = ns[IDX1]

        IDX2 = nbrs.kneighbors(x.reshape(1, -1), return_distance=False).flatten()
        IDX = np.union1d(IDX1, IDX2)

        BNbr = sample[IDX, :]
        xbar = np.mean(BNbr, axis=0) + np.finfo(float).eps

        dx = x - xbar
        dx_norm = np.linalg.norm(dx)
        if dx_norm > np.finfo(float).eps:
            dx = dx / dx_norm

        Q = np.linalg.qr(np.column_stack([dx, np.eye(dx.size)]))[0]

        sample_s = sample - x
        sample_s = sample_s @ Q

        CNbr = (np.abs(sample_s[:, 0]) < R) & (np.sum(sample_s[:, 1:] ** 2, axis=1) < r**2)

        if np.sum(CNbr) > 10:
            Mout[ii, :] = np.mean(sample[CNbr, :], axis=0)
        else:
            Mout[ii, :] = xbar

    return Mout
