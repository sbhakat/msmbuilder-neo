# The vector metrics are ported from distance_kernels.h, adapted from SciPy;
# its notice (also in THIRD_PARTY_LICENSES.txt):
#
# Author: Damian Eads
# Date:   September 22, 2007 (moved to new file on June 8, 2008)
#
# Copyright (c) 2007, 2008, Damian Eads. All rights reserved.
# Adapted for incorporation into Scipy, April 9, 2008.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#   - Redistributions of source code must retain the above
#     copyright notice, this list of conditions and the
#     following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer
#     in the documentation and/or other materials provided with the
#     distribution.
#   - Neither the name of the author nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Distance metrics between vectors and between molecular conformations.

Numba port of the former libdistance.pyx and its C kernels. The vector
metrics follow the C code's arithmetic exactly, including where float32
inputs are widened to double. RMSD uses mdtraj.rmsd (the same Theobald QCP
kernel the old code vendored from MDTraj) with the same target/reference
argument order, so results are unchanged.
"""
import numpy as np
import mdtraj as md
from numba import njit, prange

__all__ = ['assign_nearest', 'pdist', 'dist', 'cdist', 'sumdist']

VECTOR_METRICS = ("euclidean", "sqeuclidean", "cityblock", "chebyshev",
                  "canberra", "braycurtis", "hamming", "jaccard",
                  "cityblock")
_METRIC_CODES = {"euclidean": 0, "sqeuclidean": 1, "cityblock": 2, "chebyshev": 3,
                 "canberra": 4, "braycurtis": 5, "hamming": 6, "jaccard": 7}
_DBL_MAX = np.finfo(np.float64).max
_FLT_MAX = np.float32(np.finfo(np.float32).max)


# --- vector metrics --------------------------------------------------------------
# For float32 inputs the C++ kernels subtracted, added and took abs() in single
# precision and then widened to double. The casts below reproduce that.

@njit(cache=True, nogil=True, inline='always')
def _metric(code, X, i, Y, j):
    n = X.shape[1]
    if code == 0 or code == 1:  # (sq)euclidean
        s = 0.0
        for k in range(n):
            d = np.float64(X[i, k] - Y[j, k])
            s += d * d
        return np.sqrt(s) if code == 0 else s
    if code == 2:  # cityblock
        s = 0.0
        for k in range(n):
            s = s + abs(np.float64(X[i, k] - Y[j, k]))
        return s
    if code == 3:  # chebyshev
        m = 0.0
        for k in range(n):
            d = abs(np.float64(X[i, k] - Y[j, k]))
            if d > m:
                m = d
        return m
    if code == 4:  # canberra
        tot = 0.0
        for k in range(n):
            num = abs(np.float64(X[i, k] - Y[j, k]))
            # The old kernels were C++, where fabs(float) returns float, so for
            # float32 input this sum was formed in single precision.
            den = np.float64(abs(X[i, k]) + abs(Y[j, k]))
            if den > 0.0:
                tot += num / den
        return tot
    if code == 5:  # braycurtis
        s1 = 0.0
        s2 = 0.0
        for k in range(n):
            s1 += abs(np.float64(X[i, k] - Y[j, k]))
            s2 += abs(np.float64(X[i, k] + Y[j, k]))
        return s1 / s2
    if code == 6:  # hamming
        s = 0.0
        for k in range(n):
            if X[i, k] != Y[j, k]:
                s += 1.0
        return s / n
    # jaccard (real-valued, as in the original C code)
    num = 0.0
    den = 0.0
    for k in range(n):
        nz = (X[i, k] != 0.0) or (Y[j, k] != 0.0)
        if nz:
            den += 1.0
            if X[i, k] != Y[j, k]:
                num += 1.0
    return num / den


@njit(cache=True, parallel=True)
def _assign_rows(code, X, rows, Y, assignments, mins):
    for ii in prange(rows.shape[0]):
        i = rows[ii]
        min_d = _DBL_MAX
        for j in range(Y.shape[0]):
            d = _metric(code, X, i, Y, j)
            if d < min_d:
                min_d = d
                assignments[ii] = j
        mins[ii] = min_d


@njit(cache=True, parallel=True)
def _cdist(code, XA, XB, out):
    for i in prange(XA.shape[0]):
        for j in range(XB.shape[0]):
            out[i, j] = _metric(code, XA, i, XB, j)


@njit(cache=True, parallel=True)
def _pdist(code, X, rows, out):
    n = rows.shape[0]
    for ii in prange(n):
        # condensed index of (ii, ii+1)
        k = ii * n - ii * (ii + 1) // 2
        for jj in range(ii + 1, n):
            out[k] = _metric(code, X, rows[ii], X, rows[jj])
            k += 1


@njit(cache=True, parallel=True)
def _dist(code, X, rows, y2d, out):
    for ii in prange(rows.shape[0]):
        out[ii] = _metric(code, X, rows[ii], y2d, 0)


@njit(cache=True, nogil=True)
def _sumdist(code, X, pairs):
    s = 0.0
    for p in range(pairs.shape[0]):
        s += _metric(code, X, pairs[p, 0], X, pairs[p, 1])
    return s


@njit(cache=True, nogil=True)
def _serial_sum(values):
    s = 0.0
    for v in values:
        s += v
    return s


def _metric_code(metric):
    if isinstance(metric, bytes):
        metric = metric.decode('ascii')
    if metric not in VECTOR_METRICS:
        raise ValueError('metric must be one of %s' %
                         ', '.join("'%s'" % s for s in VECTOR_METRICS))
    return _METRIC_CODES[metric]


def _is_rmsd(metric):
    return metric in ('rmsd', b'rmsd')


def _rows(n, X_indices):
    if X_indices is None:
        return np.arange(n, dtype=np.intp)
    return np.ascontiguousarray(X_indices, dtype=np.intp)


def _check_float_pair(A, B, message):
    if A.dtype == np.float64 and B.dtype == np.float64:
        return np.ascontiguousarray(A), np.ascontiguousarray(B)
    if A.dtype == np.float32 and B.dtype == np.float32:
        return np.ascontiguousarray(A), np.ascontiguousarray(B)
    raise TypeError(message)


# --- RMSD via mdtraj ----------------------------------------------------------------

def _centered(traj):
    if traj._rmsd_traces is None:
        traj.center_coordinates()
    return traj


def _subset(traj, indices):
    """Frames ``indices`` of an already-centered trajectory, keeping its RMSD
    traces (mdtraj's slicing drops them, which would trigger re-centering)."""
    sub = traj[indices]
    sub._rmsd_traces = traj._rmsd_traces[indices]
    return sub


def _rmsd_to(target, reference, frame):
    """RMSD of every frame of target to reference[frame], as float32, with
    target frames as the first argument of the QCP kernel (as before)."""
    return md.rmsd(target, reference, frame=int(frame), precentered=True)


def _check_same_atoms(X, Y):
    if X.xyz.shape[1] != Y.xyz.shape[1]:
        raise ValueError("Input trajectories must have same number of atoms. "
                         "found %d and %d." % (X.xyz.shape[1], Y.xyz.shape[1]))


@njit(cache=True, nogil=True)
def _assign_from_columns(D):
    """Row-wise argmin of a float32 matrix, scanning columns in order with a
    strict '<' starting from FLT_MAX, and the double sum of the minima."""
    n, m = D.shape
    assignments = np.zeros(n, dtype=np.intp)
    inertia = 0.0
    for i in range(n):
        min_d = np.float32(3.4028234663852886e38)
        for j in range(m):
            if D[i, j] < min_d:
                min_d = D[i, j]
                assignments[i] = j
        inertia += np.float64(min_d)
    return assignments, inertia


def _assign_nearest_rmsd(X, Y, X_indices=None):
    _check_same_atoms(X, Y)
    _centered(X)
    _centered(Y)
    target = X if X_indices is None else _subset(X, _rows(X.n_frames, X_indices))
    D = np.empty((target.n_frames, Y.n_frames), dtype=np.float32)
    for j in range(Y.n_frames):
        D[:, j] = _rmsd_to(target, Y, j)
    return _assign_from_columns(D)


def _cdist_rmsd(XA, XB):
    _centered(XA)
    _centered(XB)
    if XA.xyz.shape[1] != XB.xyz.shape[1]:
        raise ValueError('XA and XB must have the same number of atoms')
    out = np.zeros((XA.n_frames, XB.n_frames), dtype=np.double)
    for j in range(XB.n_frames):
        out[:, j] = _rmsd_to(XA, XB, j)
    return out


def _pdist_rmsd(X, X_indices=None):
    _centered(X)
    sub = X if X_indices is None else _subset(X, _rows(X.n_frames, X_indices))
    n = sub.n_frames
    out = np.zeros(n * (n - 1) // 2, dtype=np.double)
    # Pair (i, j), i < j, was msd(frame i, frame j): row i of the column for
    # reference frame j. Its condensed index is n*i - i*(i+1)/2 + (j - i - 1).
    for j in range(1, n):
        # only frames 0..j-1 are needed as targets; slice without copying
        head = sub.slice(slice(0, j), copy=False)
        head._rmsd_traces = sub._rmsd_traces[:j]
        i = np.arange(j)
        out[n * i - i * (i + 1) // 2 + (j - i - 1)] = _rmsd_to(head, sub, j)
    return out


def _dist_rmsd(X, y, X_indices=None):
    _check_same_atoms(X, y)
    _centered(X)
    _centered(y)
    target = X if X_indices is None else _subset(X, _rows(X.n_frames, X_indices))
    return _rmsd_to(target, y, 0).astype(np.double)


def _sumdist_rmsd(X, pair_indices):
    pair_indices = np.asarray(pair_indices, dtype=np.intp)
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError('pair_indices must be of shape = (n_pairs, 2)')
    _centered(X)
    values = np.empty(len(pair_indices), dtype=np.float32)
    for jj in np.unique(pair_indices[:, 1]):
        sel = pair_indices[:, 1] == jj
        values[sel] = _rmsd_to(X, X, jj)[pair_indices[sel, 0]]
    return _serial_sum(values.astype(np.float64))


# --- public interface ---------------------------------------------------------------

def assign_nearest(X, Y, metric, X_indices=None):
    """assign_nearest(X, Y, metric, X_indices=None)

    For each point in X, compute the index of the nearest element in Y.

    Parameters
    ----------
    X : array, shape = (n_samples_X, n_features) or md.Trajectory
        The data array
    Y : array, shape = (n_samples_Y, n_features) or md.Trajectory
        The array of cluster centers
    metric : {"euclidean", "sqeuclidean", "cityblock", "chebyshev", "canberra",
              "braycurtis", "hamming", "jaccard", "cityblock", "rmsd"}
        The distance metric to use. metric = "rmsd" requires that both X
        and cluster centers be of type md.Trajectory; other distance metrics
        require that they be arrays.
    X_indices : array of indices, or None
        If supplied, only data points with index in X_indices will be
        considered. `X_indices = None` is equivalent to
        `X_indices = range(len(X))`

    Returns
    -------
    assignments : array, shape=(len(X),), or shape=(len(X_indices),)
        For each point in `X`, or `X[X_indices]`, the index of the nearest
        point in `Y`.
    inertia : double
        The sum of the distance from each point in X[X_indices] to its assigned
        point in `Y`.

    See Also
    --------
    mdtraj.rmsd
    """
    if isinstance(X, md.Trajectory) and isinstance(Y, md.Trajectory) and _is_rmsd(metric):
        return _assign_nearest_rmsd(X, Y, X_indices)
    if not isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
        raise TypeError()
    code = _metric_code(metric)
    X, Y = _check_float_pair(X, Y, 'X and y must be both float32 or float64')
    assert X.shape[1] == Y.shape[1]
    rows = _rows(X.shape[0], X_indices)
    assignments = np.zeros(len(rows), dtype=np.intp)
    mins = np.empty(len(rows))
    _assign_rows(code, X, rows, Y, assignments, mins)
    return assignments, _serial_sum(mins)


def cdist(XA, XB, metric):
    """cdist(XA, XB, metric):

    Computes distance between each pair of the two collections of inputs.

    Parameters
    ----------
    XA : array, shape = (na_samples, m_features) or md.Trajectory
        The data array
    XB : array, shape = (nb_samples, m_features) or md.Trajectory
        The reference array
    metric : {"euclidean", "sqeuclidean", "cityblock", "chebyshev", "canberra",
              "braycurtis", "hamming", "jaccard", "cityblock", "rmsd"}
        The distance metric to use. metric = "rmsd" requires that X be of type
        md.Trajectory; other distance metrics require that it be a numpy array

    Returns
    -------
    Y : array, shape = (na_samples, nb_samples)
        For each sample i in XA and each sample j in XB
        ``dist(u=XA[i], v=XB[j])`` is computed and stored in Y[i,j]

    Note
    ----
    Not implemented for X_indices != None

    See Also
    --------
    mdtraj.rmsd
    scipy.spatial.distance.cdist
    """
    if isinstance(XA, md.Trajectory) and isinstance(XB, md.Trajectory) and _is_rmsd(metric):
        return _cdist_rmsd(XA, XB)
    if not (isinstance(XA, np.ndarray) and isinstance(XB, np.ndarray)):
        raise TypeError('XA and XB must be numpy arrays')
    code = _metric_code(metric)
    XA, XB = _check_float_pair(XA, XB, 'XA and XB must be identically float32 or float64')
    if XA.shape[1] != XB.shape[1]:
        raise ValueError('XA and XB must have the same number of columns')
    out = np.zeros((XA.shape[0], XB.shape[0]), dtype=np.double)
    _cdist(code, XA, XB, out)
    return out


def pdist(X, metric, X_indices=None):
    """pdist(X, metric, X_indices=None)

    Pairwise distances between observations

    Parameters
    ----------
    X : array, shape = (n_samples, n_features) or md.Trajectory
        The data array
    metric : {"euclidean", "sqeuclidean", "cityblock", "chebyshev", "canberra",
              "braycurtis", "hamming", "jaccard", "cityblock", "rmsd"}
        The distance metric to use. metric = "rmsd" requires that X be of type
        md.Trajectory; other distance metrics require that it be a numpy array
    X_indices : array of indices, or None
        If supplied, only data points with index in X_indices will be considered.
        `X_indices = None` is equivalent to `X_indices = range(len(X))`

    Returns
    -------
    dist : ndarray, size=(len(X) choose 2,) or shape=(len(X_indices) choose 2,)
        Returns a condensed distance matrix `dist`.  For
        each :math:`i` and :math:`j` (where :math:`i<j<n`), the
        metric ``dist(u=X[i], v=X[j])`` is computed and stored in entry ``ij``.

    See Also
    --------
    mdtraj.rmsd
    scipy.spatial.distance.pdist
    scipy.spatial.distance.squareform
    """
    if isinstance(X, md.Trajectory) and _is_rmsd(metric):
        return _pdist_rmsd(X, X_indices)
    if not isinstance(X, np.ndarray):
        raise TypeError()
    code = _metric_code(metric)
    if X.dtype not in (np.float32, np.float64):
        raise TypeError('X must be float32 or float64')
    X = np.ascontiguousarray(X)
    rows = _rows(X.shape[0], X_indices)
    n = len(rows)
    out = np.zeros(n * (n - 1) // 2, dtype=np.double)
    _pdist(code, X, rows, out)
    return out


def dist(X, y, metric, X_indices=None):
    """dist(X, y, metric, X_indices=None)

    Distance from one point to many points.

    Parameters
    ----------
    X : array, shape = (n_samples, n_features) or md.Trajectory
        A data array
    y : array, shape = (n_features) or md.Trajectory of length 1
        A single data point
    metric : {"euclidean", "sqeuclidean", "cityblock", "chebyshev", "canberra",
              "braycurtis", "hamming", "jaccard", "cityblock", "rmsd"}
        The distance metric to use. metric = "rmsd" requires that both X
        and cluster centers be of type md.Trajectory; other distance metrics
        require that they be arrays.

    Returns
    -------
    Y : ndarray, shape=(len(X),) or len(X_indices)
        The distance from Y to each `X` or `X[X_indices]`.

    See Also
    --------
    mdtraj.rmsd
    scipy.spatial.distance.cdist
    """
    if isinstance(X, md.Trajectory) and isinstance(y, md.Trajectory) and _is_rmsd(metric):
        return _dist_rmsd(X, y, X_indices)
    if not isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        raise TypeError()
    code = _metric_code(metric)
    X, y = _check_float_pair(X, y, 'X and y must be both float32 or float64')
    assert X.shape[1] == y.shape[0]
    rows = _rows(X.shape[0], X_indices)
    out = np.zeros(len(rows), dtype=np.double)
    _dist(code, X, rows, y.reshape(1, -1), out)
    return out


def sumdist(X, metric, pair_indices):
    """sumdist(X, metric, pair_indices)

    Sum of the distance between pairs of points

    Parameters
    ----------
    X : array, shape = (n_samples, n_features) or md.Trajectory
        A data array
    metric : {"euclidean", "sqeuclidean", "cityblock", "chebyshev", "canberra",
              "braycurtis", "hamming", "jaccard", "cityblock", "rmsd"}
        The distance metric to use. metric = "rmsd" requires that both X
        and cluster centers be of type md.Trajectory; other distance metrics
        require that they be arrays.
    pair_indices : array, shape = (n_pairs, 2)
        Each element in pair_indices is a tuple of two indices -- a pair of
        elements in X to include in the summation.

    Returns
    -------
    s : float
        The sum of the distance between each pair of elements:
        ``sum(dist(X[p[0]], X[p[1]]) for p in pair_indices)``
    """
    if isinstance(X, md.Trajectory) and _is_rmsd(metric):
        return _sumdist_rmsd(X, pair_indices)
    code = _metric_code(metric)
    if X.dtype not in (np.float32, np.float64):
        raise TypeError('X must be both float32 or float64')
    pair_indices = np.ascontiguousarray(pair_indices, dtype=np.intp)
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError('pair_indices must be of shape = (n_pairs, 2)')
    return _sumdist(code, np.ascontiguousarray(X), pair_indices)
