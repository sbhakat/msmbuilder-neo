# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.
"""Reversible maximum-likelihood transition matrix (Prinz et al. 2011).

Numba port of the former C implementation, transmat_mle_prinz.c. The update
order and arithmetic follow the C code exactly.
"""
import math

import numpy as np
from numba import njit

_FLT_MAX = float(np.finfo(np.float32).max)


@njit(cache=True, nogil=True)
def _prinz(C, tol, T, pi):
    """Returns the iteration count, or -1 (zero row) / -2 (numerical error)."""
    n = C.shape[0]
    X = np.empty((n, n))
    x_rs = np.empty(n)
    c_rs = np.empty(n)
    for i in range(n):
        for j in range(n):
            X[i, j] = C[i, j] + C[j, i]
    for i in range(n):
        x_rs[i] = 0.0
        c_rs[i] = 0.0
        for j in range(n):
            x_rs[i] += X[i, j]
            c_rs[i] += C[i, j]
        if x_rs[i] <= 0 or c_rs[i] <= 0:
            return -1

    logl = 0.0
    oldlogl = _FLT_MAX
    it = 0
    while abs(oldlogl - logl) >= tol:
        oldlogl = logl
        logl = 0.0
        # update the diagonal
        for i in range(n):
            tmp = X[i, i]
            denom = c_rs[i] - C[i, i]
            if denom > 0:
                X[i, i] = C[i, i] * (x_rs[i] - X[i, i]) / denom
            x_rs[i] = x_rs[i] + (X[i, i] - tmp)
            if X[i, i] > 0:
                logl += C[i, i] * math.log(X[i, i] / x_rs[i])
        # update the off-diagonal entries
        for i in range(n - 1):
            for j in range(i + 1, n):
                a = (c_rs[i] - C[i, j]) + (c_rs[j] - C[j, i])
                b = (c_rs[i] * (x_rs[j] - X[i, j])
                     + c_rs[j] * (x_rs[i] - X[i, j])
                     - (C[i, j] + C[j, i]) * (x_rs[i] + x_rs[j] - 2 * X[i, j]))
                c = -(C[i, j] + C[j, i]) * (x_rs[i] - X[i, j]) * (x_rs[j] - X[i, j])
                if c > 0:
                    return -2  # should never happen
                if a == 0:
                    v = X[j, i]
                else:
                    v = (-b + math.sqrt((b * b) - (4 * a * c))) / (2 * a)
                x_rs[i] = x_rs[i] + (v - X[i, j])
                x_rs[j] = x_rs[j] + (v - X[j, i])
                X[i, j] = v
                X[j, i] = v
                if X[i, j] > 0:
                    logl += (C[i, j] * math.log(X[i, j] / x_rs[i])
                             + C[j, i] * math.log(X[j, i] / x_rs[j]))
        if logl != logl:  # NaN
            return -2
        it += 1

    pi_sum = 0.0
    for i in range(n):
        pi_sum += x_rs[i]
        for j in range(n):
            T[i, j] = X[i, j] / x_rs[i]
    for i in range(n):
        pi[i] = x_rs[i] / pi_sum
    return it


def _transmat_mle_prinz(C, tol=1e-10):
    """Compute a maximum likelihood reversible transition matrix, given
    a set of directed transition counts.

    Algorithim 1 of Prinz et al.[1]

    Parameters
    ----------
    C : (input) 2d array of shape=(n_states, n_states)
        The directed transition counts, in C (row-major) order.
    tol : (input) float
        Convergence tolerance. The algorithm will iterate until the
        change in the log-likelihood is less than `tol`.

    Returns
    -------
    T : (output) pointer to output 2d array of shape=(n_states, n_states)
        Once the algorithim is completed, the resulting transition
        matrix will be written to `T`.
    populations : array, shape = (n_states_,)
        The equilibrium population (stationary left eigenvector) of T

     References
     ----------
     .. [1] Prinz, Jan-Hendrik, et al. "Markov models of molecular kinetics:
        Generation and validation." J Chem. Phys. 134.17 (2011): 174105.
    """
    C = np.ascontiguousarray(C, dtype=np.float64)
    n_states = len(C)
    if n_states == 0:
        return np.zeros((0, 0)), np.zeros(0)
    if C.ndim != 2 or C.shape[1] != n_states:
        raise ValueError('C must be square')

    T = np.zeros((n_states, n_states))
    pi = np.zeros(n_states)
    n_iter = _prinz(C, float(tol), T, pi)
    if n_iter < 0:
        # diagnose the error
        msg = ' Error code=%d' % n_iter
        if np.any(np.less(C, 0)):
            msg = 'Domain error. C must be positive.' + msg
        if np.any(np.sum(C, axis=1) == 0):
            msg = 'Row-sums of C must be positive.' + msg
        raise ValueError(msg)
    return T, pi
