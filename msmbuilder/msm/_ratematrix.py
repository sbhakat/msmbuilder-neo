# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.
r"""Log-likelihood, gradient, Hessian and uncertainties for a continuous-time
reversible Markov model sampled at a regular interval.

For details on the parameterization of the `\theta` vector, refer to the
documentation in docs/ratematrix.rst.

Numba port of the former Cython modules _ratematrix.pyx,
_ratematrix_support.pyx and _ratematrix_priors.pyx. The public functions keep
their signatures, including the optional ``A``/``out`` arguments that are
filled in place. Element-wise loops are Numba kernels; the BLAS calls of the
Cython version are NumPy matrix products in the same order.
"""
import math

import numpy as np
import scipy.linalg
from numba import njit

_LOG_DBL_MIN = math.log(np.finfo(np.float64).tiny)
_LOG_DBL_MAX = math.log(np.finfo(np.float64).max)

__all__ = ['build_ratemat', 'dK_dtheta_ij', 'dK_dtheta_u', 'loglikelihood',
           'hessian', 'sigma_K', 'sigma_pi', 'sigma_eigenvalues',
           'sigma_timescales', 'eig_K', 'dw_du', 'ldirichlet_softmax',
           'lexponential']


# --- index helpers (from triu_utils.pyx) --------------------------------------

@njit(cache=True, nogil=True)
def _ij_to_k(i, j, n):
    if j > i:
        return (n * (n - 1) // 2) - (n - i) * ((n - i) - 1) // 2 + j - i - 1
    return (n * (n - 1) // 2) - (n - j) * ((n - j) - 1) // 2 + i - j - 1


@njit(cache=True, nogil=True)
def _k_to_ij(k, n):
    i = n - 2 - int(math.sqrt(-8.0 * k + 4.0 * n * (n - 1) - 7.0) / 2.0 - 0.5)
    j = k + i + 1 - n * (n - 1) // 2 + (n - i) * (n - i - 1) // 2
    return i, j


def _as_double(x):
    return np.ascontiguousarray(x, dtype=np.float64)


def _check_out(out, shape, name='out'):
    if not isinstance(out, np.ndarray) or out.dtype != np.float64:
        raise TypeError('%s must be a float64 numpy array' % name)
    if out.shape != shape:
        raise ValueError('%s must have shape %s, got %s' % (name, shape, out.shape))
    return out


# --- rate matrix and its derivatives -------------------------------------------

@njit(cache=True, nogil=True)
def _build_ratemat(theta, n, out, build_S):
    n_triu = n * (n - 1) // 2
    pi = np.empty(n)
    for i in range(n):
        pi[i] = math.exp(theta[n_triu + i])
    for u in range(n_triu):
        i, j = _k_to_ij(u, n)
        s_ij = theta[u]
        K_ij = s_ij * math.sqrt(pi[j] / pi[i])
        K_ji = s_ij * math.sqrt(pi[i] / pi[j])
        if build_S:
            out[i, j] = s_ij
            out[j, i] = s_ij
        else:
            out[i, j] = K_ij
            out[j, i] = K_ji
        out[i, i] -= K_ij
        out[j, j] -= K_ji


def build_ratemat(theta, n, out, which='K'):
    r"""build_ratemat(theta, n, out, which='K')

    Build the reversible rate matrix K or symmetric rate matrix, S,
    from the free parameters, `\theta`

    Parameters
    ----------
    theta : array
        The free parameters, `\theta`. These values are the linearized elements
        of the upper triangular portion of the symmetric rate matrix, S,
        followed by the log equilibrium weights.
    n : int
        Dimension of the rate matrix, K, (number of states)
    out : [output], array shape=(n, n)
        On exit, out contains the matrix K or S. It must start as zeros:
        diagonal entries are accumulated into it.
    which : {'S', 'K'}
        Whether to build the matrix S or the matrix K
    """
    if isinstance(which, bytes):
        which = which.decode('ascii')
    n = int(n)
    _build_ratemat(_as_double(theta), n, _check_out(out, (n, n)), which == 'S')
    return 0


@njit(cache=True, nogil=True)
def _dK_dtheta_ij(theta, n, u, A, has_A, out, has_out):
    n_triu = n * (n - 1) // 2
    total = 0.0
    if has_out:
        for a in range(n):
            for b in range(n):
                out[a, b] = 0.0
    if u < n_triu:
        # perturbation of the symmetric rate matrix element (i, j)
        i, j = _k_to_ij(u, n)
        pi_i = math.exp(theta[n_triu + i])
        pi_j = math.exp(theta[n_triu + j])
        dK_ij = math.sqrt(pi_j / pi_i)
        dK_ji = math.sqrt(pi_i / pi_j)
        if has_A:
            total = (A[i, j] * dK_ij + A[j, i] * dK_ji
                     - A[i, i] * dK_ij - A[j, j] * dK_ji)
        if has_out:
            out[i, j] = dK_ij
            out[j, i] = dK_ji
            out[i, i] -= dK_ij
            out[j, j] -= dK_ji
    else:
        # perturbation of the equilibrium distribution, element i
        i = u - n_triu
        pi_i = math.exp(theta[n_triu + i])
        for j in range(n):
            if j == i:
                continue
            k = _ij_to_k(i, j, n)
            s_ij = theta[k]
            pi_j = math.exp(theta[n_triu + j])
            dK_ij = -0.5 * s_ij * math.sqrt(pi_j / pi_i)
            dK_ji = 0.5 * s_ij * math.sqrt(pi_i / pi_j)
            if has_A:
                total += (A[i, j] * dK_ij + A[j, i] * dK_ji
                          - A[i, i] * dK_ij - A[j, j] * dK_ji)
            if has_out:
                out[i, j] = dK_ij
                out[j, i] = dK_ji
                out[i, i] -= dK_ij
                out[j, j] -= dK_ji
    return total




def dK_dtheta_ij(theta, n, u, A=None, out=None):
    r"""dK_dtheta_ij(theta, n, u, A=None, out=None)

    Compute :math:`dK_ij / dtheta_u` over all `i`, `j` for fixed `u`.

    Along with `dK_dtheta_u`, this function computes a slice of the 3-index
    tensor :math:`dK_ij / dtheta_u`, the derivative of the rate matrix `K`
    with respect to the free parameters,`\theta`. This function computes a 2D
    slice of this tensor over all (i,j) for a fixed `u`. Using the argument
    `A`, it can also return the sum of the element-wise product of that slice
    with a given matrix directly, without building the slice.

    Parameters
    ----------
    theta : array
        The free parameters, `\theta`.
    n : int
        Dimension of the rate matrix, K, (number of states)
    u : int
        The index, `0 <= u < len(theta)` of the element in `theta` to
        construct the derivative of the rate matrix, `K` with respect to.
    A : array of shape=(n, n), optional
        If not None, an arbitrary (n, n) matrix to be multiplied element-wise
        with the derivative of the rate matrix, dKu.
    out : [output], optional array of shape=(n, n)
        If not None, out will contain the matrix dKu on exit.

    Returns
    -------
    s : double
        The sum of the element-wise product of dK/du and A, if A is not None.
    """
    n = int(n)
    has_A, has_out = A is not None, out is not None
    A_arr = _as_double(A) if has_A else np.empty((0, 0))
    out_arr = _check_out(out, (n, n)) if has_out else np.empty((0, 0))
    return _dK_dtheta_ij(_as_double(theta), n, int(u), A_arr, has_A, out_arr, has_out)


@njit(cache=True, nogil=True)
def _axpy(alpha, x, y):
    for k in range(y.shape[0]):
        y[k] += alpha * x[k]


@njit(cache=True, nogil=True)
def _dK_dtheta_u(theta, n, i, j, out, A, out2, compute_out2):
    n_triu = n * (n - 1) // 2
    pi_i = math.exp(theta[n_triu + i])
    if i != j:
        u = _ij_to_k(i, j, n)
        pi_j = math.exp(theta[n_triu + j])
        s_ij = theta[u]
        sq = math.sqrt(pi_j / pi_i)
        out[u] += sq
        if compute_out2:
            _axpy(sq, A[u, :], out2)
        if s_ij != 0:
            val = 0.5 * s_ij * sq
            out[n_triu + i] -= val
            out[n_triu + j] += val
            if compute_out2:
                _axpy(-val, A[n_triu + i, :], out2)
                _axpy(val, A[n_triu + j, :], out2)
    else:
        for jj in range(n):
            if jj == i:
                continue
            u = _ij_to_k(i, jj, n)
            s_ij = theta[u]
            pi_jj = math.exp(theta[n_triu + jj])
            sq = math.sqrt(pi_jj / pi_i)
            out[u] -= sq
            if compute_out2:
                _axpy(-sq, A[u, :], out2)
            if s_ij != 0:
                val = 0.5 * s_ij * sq
                out[n_triu + i] += val
                out[n_triu + jj] -= val
                if compute_out2:
                    _axpy(val, A[n_triu + i, :], out2)
                    _axpy(-val, A[n_triu + jj, :], out2)


def dK_dtheta_u(theta, n, i, j, out, A=None, out2=None):
    r"""dK_dtheta_u(theta, n, i, j, out, A=None, out2=None)

    Compute :math:`dK_ij / dtheta_u` over all `u` for fixed (`i`, `j`).

    Along with `dK_dtheta_ij`, this function computes a slice of the 3-index
    tensor :math:`dK_ij / dtheta_u`: here the gradient vector over `u` for a
    fixed `i`, `j`. It can also compute the product of that vector with a
    matrix `A` directly.

    Parameters
    ----------
    theta : array
        The free parameters, `\theta`.
    n : int
        Dimension of the rate matrix, K, (number of states)
    i, j : int
        The fixed indices of the rate matrix, `K`.
    out : [output], array of shape=(len(theta),)
        The vector dK[i,j]/dtheta[u] for all u is *added* to this array.
    A : array of shape(len(theta), len(theta)), optional
        If not None, an arbitrary matrix
    out2: [output], array of shape=(len(theta),)
        If not None (and A is given), ``np.dot(dK[i,j]/dtheta, A)`` is
        *added* to this array.
    """
    theta = _as_double(theta)
    size = theta.shape[0]
    out = _check_out(out, (size,))
    compute_out2 = A is not None and out2 is not None
    A_arr = np.asarray(A, dtype=np.float64) if compute_out2 else np.empty((0, 0))
    out2_arr = _check_out(out2, (size,), 'out2') if compute_out2 else np.empty(0)
    _dK_dtheta_u(theta, int(n), int(i), int(j), out, A_arr, out2_arr, compute_out2)
    return 0


# --- eigendecomposition and transition-matrix derivative -----------------------

@njit(cache=True, nogil=True)
def _scale_eigvecs_S(VS, pi):
    """Left (U) and right (V) eigenvectors of K from those of S:
    V[i, j] = sqrt(pi[j]/pi[i]) VS[i, j], U[i, j] = sqrt(pi[i]/pi[j]) VS[i, j],
    then each column of V normalized to unit length (U scaled inversely)."""
    n = VS.shape[0]
    U = np.empty((n, n))
    V = np.empty((n, n))
    for j in range(n):
        for i in range(n):
            V[i, j] = math.sqrt(pi[j] / pi[i]) * VS[i, j]
            U[i, j] = math.sqrt(pi[i] / pi[j]) * VS[i, j]
    for i in range(n):
        norm = 0.0
        for j in range(n):
            norm += V[j, i] * V[j, i]
        norm = math.sqrt(norm)
        for j in range(n):
            V[j, i] /= norm
            U[j, i] *= norm
    return U, V


def _eig_S(S, pi):
    # check_finite=False: callers have already checked S; same LAPACK driver.
    w, VS = scipy.linalg.eigh(S, check_finite=False)
    U, V = _scale_eigvecs_S(np.ascontiguousarray(VS), pi)
    return np.ascontiguousarray(w), U, V


def eig_K(A, n, pi=None, which='K'):
    r"""eig_K(A, n, pi=None, which='K')

    Diagonalize the rate matrix, K, from either the matrix K or the symmetric
    rate matrix, S.

    If which == 'K', the first argument should be the rate matrix, K, and `pi`
    is ignored. If which == 'S', the first argument should be the symmetric
    rate matrix, S, and pi should contain the equilibrium distribution.
    Whichever is supplied, the return value is the eigendecomposition of `K`.
    Using S is faster and more numerically stable.

    Returns
    -------
    w : array
        The eigenvalues of K
    U : array, size=(n,n)
        The left eigenvectors of K
    V : array, size=(n,n)
        The right eigenvectors of K
    """
    A = _as_double(A)
    n = int(n)
    if which == 'S':
        return _eig_S(A, _as_double(pi))
    else:
        w_, U_, V_ = scipy.linalg.eig(A, left=True, right=True)
        w = np.ascontiguousarray(np.real(w_))
        U = np.real(U_)
        V = np.real(V_)
        norms = np.sum(U * V, axis=0)
        U = U / norms[np.newaxis, :]
    return w, np.ascontiguousarray(U), np.ascontiguousarray(V)


@njit(cache=True, nogil=True)
def _exprel(x):
    r"""(exp(x) - 1) / x, accurate for small x."""
    cut = 0.002
    if x < _LOG_DBL_MIN:
        return -1.0 / x
    if x < -cut:
        return (math.exp(x) - 1.0) / x
    if x < cut:
        return 1.0 + 0.5 * x * (1.0 + x / 3.0 * (1.0 + 0.25 * x * (1.0 + 0.2 * x)))
    if x < _LOG_DBL_MAX:
        return (math.exp(x) - 1.0) / x
    return np.nan


@njit(cache=True, nogil=True)
def _hadamard_X(w, expwt, t, A):
    r"""A <- A \circ X, where X_ij = (e^{t w_i} - e^{t w_j}) / (w_i - w_j)
    off the diagonal and t e^{t w_i} on it."""
    n = w.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j:
                A[i, j] *= _exprel(t * (w[i] - w[j])) * t * expwt[j]
            else:
                A[i, j] *= t * expwt[i]


@njit(cache=True, nogil=True)
def _transmat(expwt, U, V):
    n = expwt.shape[0]
    temp = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            temp[i, j] = V[i, j] * expwt[j]
    return np.dot(temp, np.ascontiguousarray(U.T))


@njit(cache=True, nogil=True)
def _loglik_and_grad(theta, counts, t, w, U, V):
    """Everything after the eigendecomposition: transition matrix T (rows
    clipped at 1e-20 and renormalized), dL/dK propagated to dT, the gradient
    with respect to theta and the log-likelihood."""
    n = counts.shape[0]
    expwt = np.empty(n)
    for i in range(n):
        expwt[i] = math.exp(w[i] * t)
    T = _transmat(expwt, U, V)
    for i in range(n):
        rowsum = 0.0
        for j in range(n):
            if T[i, j] <= 0.0:
                T[i, j] = 1e-20
            rowsum += T[i, j]
        for j in range(n):
            T[i, j] = T[i, j] / rowsum
    dLdK = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # NOTE: tests T[j, j] (not T[i, j]), as in the original code.
            if counts[i, j] > 0 and T[j, j] > 0:
                dLdK[i, j] = counts[i, j] / T[i, j]
    temp = np.dot(np.dot(np.ascontiguousarray(V.T), dLdK), U)
    _hadamard_X(w, expwt, t, temp)
    dT = np.dot(np.dot(U, temp), np.ascontiguousarray(V.T))

    size = theta.shape[0]
    grad = np.zeros(size)
    no_out = np.empty((0, 0))
    for u in range(size):
        grad[u] = _dK_dtheta_ij(theta, n, u, dT, True, no_out, False)
    logl = 0.0
    for i in range(n):
        for j in range(n):
            if counts[i, j] > 0:
                logl += counts[i, j] * math.log(T[i, j])
    return logl, grad


def loglikelihood(theta, counts, t=1):
    r"""loglikelihood(theta, counts, t=1)

    Log likelihood and gradient of the log likelihood of a continuous-time
    Markov model.

    Parameters
    ----------
    theta : array of shape = (n*(n-1)//2 + n)
        The free parameters of the model: the linearized upper triangle of the
        symmetric rate matrix, S, followed by the log of the equilibrium
        distribution.
    counts : array of shape = (n, n)
        The matrix of observed transition counts.
    t : double
        The lag time.

    Returns
    -------
    logl : double
        The log likelihood of the parameters.
    grad : array of shape = (n*(n-1)//2 + n)
        The gradient of the log-likelihood with respect to `\theta`
    """
    theta = _as_double(theta)
    counts = _as_double(counts)
    n = counts.shape[0]
    n_triu = n * (n - 1) // 2
    if counts.ndim != 2 or counts.shape[1] != n:
        raise ValueError('counts must be n x n')
    if theta.shape[0] != n_triu + n:
        raise ValueError('theta must have shape n*(n+1)/2+n')
    t = float(t)

    S = np.zeros((n, n))
    _build_ratemat(theta, n, S, True)
    if not np.all(np.isfinite(S)):
        # these parameters don't seem good; tell the optimizer to steer clear
        return np.nan, np.zeros(theta.shape[0])
    w, U, V = _eig_S(S, np.exp(theta[n_triu:]))
    return _loglik_and_grad(theta, counts, t, w, U, V)


@njit(cache=True, nogil=True)
def _hessian_row(theta, n, inds, uu, Au, hessian):
    no_out = np.empty((0, 0))
    for vv in range(uu, inds.shape[0]):
        v = inds[vv]
        h = _dK_dtheta_ij(theta, n, v, Au, True, no_out, False)
        hessian[uu, vv] = h
        hessian[vv, uu] = h


def hessian(theta, counts, t=1, inds=None):
    r"""hessian(theta, counts, t=1, inds=None)

    Estimate of the hessian of the log-likelihood with respect to \theta.

    Parameters
    ----------
    theta : array of shape = (n*(n-1)//2 + n)
        The free parameters of the model at the MLE.
    counts : array of shape = (n, n)
        The matrix of observed transition counts.
    t : double
        The lag time.
    inds : array, optional
        If supplied, only compute a block of the Hessian at the specified
        indices.

    Returns
    -------
    m : array, shape=(len(theta), len(theta))
        An estimate of the hessian of the log-likelihood (or the requested
        block of it).
    """
    theta = _as_double(theta)
    counts = _as_double(counts)
    n = counts.shape[0]
    n_triu = n * (n - 1) // 2
    if counts.ndim != 2 or counts.shape[1] != n:
        raise ValueError('counts must be n x n')
    if theta.shape[0] != n_triu + n:
        raise ValueError('theta must have shape n*(n+1)/2+n')
    t = float(t)
    size = theta.shape[0]
    if inds is None:
        inds = np.arange(size, dtype=np.intp)
    inds = np.ascontiguousarray(inds, dtype=np.intp)
    H = np.zeros((len(inds), len(inds)))

    pi = np.exp(theta[n_triu:])
    S = np.zeros((n, n))
    _build_ratemat(theta, n, S, True)
    w, U, V = _eig_S(S, pi)
    expwt = np.exp(w * t)
    T = _transmat(expwt, U, V)
    Q = -counts.sum(axis=1)[:, np.newaxis] / T

    dKu = np.zeros((n, n))
    for uu in range(len(inds)):
        _dK_dtheta_ij(theta, n, int(inds[uu]), np.empty((0, 0)), False, dKu, True)
        # Gu = U.T dKu V;  dPu = V (Gu o X) U.T
        temp = (U.T @ dKu) @ V
        _hadamard_X(w, expwt, t, temp)
        temp = (V @ temp) @ U.T
        # Bu = V.T (dPu o Q) U;  Au = U (Bu o X) V.T
        temp = (V.T @ (temp * Q)) @ U
        _hadamard_X(w, expwt, t, temp)
        Au = np.ascontiguousarray((U @ temp) @ V.T)
        _hessian_row(theta, n, inds, uu, Au, H)
    return H


# --- uncertainties -------------------------------------------------------------

def _check_covar(covar_theta, theta, n):
    n_triu = n * (n - 1) // 2
    size = theta.shape[0]
    if size != n_triu + n:
        raise ValueError('theta must have shape n*(n+1)/2+n')
    covar_theta = np.asarray(covar_theta, dtype=np.float64)
    if covar_theta.shape != (size, size):
        raise ValueError('covar_theta must be `size` x `size`')
    return covar_theta


@njit(cache=True, nogil=True)
def _sigma_K(covar, theta, n):
    size = theta.shape[0]
    sigma = np.zeros((n, n))
    dKij = np.empty(size)
    temp = np.empty(size)
    for i in range(n):
        for j in range(n):
            dKij[:] = 0.0
            temp[:] = 0.0
            _dK_dtheta_u(theta, n, i, j, dKij, covar, temp, True)
            var = 0.0
            for k in range(size):
                var += dKij[k] * temp[k]
            sigma[i, j] = math.sqrt(var)
    return sigma


def sigma_K(covar_theta, theta, n):
    r"""sigma_K(covar_theta, theta, n)

    Estimate the asymptotic standard deviation (uncertainty in the rate
    matrix, `K`

    Parameters
    ----------
    covar_theta : array, shape=(len(theta), len(theta))
        Covariance matrix of \theta. This is estimated by the inverse hessian
        of the log likelihood function.
    theta : array of shape = (n*(n-1)//2 + n)
        The free parameters of the model at the MLE.
    n : int
        The size of `counts`

    Returns
    -------
    sigma_K : array, shape=(n, n)
        Estimate of the element-wise asymptotic standard deviation of the
        rate matrix, K.
    """
    theta = _as_double(theta)
    n = int(n)
    covar = np.ascontiguousarray(_check_covar(covar_theta, theta, n))
    return _sigma_K(covar, theta, n)


def sigma_pi(covar_theta, theta, n):
    r"""sigma_pi(covar_theta, theta, n)

    Estimate the asymptotic standard deviation (uncertainty) in the
    stationary distribution, `\pi`.

    Parameters
    ----------
    covar_theta : array, shape=(len(theta), len(theta))
        Covariance matrix of \theta.
    theta : array of shape = (n*(n-1)//2 + n)
        The free parameters of the model at the MLE.
    n : int
        The size of `counts`

    Returns
    -------
    sigma_pi : array, shape=(n,)
        Estimate of the element-wise asymptotic standard deviation of the
        stationary distribution, \pi.
    """
    theta = _as_double(theta)
    n = int(n)
    covar = _check_covar(covar_theta, theta, n)
    n_triu = n * (n - 1) // 2
    pi = np.exp(theta[n_triu:])
    z = pi.sum()
    z_m2 = 1.0 / (z * z)
    C_block = np.ascontiguousarray(covar[n_triu:, n_triu:])
    # Jacobian d pi / d theta
    dpi = -z_m2 * np.outer(pi, pi)
    dpi[np.diag_indices(n)] = z_m2 * pi * (z - pi)
    return np.array([math.sqrt(dpi[i] @ (C_block @ dpi[i])) for i in range(n)])


def _dlambda_dtheta(theta, n):
    """Eigenvalues (sorted descending) and d lambda_i / d theta_u."""
    n_triu = n * (n - 1) // 2
    size = theta.shape[0]
    pi = np.exp(theta[n_triu:])
    S = np.zeros((n, n))
    _build_ratemat(theta, n, S, True)
    w, U, V = _eig_S(S, pi)
    order = np.argsort(w)[::-1]
    U = np.ascontiguousarray(U[:, order])
    V = np.ascontiguousarray(V[:, order])
    w = w[order]
    dlambda = np.zeros((n, size))
    dKu = np.zeros((n, n))
    for u in range(size):
        _dK_dtheta_ij(theta, n, u, np.empty((0, 0)), False, dKu, True)
        dlambda[:, u] = np.einsum('ij,ji->i', U.T, dKu @ V)
    return w, dlambda


def sigma_eigenvalues(covar_theta, theta, n):
    r"""sigma_eigenvalues(covar_theta, theta, n)

    Estimate the asymptotic standard deviation (uncertainty) in the
    eigenvalues of K, sorted in descending order.

    Parameters
    ----------
    covar_theta : array, shape=(len(theta), len(theta))
        Covariance matrix of \theta.
    theta : array of shape = (n*(n-1)//2 + n)
        The free parameters of the model at the MLE.
    n : int
        The size of `counts`

    Returns
    -------
    sigma_eigs : array, shape=(n,)
        Estimate of the element-wise asymptotic standard deviation of the
        eigenvalues of the rate matrix (in sorted order).
    """
    theta = _as_double(theta)
    n = int(n)
    covar = _check_covar(covar_theta, theta, n)
    _, dlambda = _dlambda_dtheta(theta, n)
    return np.array([math.sqrt(dlambda[i] @ (covar @ dlambda[i])) for i in range(n)])


def sigma_timescales(covar_theta, theta, n):
    r"""sigma_timescales(covar_theta, theta, n)

    Estimate the asymptotic standard deviation (uncertainty) in the
    implied timescales.

    Parameters
    ----------
    covar_theta : array, shape=(len(theta), len(theta))
        Covariance matrix of \theta.
    theta : array of shape = (n*(n-1)//2 + n)
        The free parameters of the model at the MLE.
    n : int
        The size of `counts`

    Returns
    -------
    sigma_t : array, shape=(n-1,)
        Estimate of the element-wise asymptotic standard deviation of the
        relaxation timescales of the model.
    """
    theta = _as_double(theta)
    n = int(n)
    covar = _check_covar(covar_theta, theta, n)
    w, dlambda = _dlambda_dtheta(theta, n)
    with np.errstate(divide='ignore', invalid='ignore'):
        dtau = dlambda / (w ** 2)[:, np.newaxis]
    return np.array([math.sqrt(dtau[i] @ (covar @ dtau[i])) for i in range(1, n)])


def dw_du(dKu, U, V, n, temp, out):
    r"""dw_du(dKu, U, V, n, temp, out)

    Derivative of the eigenvalues of K with respect to one free parameter:
    ``out[i] = U[:, i] . (dKu V[:, i])``. ``temp`` is unused workspace kept
    for compatibility.
    """
    n = int(n)
    dKu, U, V = _as_double(dKu), _as_double(U), _as_double(V)
    out[:] = np.einsum('ij,ji->i', U.T, dKu @ V)[:n]
    return 0


# --- priors (from _ratematrix_priors.pyx) --------------------------------------

def ldirichlet_softmax(theta, alpha, grad=None):
    r"""Log of the symmetric Dirichlet probability density, evaluated at a
    softmax-transformed point, with an extra Gaussian term that pins
    ``sum(theta)`` near zero.

    Parameters
    ----------
    theta : array, shape=(n,)
        Point at which to evaluate the log density. The Dirichlet variables
        are ``exp(theta) / sum(exp(theta))``.
    alpha : array, shape=(n,)
        Dirichlet concentration parameters.
    grad : array, shape=(n,), optional
        If not None, the gradient with respect to theta is *added* to it.

    Returns
    -------
    logp : float
    """
    theta = _as_double(theta)
    alpha = _as_double(alpha)
    if theta.shape[0] != alpha.shape[0]:
        raise ValueError('len(theta) != len(alpha)')
    if grad is not None and theta.shape[0] != grad.shape[0]:
        raise ValueError('len(theta) != len(grad)')
    exptheta = np.exp(theta)
    norm = exptheta.sum()
    sumtheta = theta.sum()
    sumalpha = alpha.sum()
    logpi = theta - math.log(norm)
    logp = float(alpha @ logpi) - sumtheta * sumtheta / 2
    if grad is not None:
        grad += -exptheta * (sumalpha / norm) + alpha - sumtheta
    return logp


def lexponential(theta, beta, grad=None):
    r"""Log density of independent exponential distributions, with scales
    ``beta``, evaluated at ``theta``.

    Parameters
    ----------
    theta : array, shape=(n,)
    beta : array, shape=(n,)
        Scale parameters of the exponential distributions.
    grad : array, shape=(n,), optional
        If not None, the gradient with respect to theta is *added* to it.

    Returns
    -------
    logp : float
    """
    theta = _as_double(theta)
    beta = _as_double(beta)
    if theta.shape[0] != beta.shape[0]:
        raise ValueError('len(theta) != len(beta)')
    if grad is not None and theta.shape[0] != grad.shape[0]:
        raise ValueError('len(theta) != len(grad)')
    logp = -float(np.sum(np.log(beta) + theta / beta))
    if grad is not None:
        grad -= 1 / beta
    return logp
