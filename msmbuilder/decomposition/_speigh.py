"""Sparse generalized eigenvalue problem used by SparseTICA.

Numba port of the former _speigh.pyx. The ADMM solver and the projection
step run as Numba kernels; the remaining linear algebra is NumPy/SciPy, in the
same order as before.
"""
import math

import numpy as np
import scipy.linalg
import scipy.special
from numba import njit

SPEIGH_TAU_MIN = 0.1
ADMM_RHO_INITIAL = 16.0
ADMM_PENALTY_MU_FACTOR = 10.0
ADMM_PENALTY_TAU_FACTOR = 2.0
ADMM_PENALTY_RHO_MAX = 2.0**10
ADMM_PENALTY_RHO_MIN = 2.0**-10


def scdeflate(A, x):
    """Schur complement matrix deflation

    Eliminate the influence of a psuedo-eigenvector of A using the Schur complement
    deflation technique from [1]::

        A_new = A - \frac{A x x^T A}{x^T A x}

    Parameters
    ----------
    A : np.ndarray, shape=(N, N)
        A matrix
    x : np.ndarray, shape=(N, )
        A vector, ideally one that is "close to" an eigenvector of A

    Returns
    -------
    A_new : np.ndarray, shape=(N, N)
        A new matrix, determined from A by eliminating the influence of x

    References
    ----------
    ... [1] Mackey, Lester. "Deflation Methods for Sparse PCA." NIPS. Vol.
        21. 2008.
    """
    return A - np.outer(np.dot(A, x), np.dot(x, A)) / np.dot(np.dot(x, A), x)


def speigh(A, B, rho, eps=1e-6, tol=1e-8, maxiter=100, verbose=False, method=1):
    r"""Find a sparse approximate generalized eigenpair.

    The generalized eigenvalue equation, :math:`Av = lambda Bv`,
    can be expressed as a variational optimization ::
    :math:`max_{x} x^T A x  s.t. x^T B x = 1`. We can search for sparse
    approximate eigenvectors then by adding a penalty to the optimization.
    This function solves an approximation to::

    max_{x}   x^T A x - \rho ||x||_0

        s.t.      x^T B x <= 1

    Where `||x||_0` is the number of nonzero elements in `x`. Note that
    because of the ||x||_0 term, that problem is NP-hard. Here, we replace
    the ||x||_0 term with

    rho * \sum_i^N \frac{\log(1 + |x_i|/eps)}{1 + 1/eps}

    which converges to ||x||_0 in the limit that eps goes to zero.

    Parameters
    ----------
    A : np.ndarray, shape=(N, N)
        A is symmetric matrix, the left-hand-side of the eigenvalue equation.
    B : np.ndarray, shape=(N, N)
        B is a positive semidefinite matrix, the right-hand-side of the
        eigenvalue equation.
    rho : float
        Regularization strength. Larger values for rho will lead to more sparse
        solutions.
    eps : float
        Small number, used in the approximation to the L0. Smaller is better
        (closer to L0), but trickier from a numerical standpoint and can lead
        to the solver complaining when it gets too small.
    tol : float
        Convergence criteria for the eigensolver.
    maxiter : int
        Maximum number of iterations.
    method : int
        1: Default ADMM solver.
        2. CVXOPT interior point solver for the QCQP.

    Returns
    -------
    u : float
        The approximate eigenvalue.
    x : np.ndarray, shape=(N,)
        The sparse approximate eigenvector

    References
    ----------
    ... [1] Sriperumbudur, B. K., D. A. Torres, and G. R. G. Lanckriet.
        "A majorization-minimization approach to the sparse generalized
        eigenvalue problem." Machine learning 85.1-2 (2011): 3-39.
    ... [2] McGibbon, R. T. and V. S. Pande, in preparation (2015)
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    rho, eps, tol = float(rho), float(eps), float(tol)
    maxiter, method = int(maxiter), int(method)
    N = len(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError('A must be a square matrix')
    if B.shape[0] != B.shape[1]:
        raise ValueError('B must be square matrix')
    if A.shape[0] != B.shape[0]:
        raise ValueError('Wrong B dimensions (%d, %d) should be (%d, %d)' % (
            B.shape[0], B.shape[1], A.shape[0], A.shape[1]))

    rho_e = rho / scipy.special.log1p(1/eps)
    tau = SPEIGH_TAU_MIN + max(0, -np.min(scipy.linalg.eigvalsh(A)))
    f, old_f = np.inf, np.inf

    x = np.ascontiguousarray(scipy.linalg.eigh(A, B, subset_by_index=[N-1, N-1])[1][:, 0])
    if method == 1:
        B_eigvals, B_eigvecs = map(np.ascontiguousarray, scipy.linalg.eigh(B))

    for i in range(maxiter):
        old_f = f
        Ax = np.dot(A, x)
        f = _objective(Ax, x, rho_e, eps)
        if verbose:
            print("QCQP objective=%.5f" % f)
        if abs(old_f - f) < tol:
            break
        b = Ax / tau + x
        w = rho_e / (2 * tau * (np.abs(x) + eps))
        if method == 1:
            f2 = solve_admm(b, w, B_eigvals, B_eigvecs, x, tol=tol, maxiter=maxiter,
                            verbose=verbose)
        elif method == 2:
            x, f2 = solve_cvxpy(b, w, B)
        else:
            raise ValueError('Unknown method')
        if verbose:
            print("ADMM objective=%.5f" % f2)

    if verbose:
        print('Optimized vector (before renormalization)')
        print(np.asarray(x))

    mask = (np.abs(x) > 0)
    grid = np.ix_(mask, mask)
    Ak, Bk = A[grid], B[grid]  # form the submatrices
    if len(Ak) == 0:
        u, v = 0, np.zeros(N)
    elif len(Ak) == 1:
        v = np.zeros(N)
        v[mask] = 1.0 / np.sqrt(Bk[0, 0])
        u = Ak[0, 0] / Bk[0, 0]
    else:
        gevals, gevecs = scipy.linalg.eigh(
            Ak, Bk, subset_by_index=[Ak.shape[0]-1, Ak.shape[0]-1])
        u = gevals[0]
        v = np.zeros(N)
        v[mask] = gevecs[:, 0]
        v *= np.sign(np.sum(v))
        v[v == -0] = 0
    return u, v


@njit(cache=True, nogil=True)
def _objective(Ax, x, rho_e, eps):
    f = 0.0
    for j in range(x.shape[0]):
        f += Ax[j] * x[j]
    for j in range(x.shape[0]):
        f -= rho_e * math.log(abs(x[j]) + eps)
    return f


def solve_cvxpy(b, w, B):
    """Solve a convex optimization problem using cvxpy

    Minimize    1/2 ||x-b||^2 + ||D(w)x||_1
    subject to  x^T B x <= 1

    Parameters
    ----------
    b : array, shape=(n,)
    w : array, shape=(n,)
    B : array, shape=(n,n)

    Returns
    -------
    xf : array, shape=(n,)
        The optimal value of x
    fun : float
        The value of the objective function at ``xf``
    """
    import cvxpy as cp
    n = len(b)
    x = cp.Variable(n)
    #objective = cp.Minimize(
    #    0.5 * cp.norm2(x-b)**2 + lambd*cp.norm1(cp.diag(w) * x))

    # more accurate this way, expanding out the norm
    objective = cp.Minimize(
        0.5*cp.norm2(x)**2 - x.T*np.asarray(b) + cp.norm1(cp.diag(np.asarray(w)) * x))

    constraints = [cp.quad_form(x, np.asarray(B)) <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status != 'optimal':
        print(problem, problem.status)
        raise ValueError('No solution found')

    x = np.asarray(x.value)[:,0]
    x[np.abs(x) < 1e-6] = 0
    fun = 0.5*np.dot(x-b, x-b) + np.sum(np.abs(np.multiply(w, x)))
    return x, fun



def solve_admm(b, w, B_eigvals, B_eigvecs, x, tol=1e-6, maxiter=100, verbose=0):
    """Solve a particular convex optimization problem with ADMM

    Minimize    1/2 ||x-b||^2 + ||D(w)x||_1
    subject to  x^T B x <= 1

    Parameters
    ----------
    b : array, shape=(n,)
    w : array, shape=(n,)
    B_eigvals : array, shape=(n,)
    B_eigvecs : array, shape=(n,n)
    x : array, shape=(n,)
    tol : float, default=1e-6
    maxiter : int, default=100
    """
    b = np.ascontiguousarray(b, dtype=np.float64)
    w = np.ascontiguousarray(w, dtype=np.float64)
    B_eigvals = np.ascontiguousarray(B_eigvals, dtype=np.float64)
    B_eigvecs = np.ascontiguousarray(B_eigvecs, dtype=np.float64)
    N = len(b)
    if not (len(b) == len(w) == len(x) == len(B_eigvals) == B_eigvecs.shape[0]
            == B_eigvecs.shape[1]):
        raise ValueError('Incompatible matrix dimensions')
    if verbose > 1:
        # keep the per-iteration residual printout available in pure Python
        return _solve_admm_verbose(b, w, B_eigvals, B_eigvecs, x, tol, maxiter)
    _admm(b, w, B_eigvals, B_eigvecs, x, float(tol), int(maxiter))
    return 0.5*np.dot(x, x) - np.dot(x, b) + 0.5*np.dot(b, b) + np.sum(np.abs(np.multiply(w, x)))


@njit(cache=True, nogil=True)
def _soft_thresh(k, a, out):
    for i in range(k.shape[0]):
        if a[i] > k[i]:
            out[i] = a[i] - k[i]
        elif a[i] < -k[i]:
            out[i] = a[i] + k[i]
        else:
            out[i] = 0.0


@njit(cache=True, nogil=True)
def _project(a, w, V, out):
    N = a.shape[0]
    c = np.dot(np.ascontiguousarray(V.T), a)   # c = V.T.dot(a)
    c2w = np.empty(N)
    c2w_sum = 0.0
    for j in range(N):
        c2w[j] = c[j] * c[j] * w[j]
        c2w_sum += c2w[j]
    if c2w_sum < 1:
        for j in range(N):
            out[j] = a[j]
        return
    mu = 0.0
    for _ in range(100):
        G = -1.0
        Gp = 0.0
        for j in range(N):
            muw1 = mu * w[j] + 1
            G += c2w[j] / (muw1 * muw1)
            Gp -= 2 * c2w[j] * w[j] / (muw1 * muw1 * muw1)
        delta = -G / Gp
        mu = mu + delta
        if abs(delta) < 1e-12:
            break
    for j in range(N):
        c2w[j] = c[j] / (mu * w[j] + 1)
    res = np.dot(V, c2w)
    for j in range(N):
        out[j] = res[j]


@njit(cache=True, nogil=True)
def _admm_step(b, w, absw, B_eigvals, B_eigvecs, x, z, z_old, u, rho):
    N = b.shape[0]
    b_rho_zu = np.empty(N)
    v = np.empty(N)
    for j in range(N):
        z_old[j] = z[j]
    for j in range(N):
        b_rho_zu[j] = b[j] + rho * (z[j] - u[j])
    _soft_thresh(absw, b_rho_zu, x)
    for j in range(N):
        x[j] = x[j] / (rho + 1.0)
    for j in range(N):
        v[j] = x[j] + u[j]
    _project(v, B_eigvals, B_eigvecs, z)
    r_norm2 = 0.0
    s_norm2 = 0.0
    for j in range(N):
        r = x[j] - z[j]               # primal residual
        s = rho * (z[j] - z_old[j])   # dual residual
        u[j] = u[j] + r
        r_norm2 += r * r
        s_norm2 += s * s
    return r_norm2, s_norm2


@njit(cache=True, nogil=True)
def _admm_update_rho(rho, r_norm2, s_norm2, u):
    if r_norm2 > ADMM_PENALTY_MU_FACTOR**2 * s_norm2 and rho < ADMM_PENALTY_RHO_MAX:
        rho *= ADMM_PENALTY_TAU_FACTOR
        for j in range(u.shape[0]):
            u[j] /= ADMM_PENALTY_TAU_FACTOR
    if s_norm2 > ADMM_PENALTY_MU_FACTOR**2 * r_norm2 and rho > ADMM_PENALTY_RHO_MIN:
        rho /= ADMM_PENALTY_TAU_FACTOR
        for j in range(u.shape[0]):
            u[j] *= ADMM_PENALTY_TAU_FACTOR
    return rho


@njit(cache=True, nogil=True)
def _admm(b, w, B_eigvals, B_eigvecs, x, tol, maxiter):
    N = b.shape[0]
    rho = ADMM_RHO_INITIAL
    z = x.copy()
    z_old = x.copy()
    u = np.zeros(N)
    absw = np.abs(w)
    for _ in range(maxiter):
        r_norm2, s_norm2 = _admm_step(b, w, absw, B_eigvals, B_eigvecs, x, z, z_old, u, rho)
        if r_norm2 < N * tol * tol and s_norm2 < N * tol * tol:
            break
        rho = _admm_update_rho(rho, r_norm2, s_norm2, u)


def _solve_admm_verbose(b, w, B_eigvals, B_eigvecs, x, tol, maxiter):
    N = len(b)
    rho = ADMM_RHO_INITIAL
    z, z_old, u = x.copy(), x.copy(), np.zeros(N)
    absw = np.abs(w)
    for _ in range(int(maxiter)):
        r_norm2, s_norm2 = _admm_step(b, w, absw, B_eigvals, B_eigvecs, x, z, z_old, u, rho)
        print(' rho', rho, ' residuals ', math.sqrt(r_norm2), math.sqrt(s_norm2))
        if r_norm2 < N * tol * tol and s_norm2 < N * tol * tol:
            break
        rho = _admm_update_rho(rho, r_norm2, s_norm2, u)
    return 0.5*np.dot(x, x) - np.dot(x, b) + 0.5*np.dot(b, b) + np.sum(np.abs(np.multiply(w, x)))


def project(a, w, V, out):
    """ Compute the projection of a vector a onto an ellipsoid

        Minimize_x ||x-a||^2
        subject to x^TBx <= 1

    where (w, V) are the eigenvalues and eigenvectors of the positive definite
    matrix B.

    Parameters
    ----------
    a : array, shape=(n,)
        The vector to project
    w : array, shape=(n,)
        eigenvalues of B
    V : array, shape=(n,n)
        eigenvectors of B
    out : array, shape=(n,)
        On exit, the results will be written here
    """
    a = np.ascontiguousarray(a, dtype=np.float64)
    w = np.ascontiguousarray(w, dtype=np.float64)
    V = np.ascontiguousarray(V, dtype=np.float64)
    if not (len(a) == len(w) == V.shape[0] == V.shape[1]):
        raise ValueError('Incompatible matrix dimensions')
    _project(a, w, V, out)
