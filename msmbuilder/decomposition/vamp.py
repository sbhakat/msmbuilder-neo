"""Variational Approach for Markov Processes (VAMP).

VAMP finds the linear combinations of input features that best approximate
the dominant singular functions of the Koopman (transfer) operator. Unlike
tICA it does not assume the dynamics are reversible or at equilibrium, and
its score -- the VAMP-r score -- can rank models and features by
cross-validation for both reversible and non-reversible processes.

The definitions follow Wu and Noe (2020) and match deeptime's
``deeptime.decomposition.VAMP`` (mean-free covariances without Bessel
correction, whitening with an eigenvalue cut-off ``epsilon``, and a constant
singular function contributing +1 to every score).
"""
import numpy as np
import scipy.linalg
from sklearn.base import TransformerMixin

from ..base import BaseEstimator
from ..utils import array2d, check_iter_of_sequences

__all__ = ['VAMP']

_SCORE_METHODS = ('VAMP1', 'VAMP2', 'VAMPE')


def _inv_sqrt_split(C, epsilon):
    """L such that L.T @ C @ L = I on the subspace where the eigenvalues of
    the symmetric matrix C exceed epsilon (columns sorted by eigenvalue,
    largest first)."""
    s, V = np.linalg.eigh(0.5 * (C + C.T))
    order = np.argsort(s)[::-1]
    s, V = s[order], V[:, order]
    keep = s > epsilon
    if not np.any(keep):
        raise ValueError('All eigenvalues of the covariance matrix are below '
                         'epsilon=%g; there is nothing to whiten.' % epsilon)
    return V[:, keep] / np.sqrt(s[keep])


def _sym_inv_sqrt(C, epsilon):
    """Symmetric inverse square root of C, discarding eigenvalues <= epsilon."""
    s, V = np.linalg.eigh(0.5 * (C + C.T))
    keep = s > epsilon
    return (V[:, keep] / np.sqrt(s[keep])) @ V[:, keep].T


class _Covariances:
    """Mean-free instantaneous and time-lagged covariances of a set of
    sequences, pooled over sequences and normalized by the number of
    (x_t, x_{t+lag}) pairs."""

    def __init__(self, sequences, lag_time):
        n_pairs = 0
        sum_0 = sum_t = None
        usable = []
        for X in sequences:
            X = np.asarray(array2d(X), dtype=np.float64)
            if len(X) <= lag_time:
                continue
            usable.append(X)
            if sum_0 is None:
                sum_0 = np.zeros(X.shape[1])
                sum_t = np.zeros(X.shape[1])
            sum_0 += X[:-lag_time].sum(axis=0)
            sum_t += X[lag_time:].sum(axis=0)
            n_pairs += len(X) - lag_time
        if n_pairs == 0:
            raise ValueError('No sequence is longer than lag_time=%d.' % lag_time)

        self.mean_0 = sum_0 / n_pairs
        self.mean_t = sum_t / n_pairs
        n_features = len(self.mean_0)
        c00 = np.zeros((n_features, n_features))
        c0t = np.zeros((n_features, n_features))
        ctt = np.zeros((n_features, n_features))
        # second pass with the pooled means removed (numerically safer than
        # E[xy] - E[x]E[y] when features have large means)
        for X in usable:
            X0 = X[:-lag_time] - self.mean_0
            Xt = X[lag_time:] - self.mean_t
            c00 += X0.T @ X0
            c0t += X0.T @ Xt
            ctt += Xt.T @ Xt
        self.cov_00 = c00 / n_pairs
        self.cov_0t = c0t / n_pairs
        self.cov_tt = ctt / n_pairs
        self.n_pairs = n_pairs
        self.n_sequences = len(usable)


class VAMP(BaseEstimator, TransformerMixin):
    r"""Variational Approach for Markov Processes (VAMP)

    Linear dimensionality reduction that finds the features' linear
    combinations which best approximate the dominant singular functions of
    the Koopman operator at a given lag time. Unlike tICA, VAMP does not
    assume reversible, equilibrium dynamics, and its VAMP-r score can be used
    to select models, features and lag times by cross-validation.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of singular components to keep (and to use when scoring).
        None keeps all components whose singular value is resolvable at the
        given ``epsilon``.
    lag_time : int, default=1
        Lag time, in frames, between the instantaneous and time-lagged data.
    epsilon : float, default=1e-6
        Eigenvalues of the covariance matrices at or below ``epsilon`` are
        discarded when whitening, which removes linearly dependent
        directions.
    scaling : {None, 'kinetic_map'}, default=None
        If 'kinetic_map', scale the projected coordinates by the singular
        values, so Euclidean distances in the projection approximate kinetic
        distances.
    score_method : {'VAMP1', 'VAMP2', 'VAMPE'}, default='VAMP2'
        Score returned by :meth:`score`. VAMP-r is the sum of the r-th powers
        of the singular values (plus 1 for the constant function); VAMP-E is
        the variational score of Wu and Noe based on the approximation error
        of the Koopman operator.

    Attributes
    ----------
    singular_values_ : array, shape (n_components,)
        Singular values of the estimated Koopman operator, in decreasing
        order, excluding the constant function (whose singular value is 1).
    left_singular_vectors_ : array, shape (n_features, n_components)
        Coefficients of the instantaneous (left) singular functions in terms
        of the mean-free input features. ``transform`` projects onto these.
    right_singular_vectors_ : array, shape (n_features, n_components)
        Coefficients of the time-lagged (right) singular functions.
    components_ : array, shape (n_components, n_features)
        ``left_singular_vectors_.T``, for consistency with scikit-learn.
    means_ : array, shape (n_features,)
        Mean of the instantaneous data, removed before projecting.
    means_lagged_ : array, shape (n_features,)
        Mean of the time-lagged data.
    cov_00_, cov_0t_, cov_tt_ : arrays, shape (n_features, n_features)
        Mean-free instantaneous, cross and time-lagged covariance matrices.
    score_ : float
        Score of the model on its own training data (see ``score_method``).
    timescales_ : array, shape (n_components,)
        ``-lag_time / log(singular_values_)``. These are implied timescales
        only if the dynamics are reversible; otherwise they are the decay
        times of the singular functions.
    n_observations_ : int
        Number of (x_t, x_{t+lag}) pairs used in the fit.
    n_sequences_ : int
        Number of sequences longer than ``lag_time`` used in the fit.

    References
    ----------
    .. [1] Wu, H. and Noe, F. "Variational approach for learning Markov
       processes from time series data." J. Nonlinear Sci. 30, 23-66 (2020).
    .. [2] Hoffmann, M. et al. "Deeptime: a Python library for machine
       learning dynamical models from time series data." Mach. Learn.: Sci.
       Technol. 3, 015009 (2022).

    See Also
    --------
    tICA : the reversible, equilibrium special case, scored by GMRQ.
    """

    def __init__(self, n_components=None, lag_time=1, epsilon=1e-6,
                 scaling=None, score_method='VAMP2'):
        self.n_components = n_components
        self.lag_time = lag_time
        self.epsilon = epsilon
        self.scaling = scaling
        self.score_method = score_method

    def _check_params(self):
        if int(self.lag_time) < 1:
            raise ValueError('lag_time must be at least 1, got %r' % self.lag_time)
        if self.scaling not in (None, 'kinetic_map', 'km'):
            raise ValueError("scaling must be None or 'kinetic_map', got %r" % self.scaling)
        if _normalize_method(self.score_method) is None:
            raise ValueError('score_method must be one of %s, got %r'
                             % (', '.join(_SCORE_METHODS), self.score_method))

    def fit(self, sequences, y=None):
        """Estimate the VAMP model.

        Parameters
        ----------
        sequences : list of array-like, each of shape (n_samples_i, n_features)
            Time series. Sequences may have different lengths but must have
            the same number of features. Sequences no longer than
            ``lag_time`` are ignored.
        y : None
            Ignored

        Returns
        -------
        self
        """
        self._check_params()
        check_iter_of_sequences(sequences)
        cov = _Covariances(sequences, int(self.lag_time))
        self.cov_00_, self.cov_0t_, self.cov_tt_ = cov.cov_00, cov.cov_0t, cov.cov_tt
        self.means_, self.means_lagged_ = cov.mean_0, cov.mean_t
        self.n_observations_ = cov.n_pairs
        self.n_sequences_ = cov.n_sequences

        L0 = _inv_sqrt_split(cov.cov_00, self.epsilon)
        Lt = _inv_sqrt_split(cov.cov_tt, self.epsilon)
        W = L0.T @ cov.cov_0t @ Lt
        A, s, BT = scipy.linalg.svd(W, lapack_driver='gesvd')
        m = min(L0.shape[1], Lt.shape[1])
        if self.n_components is not None:
            m = min(m, int(self.n_components))
        self.singular_values_ = s[:m]
        self.left_singular_vectors_ = L0 @ A[:, :m]
        self.right_singular_vectors_ = Lt @ BT[:m, :].T
        self.score_ = self._score_from(cov)
        return self

    def _check_fitted(self):
        if not hasattr(self, 'singular_values_'):
            raise ValueError('This VAMP instance is not fitted yet. Call fit() first.')

    def _projection(self):
        U = self.left_singular_vectors_
        if self.scaling in ('kinetic_map', 'km'):
            U = U * self.singular_values_[np.newaxis, :]
        return U

    @property
    def components_(self):
        self._check_fitted()
        return self.left_singular_vectors_.T

    @property
    def timescales_(self):
        self._check_fitted()
        with np.errstate(divide='ignore', invalid='ignore'):
            return -self.lag_time / np.log(self.singular_values_)

    def partial_transform(self, sequence):
        """Project one sequence onto the left singular functions.

        Parameters
        ----------
        sequence : array-like, shape (n_samples, n_features)

        Returns
        -------
        projected : array, shape (n_samples, n_components)
        """
        self._check_fitted()
        X = np.asarray(array2d(sequence), dtype=np.float64)
        return (X - self.means_) @ self._projection()

    def transform(self, sequences):
        """Project each sequence onto the left singular functions.

        Parameters
        ----------
        sequences : list of array-like, each of shape (n_samples_i, n_features)

        Returns
        -------
        projected : list of arrays, each of shape (n_samples_i, n_components)
        """
        check_iter_of_sequences(sequences, max_iter=3)
        return [self.partial_transform(X) for X in sequences]

    def fit_transform(self, sequences, y=None):
        """Fit the model and project ``sequences``."""
        return self.fit(sequences).transform(sequences)

    def _score_from(self, cov, method=None):
        method = _normalize_method(method or self.score_method)
        U = self.left_singular_vectors_
        V = self.right_singular_vectors_
        if method == 'VAMPE':
            K = np.diag(self.singular_values_)
            score = np.trace(2.0 * K @ U.T @ cov.cov_0t @ V
                             - K @ U.T @ cov.cov_00 @ U @ K @ V.T @ cov.cov_tt @ V)
        else:
            A = np.atleast_2d(_sym_inv_sqrt(U.T @ cov.cov_00 @ U, self.epsilon))
            B = np.atleast_2d(U.T @ cov.cov_0t @ V)
            C = np.atleast_2d(_sym_inv_sqrt(V.T @ cov.cov_tt @ V, self.epsilon))
            s = np.linalg.svd(A @ B @ C, compute_uv=False)
            score = np.sum(s) if method == 'VAMP1' else np.sum(s ** 2)
        return float(score) + 1.0  # the constant singular function

    def score(self, sequences, y=None):
        """VAMP score of this model on (new) data.

        The model's singular functions are evaluated on ``sequences``: a
        higher score means they capture more of the slow dynamics of that
        data. Use held-out data for cross-validation.

        Parameters
        ----------
        sequences : list of array-like, each of shape (n_samples_i, n_features)
            Test data, with the same features as the training data.
        y : None
            Ignored

        Returns
        -------
        score : float
            The ``score_method`` score (VAMP-1, VAMP-2 or VAMP-E), including
            +1 for the constant singular function.
        """
        self._check_fitted()
        check_iter_of_sequences(sequences)
        return self._score_from(_Covariances(sequences, int(self.lag_time)))

    def summarize(self):
        """Some summary information."""
        self._check_fitted()
        return '\n'.join([
            "Variational Approach for Markov Processes (VAMP)",
            "-------------------------------------------------",
            "n_components        : {}".format(len(self.singular_values_)),
            "lag_time            : {}".format(self.lag_time),
            "score_method        : {}".format(self.score_method),
            "Training score      : {:.6f}".format(self.score_),
            "",
            "Top 5 singular values:",
            "{}".format(self.singular_values_[:5]),
            "",
            "Top 5 timescales:",
            "{}".format(self.timescales_[:5]),
        ])


def _normalize_method(method):
    if not isinstance(method, str):
        return None
    m = method.upper().replace('-', '')
    return m if m in _SCORE_METHODS else None
