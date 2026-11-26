import numpy as np

import time
import warnings
import scipy.special
from sklearn import cluster, mixture
from sklearn.utils import check_random_state
from mdtraj.utils import ensure_type
from .discrete_approx import discrete_approx_mvn, NotSatisfiableError
from ..utils import check_iter_of_sequences, printoptions
from ..msm._markovstatemodel import _transmat_mle_prinz
from . import _hmm_core


class VonMisesHMM:
    """Hidden Markov Model with von Mises Emissions

    The von Mises distribution, (also known as the circular normal
    distribution or Tikhonov distribution) is a continuous probability
    distribution on the circle. For multivariate signals, the emmissions
    distribution implemented by this model is a product of univariate
    von Mises distributuons -- analogous to the multivariate Gaussian
    distribution with a diagonal covariance matrix.

    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a HMM.

    Notes
    -----
    The formulas for the maximization step of the E-M algorithim are
    adapted from [1]_, especially equations (11) and (13).

    Parameters
    ----------
    n_states : int
        The number of components (states) in the model
    n_init : int
        Number of time the EM algorithm will be run with different
        random seeds. The final results will be the best output of
        n_init consecutive runs in terms of log likelihood.
    n_iter : int
        The maximum number of iterations of expectation-maximization to
        run during each fitting round.
    thresh : float
        Convergence threshold for the log-likelihood during expectation
        maximization. When the increase in the log-likelihood is less
        than thresh between subsequent rounds of E-M, fitting will finish.
    reversible_type : str
        Method by which the reversibility of the transition matrix
        is enforced. 'mle' uses a maximum likelihood method that is
        solved by numerical optimization (BFGS), and 'transpose'
        uses a more restrictive (but less computationally complex)
        direct symmetrization of the expected number of counts.
    random_state : int, optional
        Random state, used during sampling.

    Attributes
    ----------
    means_ : array, shape (n_states, n_features)
        Mean parameters for each state.
    kappas_ : array, shape (n_states, n_features)
        Concentration parameter for each state. If `kappa` is zero, the
        distriution is uniform. If large, the distribution is very
        concentrated around the mean.
    transmat_ : array, shape (n_states, n_states)
        Matrix of transition probabilities between states.
    populations_ : array, shape(n_states)
        Population of each state.
    fit_logprob_ : array
        The log probability of the model after each iteration of fitting.

    References
    ----------
    .. [1] Prati, Andrea, Simone Calderara, and Rita Cucchiara. "Using circular
    statistics for trajectory shape analysis." Computer Vision and Pattern
    Recognition, 2008. CVPR 2008. IEEE Conference on. IEEE, 2008.
    .. [2] Murray, Richard F., and Yaniv Morgenstern. "Cue combination on the
    circle and the sphere." Journal of vision 10.11 (2010).
    """

    # Fitted parameters; None until fit() runs.
    _means_ = None
    _kappas_ = None
    _transmat_ = None
    _populations_ = None
    _fit_logprob_ = None
    _fit_time_ = None

    def __sklearn_is_fitted__(self):
        # Fitted attributes are private fields exposed as properties, which
        # sklearn's default check (vars(estimator)) cannot see.
        return self._transmat_ is not None


    def __init__(self, n_states, n_init=10, n_iter=10, thresh=1e-2, reversible_type='mle', random_state=None):
        self.n_states = int(n_states)
        self.n_features = -1
        self.n_init = int(n_init)
        self.n_iter = int(n_iter)
        # Stored at float32 precision, as the original Cython attribute was.
        self.thresh = float(np.float32(thresh))
        self.reversible_type = reversible_type
        self.random_state = random_state
        self.startprob = np.tile(1.0/n_states, n_states)
        self.stats = {}

    @classmethod
    def _init_argspec(self):
        # Returns the equivalent of `inspect.getargspec(VonMisesHMM.__init__)`.
        # this is required for the command line infastructure for all estimators
        # written in cython as a workaround, since cython modules don't
        # interact correctly with the `inspect` module.

        # any changes to the signature of __init__ need to be reflected here.
        from inspect import FullArgSpec
        return FullArgSpec(
        ['self', 'n_states', 'n_init', 'n_iter', 'thresh', 'reversible_type', 'random_state'],
          None, None,
          [10, 10, 1e-2, 1e-2, 'mle', None],
          None, None, None
        )

    @property
    def means_(self):
        return self._means_

    @property
    def kappas_(self):
        return self._kappas_

    @property
    def transmat_(self):
        return self._transmat_

    @property
    def populations_(self):
        return self._populations_

    @property
    def fit_logprob_(self):
        return self._fit_logprob_

    @property
    def overlap_(self):
        """
        Compute the matrix of normalized log overlap integrals between the hidden state distributions

        Notes
        -----
        The analytic formula used here follows from equation (A4) in :ref:`2`

        Returns
        -------
        noverlap : array, shape=(n_states, n_states)
            `noverlap[i,j]` gives the log of normalized overlap integral between
            states i and j. The normalized overlap integral is `integral(f(i)*f(j))
            / sqrt[integral(f(i)*f(i)) * integral(f(j)*f(j))]
        """
        logi0 = lambda x: np.log(scipy.special.i0(x))
        log2pi = np.log(2 * np.pi)

        log_overlap = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_states):
                for s in range(self.n_features):
                    kij = np.sqrt(self._kappas_[i, s] ** 2 + self._kappas_[j, s] ** 2 +
                                  2 * self._kappas_[i, s] * self._kappas_[j, s] *
                                  np.cos(self._means_[i, s] - self._means_[j, s]))
                    val = logi0(kij) - (log2pi + logi0(self._kappas_[i, s]) +
                                        logi0(self._kappas_[j, s]))
                    log_overlap[i, j] += val

        for i in range(self.n_states):
            for j in range(self.n_states):
                log_overlap[i, j] -= 0.5 * (log_overlap[i, i] + log_overlap[j, j])

        return log_overlap

    @property
    def timescales_(self):
        """The implied relaxation timescales of the hidden Markov transition
        matrix

        By diagonalizing the transition matrix, its propagation of an arbitrary
        initial probability vector can be written as a sum of the eigenvectors
        of the transition weighted by per-eigenvector term that decays
        exponentially with time. Each of these eigenvectors describes a
        "dynamical mode" of the transition matrix and has a characteristic
        timescale, which gives the timescale on which that mode decays towards
        equilibrium. These timescales are given by :math:`-1/log(u_i)` where
        :math:`u_i` are the eigenvalues of the transition matrix. In a
        reversible HMM with N states, the number of timescales is at most N-1.
        (The -1 comes from the fact that the stationary distribution of the chain
        is associated with an eigenvalue of 1, and an infinite characteristic
        timescale). The number of timescales can be less than N-1 for every
        eigenvalue of the transition matrix that is negative (which is
        allowable by detailed balance).

        Returns
        -------
        timescales : array, shape=[n_timescales]
            The characteristic timescales of the transition matrix. If the model
            has not been fit or does not have a transition matrix, the return
            value will be None.
        """
        if self._transmat_ is None:
            return None
        try:
            eigvals = np.linalg.eigvals(self._transmat_)

            # sort the eigenvalues in descending order (i.e. sort + reverse),
            # then discard the very first (largest) eigenvalue, which is the
            # stationary one.
            eigvals = np.sort(eigvals)[::-1][1:]

            # retain only eigenvalues > 0. These are the ones that correspond
            # to implied timescales. Eigenvalues below zero are possible (the
            # detailed balance constraint guarentees that the eigenvalues are
            # in -1 < l <= 1, but not that they strictly positive). But
            # eigevalues below zero do not have a real interpretation as
            # "implied timescales" because they correspond to sort of
            # damped oscillatory decays.

            eigvals = eigvals[eigvals > 0]
            return -1.0 / np.log(eigvals)
        except np.linalg.LinAlgError:
            # this can happen if the transition matrix contains Nans
            # or Infs, and possibly for other reasons like convergence
            return np.nan * np.ones(self.n_states - 1)

    def summarize(self):
        """Get a string summarizing the model."""
        with printoptions(precision=4, suppress=True):
            return """VonMises HMM
------------
n_states : {n_states}
logprob: {logprob}
fit_time: {fit_time:.3f}s

populations: {populations}
transmat:
{transmat}
timescales: {timescales}
    """.format(n_states=self.n_states, logprob=self._fit_logprob_[-1],
               populations=str(self._populations_), transmat=str(self._transmat_),
               timescales=self.timescales_, fit_time=self._fit_time_)

    def fit_predict(self, sequences, y=None):
        """Find most likely hidden-state sequence corresponding to
        each data timeseries.

        Uses the Viterbi algorithm.

        Parameters
        ----------
        sequences : list
            List of 2-dimensional array observation sequences, each of which
            has shape (n_samples_i, n_features), where n_samples_i
            is the length of the i_th observation.

        Returns
        -------
        viterbi_logprob : float
            Log probability of the maximum likelihood path through the HMM.

        hidden_sequences : list of np.ndarrays[dtype=int, shape=n_samples_i]
            Index of the most likely states for each observation.
        """
        self.fit(sequences, y=y)
        return self.predict(sequences)

    def fit(self, sequences, y=None):
        """Estimate model parameters.

        Parameters
        ----------
        sequences : list
            List of 2-dimensional array observation sequences, each of which
            has shape (n_samples_i, n_features), where n_samples_i
            is the length of the i_th observation.
        """
        self._validate_sequences(sequences)
        self.n_features = sequences[0].shape[1]
        dtype = sequences[0].dtype
        best_fit = {'params': {}, 'loglikelihood': -np.inf}
        start_time = time.time()
        self.stats = {}
        total_iters = 0
        for run in range(self.n_init):
            self._init(sequences)
            if dtype not in (np.float32, np.float64):
                raise ValueError('Unsupported data type: '+str(dtype))
            self._fit_em(sequences)
            total_iters += len(self.stats['log_probability'])

            # If this is better than our other runs, keep it
            if self.stats['log_probability'][-1] > best_fit['loglikelihood']:
                best_fit['loglikelihood'] = self.stats['log_probability'][-1]
                best_fit['params'] = {'means': self.means_,
                                      'kappas': self.kappas_,
                                      'populations': self.populations_,
                                      'transmat': self.transmat_,
                                      'fit_logprob': self.stats['log_probability']}

        # Set the final values
        self._means_ = best_fit['params']['means']
        self._kappas_ = best_fit['params']['kappas']
        self._transmat_ = best_fit['params']['transmat']
        self._populations_ = best_fit['params']['populations']
        self._fit_logprob_ = best_fit['params']['fit_logprob']
        self._fit_time_ = time.time() - start_time

        return self

    def _validate_sequences(self, sequences):
        """Make sure the sequences supplied by the user are valid."""
        if len(sequences) == 0:
           raise ValueError('sequences is empty')
        check_iter_of_sequences(sequences)
        dtype = sequences[0].dtype
        if any(s.dtype != dtype for s in sequences):
            raise ValueError('All sequences must have the same data type')
        if self.n_features == -1:
            # It hasn't been set yet.
            n_features = sequences[0].shape[1]
        else:
            n_features = self.n_features
        if any(s.shape[1] != n_features for s in sequences):
            raise ValueError('All sequences must have %d features' % n_features)

    def _init(self, sequences):
        """Find initial means (hot start)"""
        self._transmat_ = np.ones((self.n_states, self.n_states)) * (1.0 / self.n_states)
        self._populations_ = np.ones(self.n_states) / self.n_states

        # Cluster the sine and cosine of the input data with kmeans to
        # get initial centers
        # the number of initial trajectories used should be configurable...
        # currently it's just the 0-th one
        sequences = [ensure_type(s, dtype=np.float32, ndim=2, name='s', warn_on_cast=False)
                     for s in sequences]
        dataset = np.vstack(sequences)
        cluster_centers = cluster.MiniBatchKMeans(n_clusters=self.n_states).fit(
            np.hstack((np.sin(dataset), np.cos(dataset)))).cluster_centers_
        self._means_ = np.arctan2(cluster_centers[:, :self.n_features],
                                  cluster_centers[:, self.n_features:])
        self._kappas_ = np.ones((self.n_states, self.n_features))

    # --- numerical core (Numba kernels in _hmm_core) -------------------------

    def _emission_params(self):
        return (np.asarray(self._means_, dtype=np.float64), np.asarray(self._kappas_, dtype=np.float64), np.log(scipy.special.i0(np.asarray(self._kappas_, dtype=np.float64))))

    def _log_startprob(self):
        # The C++ fitter used before MSMBuilder Neo was handed startprob
        # where it expected log(startprob), which shifted every reported
        # log-likelihood by n_trajectories * (1/n_states + log(n_states)).
        return np.log(np.asarray(self.startprob, dtype=np.float64))

    def _fit_em(self, sequences):
        """Expectation-maximization with the current parameters as the
        starting point. Mirrors the old C++ HMMFitter::fit loop: the M-step
        runs after every E-step unless the log-probability has converged,
        and self.stats is refreshed just before each M-step."""
        X, offsets = _hmm_core.concatenate(sequences)
        log_startprob = self._log_startprob()
        iter_logprob = []
        for i in range(self.n_iter):
            logprob, trans, post, s1, s2 = _hmm_core.vonmises_estep(
                X, offsets, log_startprob, _hmm_core.log_transmat(self._transmat_),
                *self._emission_params())
            iter_logprob.append(logprob)
            if i > 0 and abs(iter_logprob[i] - iter_logprob[i - 1]) < self.thresh:
                break
            self.stats = {'trans': trans, 'cosobs': s1, 'sinobs': s2, 'post': post,
                          'log_probability': np.array(iter_logprob)}
            self._do_mstep()

    def _score(self, sequences):
        X, offsets = _hmm_core.concatenate(sequences)
        return _hmm_core.vonmises_score(X, offsets, self._log_startprob(),
                                  _hmm_core.log_transmat(self._transmat_),
                                  *self._emission_params())

    def _predict(self, sequences):
        log_startprob = self._log_startprob()
        log_tm = _hmm_core.log_transmat(self._transmat_)
        params = self._emission_params()
        logprob = 0.0
        viterbi_sequences = []
        for s in sequences:
            lp, states = _hmm_core.vonmises_viterbi(
                np.ascontiguousarray(s, dtype=np.float64), log_startprob, log_tm, *params)
            logprob += lp
            viterbi_sequences.append(states)
        return (logprob, viterbi_sequences)


    def _getdiff(self, means, difference_cutoff):
        diff = np.zeros((self.n_features, self.n_states, self.n_states))
        for i in range(self.n_features):
            diff[i] = np.maximum(
                np.abs(np.subtract.outer(means[:, i], means[:, i])), difference_cutoff)
        return diff

    def _do_mstep(self):
        stats = self.stats
        transmat_prior = 1.0
        if self.reversible_type == 'mle':
            counts = np.maximum(
                stats['trans'] + transmat_prior - 1.0, 1e-20).astype(np.float64)
            self._transmat_, self._populations_ = _transmat_mle_prinz(counts)
        elif self.reversible_type == 'transpose':
            revcounts = np.maximum(
                transmat_prior - 1.0 + stats['trans'] + stats['trans'].T, 1e-20)
            populations = np.sum(revcounts, axis=0)
            self._populations_ = populations / np.sum(populations)
            self._transmat_ = revcounts / np.sum(revcounts, axis=1)[:, np.newaxis]
        else:
            raise ValueError('Invalid value for reversible_type: %s '
                             'Must be either "mle" or "transpose"'
                             % self.reversible_type)

        np.arctan2(stats['sinobs'], stats['cosobs'], self._means_)

        # we don't want denom to be zero, because then the new value of the means
        # will be nan/inf. so padd it up by a very small constant. This particular
        # padding is following the sklearn mixture model m_step code from
        # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/mixture/gmm.py#L496
        EPS = np.finfo(np.float32).eps
        kappa_denom = (stats['post'][:, np.newaxis] + 10 * EPS)
        kappa_num = stats['cosobs']*np.cos(self._means_) + stats['sinobs']*np.sin(self._means_)
        inv_kappas = kappa_num / kappa_denom
        self._kappas_ = inverse_mbessel_ratio(inv_kappas)

    def score(self, sequences):
        """Log-likelihood of sequences under the model

        Parameters
        ----------
        sequences : list
            List of 2-dimensional array observation sequences, each of which
            has shape (n_samples_i, n_features), where n_samples_i
            is the length of the i_th observation.
        """
        self._validate_sequences(sequences)
        dtype = sequences[0].dtype
        if dtype not in (np.float32, np.float64):
            raise ValueError('Unsupported data type: '+str(dtype))
        return self._score(sequences)

    def predict(self, sequences):
        """Find most likely hidden-state sequence corresponding to
        each data timeseries.

        Uses the Viterbi algorithm.

        Parameters
        ----------
        sequences : list
            List of 2-dimensional array observation sequences, each of which
            has shape (n_samples_i, n_features), where n_samples_i
            is the length of the i_th observation.

        Returns
        -------
        viterbi_logprob : float
            Log probability of the maximum likelihood path through the HMM.

        hidden_sequences : list of np.ndarrays[dtype=int, shape=n_samples_i]
            Index of the most likely states for each observation.
        """
        self._validate_sequences(sequences)
        dtype = sequences[0].dtype
        if dtype not in (np.float32, np.float64):
            raise ValueError('Unsupported data type: '+str(dtype))
        return self._predict(sequences)

    def __reduce__(self):
        """Pickle support"""
        args = (self.n_states, self.n_init, self.n_iter, self.thresh, self.reversible_type, self.random_state)
        state = (self._means_, self._kappas_, self._transmat_, self._populations_, self._fit_logprob_, self._fit_time_, self.n_features)
        return (self.__class__, args, state)

    def __setstate__(self, state):
        """Pickle support"""
        self._means_ = state[0]
        self._kappas_ = state[1]
        self._transmat_ = state[2]
        self._populations_ = state[3]
        self._fit_logprob_ = state[4]
        self._fit_time_ = state[5]
        self.n_features = state[6]


class _InverseMBesselRatio:

    """
    Inverse the function given by the ratio modified Bessel function of the
    first kind of order 1 to the modified Bessel function of the first kind
    of order 0.

    y = A(x) = I_1(x) / I_0(x)

    This function computes A^(-1)(y) by way of a precomputed spline
    interpolation
    """

    def __init__(self, n_points=512):
        self._n_points = n_points
        self._is_fit = False
        self._min_x = 1e-5
        self._max_x = 700

    def _fit(self):
        """We want to do the fitting once, but not at import time since it
        slows down the loading of the interpreter"""
        # Fitting takes about 0.5s on a laptop, and wth 512 points and cubic
        # interpolation, gives typical errors around 1e-9
        x = np.logspace(np.log10(self._min_x), np.log10(self._max_x),
                        self._n_points)
        y = self.bessel_ratio(x)
        self._min = np.min(y)
        self._max = np.max(y)

        # Spline fit the log of the inverse function
        from scipy.interpolate import interp1d
        self._spline = interp1d(y, np.log(x), kind='cubic')
        self._is_fit = True

    def __call__(self, y):
        if not self._is_fit:
            self._fit()

        y = np.asarray(y)
        y = np.clip(y, a_min=self._min, a_max=self._max)

        if np.any(np.logical_or(0 > y, y > 1)):
            raise ValueError('Domain error. y must be in (0, 1)')
        x = np.exp(self._spline(y))

        # DEBUGGING CODE
        # for debugging, the line below prints the error in the inverse
        # by printing y - A(A^(-1)(y
        # print('spline inverse error', y - self.bessel_ratio(x))

        return x

    @staticmethod
    def bessel_ratio(x):
        numerator = scipy.special.iv(1, x)
        denominator = scipy.special.iv(0, x)
        return numerator / denominator

# Module-level instance; the spline is fitted on first use.
inverse_mbessel_ratio = _InverseMBesselRatio()
