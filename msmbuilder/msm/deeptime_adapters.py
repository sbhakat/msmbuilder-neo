"""Adapters to deeptime's Markov-model estimators.

deeptime [1]_ implements estimators that MSMBuilder does not have. These
adapters run them on MSMBuilder-style input (lists of state-label sequences)
and return MSMBuilder objects, so the rest of MSMBuilder (``timescales_``,
``eigtransform``, ``score``, TPT, lumping, ...) works on the results:

* :class:`TRAM` -- the transition-based reweighting analysis method [2]_:
  combines biased simulations (umbrella sampling, replica exchange,
  metadynamics with static bias, ...) into unbiased free energies and one
  Markov state model per thermodynamic state.
* :class:`OOMReweightedMSM` -- an MSM corrected for non-equilibrium starting
  conditions and non-Markovian discretizations with observable operator
  models [3]_; useful for many short simulations started out of equilibrium.
* :func:`ck_test` -- the Chapman-Kolmogorov test of a Markov state model.
* :func:`to_deeptime` / :func:`from_deeptime` -- convert fitted Markov state
  models between the two packages.

deeptime is an optional dependency: ``pip install msmbuilder-neo[deeptime]``.

References
----------
.. [1] Hoffmann, M. et al. "Deeptime: a Python library for machine learning
   dynamical models from time series data." Mach. Learn.: Sci. Technol. 3,
   015009 (2022).
.. [2] Wu, H., Paul, F., Wehmeyer, C. and Noe, F. "Multiensemble Markov
   models of molecular thermodynamics and kinetics." PNAS 113, E3221-E3230
   (2016).
.. [3] Nueske, F., Wu, H., Prinz, J.-H., Wehmeyer, C., Clementi, C. and
   Noe, F. "Markov state models from short non-equilibrium simulations --
   analysis and correction of estimation bias." J. Chem. Phys. 146, 094104
   (2017).
"""
import numbers
import warnings

import numpy as np
from sklearn.base import clone

from ..base import BaseEstimator
from ..utils import list_of_1d
from .msm import MarkovStateModel

__all__ = ['TRAM', 'OOMReweightedMSM', 'ck_test', 'to_deeptime', 'from_deeptime']


def _deeptime():
    try:
        import deeptime
        import deeptime.markov.msm  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without deeptime
        raise ImportError(
            'This needs deeptime. Install it with "pip install deeptime" or '
            '"pip install msmbuilder-neo[deeptime]".') from e
    return deeptime


def _encode(sequences, mapping=None):
    """Map state labels to integers 0..n-1 (sorted labels, as MSMBuilder's
    ``mapping_``). Returns the integer sequences and the mapping."""
    sequences = list_of_1d(sequences)
    if mapping is None:
        labels = np.unique(np.concatenate(sequences))
        if labels.dtype.kind == 'f':
            if np.isnan(labels).any():
                raise ValueError('NaN state labels are not supported here')
            if not np.all(labels == np.round(labels)):
                raise ValueError('state labels must be integers or strings, not fractions')
        mapping = {lab.item() if hasattr(lab, 'item') else lab: i for i, lab in enumerate(labels)}
    encoded = []
    for seq in sequences:
        try:
            encoded.append(np.array([mapping[s.item() if hasattr(s, 'item') else s]
                                     for s in seq], dtype=np.int32))
        except KeyError as e:
            raise ValueError('state label %r was not seen during fit' % (e.args[0],)) from None
    return encoded, mapping


def _labels_of(mapping):
    """Labels in order of their integer codes."""
    return sorted(mapping, key=mapping.get)


def _dense(a):
    return a.toarray() if hasattr(a, 'toarray') else np.asarray(a)


def _fill_from_deeptime(msm, model, labels):
    """Set the fitted attributes of an MSMBuilder MarkovStateModel from a
    deeptime MarkovStateModel whose states are codes into ``labels``."""
    transmat = np.array(_dense(model.transition_matrix), dtype=float)
    msm.transmat_ = transmat
    msm.populations_ = np.array(model.stationary_distribution, dtype=float)
    msm.n_states_ = transmat.shape[0]
    count_model = model.count_model if model.has_count_model else None
    if count_model is not None:
        msm.countsmat_ = np.array(_dense(count_model.count_matrix), dtype=float)
        symbols = np.asarray(count_model.state_symbols)
        msm.percent_retained_ = 100.0
        full = getattr(count_model, 'count_matrix_full', None)
        if full is not None and _dense(full).sum() > 0:
            msm.percent_retained_ = 100.0 * msm.countsmat_.sum() / _dense(full).sum()
    else:
        msm.countsmat_ = None
        symbols = np.arange(msm.n_states_)
        msm.percent_retained_ = 100.0
    if labels is None:
        labels = list(range(int(symbols.max()) + 1 if len(symbols) else 0))
    msm.mapping_ = {labels[int(s)]: i for i, s in enumerate(symbols)}
    msm._is_dirty = True
    return msm


def _largest_connected(msm):
    """Restrict a fitted MarkovStateModel to its strongly connected set of
    states with the most transition counts (all states without counts)."""
    from scipy.sparse.csgraph import connected_components
    n_comp, comp = connected_components(msm.transmat_ > 0, directed=True, connection='strong')
    if n_comp == 1:
        return msm
    weight = msm.countsmat_ if msm.countsmat_ is not None else np.ones_like(msm.transmat_)
    totals = [weight[np.ix_(comp == c, comp == c)].sum() for c in range(n_comp)]
    keep = np.flatnonzero(comp == int(np.argmax(totals)))
    labels = _labels_of(msm.mapping_)
    transmat = msm.transmat_[np.ix_(keep, keep)]
    msm.transmat_ = transmat / transmat.sum(axis=1, keepdims=True)
    msm.populations_ = msm.populations_[keep] / msm.populations_[keep].sum()
    if msm.countsmat_ is not None:
        total = msm.countsmat_.sum()
        msm.countsmat_ = msm.countsmat_[np.ix_(keep, keep)]
        if total > 0:
            msm.percent_retained_ *= msm.countsmat_.sum() / total
    msm.mapping_ = {labels[i]: k for k, i in enumerate(keep)}
    msm.n_states_ = len(keep)
    msm._is_dirty = True
    return msm


def from_deeptime(model, labels=None, n_timescales=None):
    """Convert a deeptime ``MarkovStateModel`` to an MSMBuilder one.

    Parameters
    ----------
    model : deeptime.markov.msm.MarkovStateModel
        A fitted deeptime model (for example from ``MaximumLikelihoodMSM``,
        ``BayesianMSM`` or ``TRAM``).
    labels : sequence, optional
        ``labels[s]`` is the MSMBuilder state label of deeptime state symbol
        ``s`` (the integer that appeared in the discrete trajectories). By
        default the symbols themselves are the labels.
    n_timescales : int, optional
        Passed to :class:`MarkovStateModel`.

    Returns
    -------
    msm : MarkovStateModel
        A fitted model with ``transmat_``, ``populations_``, ``countsmat_``
        (when the deeptime model has a count model) and ``mapping_``.
    """
    _deeptime()
    lag = model.lagtime if model.lagtime is not None else 1
    if model.has_count_model and model.count_model.lagtime is not None:
        lag = model.count_model.lagtime
    sliding = True
    if model.has_count_model and model.count_model.counting_mode is not None:
        sliding = 'sliding' in str(model.count_model.counting_mode)
    msm = MarkovStateModel(lag_time=int(lag), n_timescales=n_timescales,
                           reversible_type='mle' if model.reversible else 'none',
                           ergodic_cutoff='off', sliding_window=sliding, verbose=False)
    return _fill_from_deeptime(msm, model, labels)


def to_deeptime(msm):
    """Convert a fitted MSMBuilder ``MarkovStateModel`` to deeptime.

    The deeptime model has the same transition matrix, stationary
    distribution and count matrix. Its count model's state symbols are the
    MSMBuilder state labels when those are integers (so models fitted on the
    same labelled trajectories, at different lag times, line up in deeptime),
    and the internal state indices otherwise.

    Parameters
    ----------
    msm : MarkovStateModel

    Returns
    -------
    model : deeptime.markov.msm.MarkovStateModel
    """
    dt = _deeptime()
    from deeptime.markov import TransitionCountModel
    if getattr(msm, 'transmat_', None) is None:
        raise ValueError('The MarkovStateModel is not fitted yet.')
    labels = _labels_of(msm.mapping_)
    if all(isinstance(lab, numbers.Integral) and lab >= 0 for lab in labels):
        symbols = np.array(labels, dtype=int)
    else:
        symbols = np.arange(msm.n_states_)
    count_model = None
    if getattr(msm, 'countsmat_', None) is not None:
        n_full = int(symbols.max()) + 1
        full = np.zeros((n_full, n_full))
        full[np.ix_(symbols, symbols)] = msm.countsmat_
        count_model = TransitionCountModel(
            np.asarray(msm.countsmat_, dtype=float),
            counting_mode='sliding' if getattr(msm, 'sliding_window', True) else 'sample',
            lagtime=int(msm.lag_time), state_symbols=symbols, count_matrix_full=full)
    reversible = str(getattr(msm, 'reversible_type', 'mle')).lower() in ('mle', 'transpose')
    return dt.markov.msm.MarkovStateModel(
        np.asarray(msm.transmat_, dtype=float),
        stationary_distribution=np.asarray(msm.populations_, dtype=float),
        reversible=reversible, count_model=count_model, lagtime=int(msm.lag_time))


class OOMReweightedMSM(MarkovStateModel):
    """Markov state model with observable-operator-model reweighting.

    Short simulations started out of equilibrium, analysed on a discretization
    that is not exactly Markovian, give biased MSM estimates. This estimator
    (deeptime's ``OOMReweightedMSM``, [1]_) estimates an observable operator
    model from two-step transition counts and uses it to reweight the
    transition counts to their equilibrium values before estimating the
    transition matrix.

    The result is an ordinary fitted :class:`MarkovStateModel`
    (``transmat_``, ``populations_``, ``timescales_``, ``transform``,
    ``score``, ...); the deeptime model is kept as ``model_`` for the
    OOM-specific quantities (``oom_eigenvalues``, ``oom_timescales``,
    ``oom_rank``, ...).

    Parameters
    ----------
    lag_time : int, default=1
        Lag time, in frames.
    reversible : bool, default=True
        Estimate a reversible transition matrix.
    sliding_window : bool, default=True
        Count transitions with a sliding window (otherwise every
        ``lag_time`` frames).
    rank_mode : {'bootstrap_counts', 'bootstrap_trajs'}, default='bootstrap_counts'
        How the rank of the OOM is chosen: by bootstrapping the two-step
        counts or the trajectories.
    nbs : int, default=10000
        Number of bootstrap samples for the rank decision.
    tol_rank : float, default=10.0
        Signal-to-noise threshold of the rank decision.
    n_timescales : int, optional
        Number of timescales to compute (all by default).
    random_state : int or None, default=None
        Seed for the bootstrap of the rank decision. The rank decision uses
        NumPy's global random state; with a seed it is saved and restored
        around the fit.
    verbose : bool, default=False
    score_method : {'gmrq', 'vamp2', 'vamp1'}, default='gmrq'
        As for :class:`MarkovStateModel`.

    Attributes
    ----------
    model_ : deeptime.markov.msm.KoopmanReweightedMSM
        The deeptime model.
    transmat_, populations_, countsmat_, mapping_, n_states_
        As for :class:`MarkovStateModel`; ``countsmat_`` are the (not
        reweighted) transition counts on the model's states.

    References
    ----------
    .. [1] Nueske, F., Wu, H., Prinz, J.-H., Wehmeyer, C., Clementi, C. and
       Noe, F. J. Chem. Phys. 146, 094104 (2017).
    """

    def __init__(self, lag_time=1, reversible=True, sliding_window=True,
                 rank_mode='bootstrap_counts', nbs=10000, tol_rank=10.0, n_timescales=None,
                 random_state=None, verbose=False, score_method='gmrq'):
        super().__init__(lag_time=lag_time, n_timescales=n_timescales, verbose=verbose,
                         sliding_window=sliding_window, score_method=score_method)
        self.reversible = reversible
        self.rank_mode = rank_mode
        self.nbs = nbs
        self.tol_rank = tol_rank
        self.random_state = random_state

    # MarkovStateModel attributes that are not parameters of this estimator
    reversible_type = property(lambda self: 'mle' if self.reversible else 'none',
                               lambda self, value: None)
    ergodic_cutoff = property(lambda self: 'off', lambda self, value: None)
    prior_counts = property(lambda self: 0, lambda self, value: None)

    def fit(self, sequences, y=None):
        """Estimate the reweighted MSM from a list of state-label sequences."""
        dt = _deeptime()
        if int(self.lag_time) < 1:
            raise ValueError('lag_time must be at least 1, got %r' % self.lag_time)
        encoded, mapping = _encode(sequences)
        est = dt.markov.msm.OOMReweightedMSM(
            lagtime=int(self.lag_time), reversible=bool(self.reversible),
            count_mode='sliding' if self.sliding_window else 'sample',
            rank_mode=self.rank_mode, nbs=int(self.nbs), tol_rank=float(self.tol_rank))
        if self.random_state is None:
            self.model_ = est.fit_fetch(encoded)
        else:
            saved = np.random.get_state()
            np.random.seed(int(self.random_state))
            try:
                self.model_ = est.fit_fetch(encoded)
            finally:
                np.random.set_state(saved)
        _fill_from_deeptime(self, self.model_, _labels_of(mapping))
        return self


class TRAM(BaseEstimator):
    """Transition-based reweighting analysis method (TRAM).

    TRAM [1]_ combines simulations run in several thermodynamic states
    (different bias potentials, temperatures or Hamiltonians) into one
    model: the free energy of each thermodynamic state, the unbiased free
    energy of each Markov state, and a Markov state model for every
    thermodynamic state. Unlike WHAM or MBAR it does not need each
    simulation to be in global equilibrium, only in local equilibrium within
    each Markov state. This adapter runs deeptime's TRAM on MSMBuilder
    state-label sequences.

    Parameters
    ----------
    lag_time : int, default=1
        Lag time, in frames, of the transition counts.
    sliding_window : bool, default=True
        Count transitions with a sliding window (otherwise every
        ``lag_time`` frames).
    maxiter : int, default=1000
        Maximum number of self-consistent iterations.
    maxerr : float, default=1e-8
        Convergence criterion (change of the free energies, in kT).
    init_strategy : {'MBAR', None}, default='MBAR'
        Initialize the free energies with MBAR (or with zeros).
    init_maxiter : int, default=1000
    init_maxerr : float, default=1e-8
        Iterations and tolerance of the MBAR initialization.

    Attributes
    ----------
    model_ : deeptime.markov.msm.TRAMModel
        The deeptime model.
    n_therm_states_ : int
        Number of thermodynamic states.
    therm_state_energies_ : array, shape (n_therm_states_,)
        Free energy of each thermodynamic state, in kT (up to a constant).
    markov_state_energies_ : array, shape (n_states_,)
        Unbiased free energy of each Markov state, in kT.
    populations_ : array, shape (n_states_,)
        Unbiased equilibrium population of each Markov state,
        ``exp(-markov_state_energies_)`` normalized.
    biased_conf_energies_ : array, shape (n_therm_states_, n_states_)
        Free energy of each Markov state in each thermodynamic state.
    msms_ : list of MarkovStateModel
        The Markov state model of each thermodynamic state, on the largest
        set of states connected by transitions observed in that
        thermodynamic state (deeptime's models span all states of the
        estimate, with unvisited states isolated).
    mapping_ : dict
        State label -> index into ``markov_state_energies_``.
    n_states_ : int
        Number of Markov states (distinct labels).
    converged_ : bool
        Whether the free energies converged to ``maxerr`` within ``maxiter``
        iterations (otherwise deeptime also issues a ConvergenceWarning).

    References
    ----------
    .. [1] Wu, H., Paul, F., Wehmeyer, C. and Noe, F. Proc. Natl. Acad. Sci.
       USA 113, E3221-E3230 (2016).
    """

    def __init__(self, lag_time=1, sliding_window=True, maxiter=1000, maxerr=1e-8,
                 init_strategy='MBAR', init_maxiter=1000, init_maxerr=1e-8):
        self.lag_time = lag_time
        self.sliding_window = sliding_window
        self.maxiter = maxiter
        self.maxerr = maxerr
        self.init_strategy = init_strategy
        self.init_maxiter = init_maxiter
        self.init_maxerr = init_maxerr

    @staticmethod
    def _check_bias(encoded, bias_matrices, n_therm=None):
        if len(bias_matrices) != len(encoded):
            raise ValueError('bias_matrices must have one entry per sequence '
                             '(got %d for %d sequences)' % (len(bias_matrices), len(encoded)))
        out = []
        for seq, b in zip(encoded, bias_matrices):
            b = np.asarray(b, dtype=np.float64)
            if b.ndim != 2 or b.shape[0] != len(seq):
                raise ValueError('each bias matrix must have shape (n_frames, n_therm_states)')
            if n_therm is not None and b.shape[1] != n_therm:
                raise ValueError('bias matrices have %d thermodynamic states, the model %d'
                                 % (b.shape[1], n_therm))
            out.append(b)
        if len({b.shape[1] for b in out}) != 1:
            raise ValueError('all bias matrices must have the same number of columns')
        return out

    def fit(self, sequences, bias_matrices, ttrajs=None):
        """Estimate TRAM.

        Parameters
        ----------
        sequences : list of array-like
            Markov state label of each frame of each trajectory (e.g. cluster
            labels).
        bias_matrices : list of array-like, each of shape (n_frames_i, n_therm_states)
            ``bias_matrices[i][n, k]`` is the bias energy, in kT, of frame
            ``n`` of trajectory ``i`` evaluated in thermodynamic state ``k``
            (its reduced potential relative to the unbiased one).
        ttrajs : list of array-like, optional
            Thermodynamic state in which each frame was sampled (for
            replica-exchange-like data). By default trajectory ``i`` is
            sampled in thermodynamic state ``i``.
        """
        dt = _deeptime()
        encoded, mapping = _encode(sequences)
        bias = self._check_bias(encoded, bias_matrices)
        n_therm = bias[0].shape[1]
        if ttrajs is None:
            if len(encoded) != n_therm:
                raise ValueError('without ttrajs, there must be one trajectory per '
                                 'thermodynamic state (%d trajectories, %d states)'
                                 % (len(encoded), n_therm))
            ttrajs = [np.full(len(s), i, dtype=np.int32) for i, s in enumerate(encoded)]
        else:
            ttrajs = [np.asarray(t, dtype=np.int32) for t in ttrajs]
            if len(ttrajs) != len(encoded) or any(len(t) != len(s)
                                                  for t, s in zip(ttrajs, encoded)):
                raise ValueError('ttrajs must have one value per frame of each sequence')
        est = dt.markov.msm.TRAM(
            lagtime=int(self.lag_time), count_mode='sliding' if self.sliding_window else 'sample',
            maxiter=int(self.maxiter), maxerr=float(self.maxerr),
            init_strategy=self.init_strategy, init_maxiter=int(self.init_maxiter),
            init_maxerr=float(self.init_maxerr))
        dataset = dt.markov.msm.TRAMDataset(
            dtrajs=encoded, bias_matrices=bias, ttrajs=ttrajs, n_therm_states=n_therm,
            n_markov_states=len(mapping), lagtime=int(self.lag_time),
            count_mode='sliding' if self.sliding_window else 'sample')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = est.fit_fetch(dataset)
        self.converged_ = True
        for w in caught:
            if w.category.__name__ == 'ConvergenceWarning':
                self.converged_ = False
            warnings.warn(w.message, w.category, stacklevel=2)
        self.model_ = model
        self.mapping_ = mapping
        self.n_states_ = len(mapping)
        self.n_therm_states_ = n_therm
        self.therm_state_energies_ = np.array(model.therm_state_energies, dtype=float)
        self.markov_state_energies_ = np.array(model.markov_state_energies, dtype=float)
        self.biased_conf_energies_ = np.array(model.biased_conf_energies, dtype=float)
        f = self.markov_state_energies_
        finite = np.isfinite(f)
        pops = np.zeros_like(f)
        pops[finite] = np.exp(-(f[finite] - f[finite].min()))
        self.populations_ = pops / pops.sum()

        labels = _labels_of(mapping)
        collection = model.msm_collection
        self.msms_ = []
        for k in range(collection.n_connected_msms):
            collection.select(k)
            self.msms_.append(_largest_connected(from_deeptime(collection, labels=labels)))
        collection.select(0)
        return self

    def _check_fitted(self):
        if not hasattr(self, 'model_'):
            raise ValueError('This TRAM instance is not fitted yet. Call fit() first.')

    def sample_weights(self, sequences, bias_matrices, therm_state=-1):
        """Weight of every frame for computing averages in one thermodynamic
        state (``therm_state=-1``: the unbiased state).

        Returns a list with one array of weights per sequence; over all
        frames they sum to one.
        """
        self._check_fitted()
        encoded, _ = _encode(sequences, self.mapping_)
        bias = self._check_bias(encoded, bias_matrices, self.n_therm_states_)
        logw = self.model_.compute_sample_weights_log(encoded, bias, therm_state=therm_state)
        return [np.exp(np.asarray(w, dtype=float)) for w in logw]

    def pmf(self, sequences, bias_matrices, bin_indices, n_bins=None, therm_state=-1):
        """Potential of mean force (free energy, in kT, minimum zero) along a
        discretized coordinate, from TRAM's frame weights.

        Parameters
        ----------
        sequences, bias_matrices
            Trajectories as passed to ``fit`` (or new ones on the same states
            and thermodynamic states).
        bin_indices : list of array-like of int
            Bin of each frame along the coordinate.
        n_bins : int, optional
            Number of bins (by default the largest index plus one).
        therm_state : int, default=-1
            Thermodynamic state; -1 is the unbiased state.

        Returns
        -------
        pmf : array, shape (n_bins,)
            Bins without samples are ``inf``.
        """
        self._check_fitted()
        encoded, _ = _encode(sequences, self.mapping_)
        bias = self._check_bias(encoded, bias_matrices, self.n_therm_states_)
        bins = [np.asarray(b, dtype=np.int32) for b in bin_indices]
        pmf = np.array(self.model_.compute_PMF(encoded, bias, bins, n_bins=n_bins,
                                               therm_state=therm_state), dtype=float)
        finite = np.isfinite(pmf)
        if finite.any():
            pmf[finite] -= pmf[finite].min()
        return pmf


def ck_test(estimator, sequences, lag_times, n_metastable_sets, include_lag0=True,
            err_est=False):
    """Chapman-Kolmogorov test of a Markov state model.

    Markov state models are fitted at each lag time in ``lag_times`` (with
    the parameters of ``estimator``); the model at the first, shortest lag
    time is propagated, ``T(tau)^k``, and its predictions of the probability
    of staying in (and moving between) ``n_metastable_sets`` PCCA+
    metastable sets are compared with the models estimated directly at lag
    time ``k * tau``. For a Markovian model they agree. This uses deeptime's
    ``ck_test``.

    Parameters
    ----------
    estimator : MarkovStateModel
        Supplies the parameters (fitted or not); it is cloned for each lag
        time. Must be reversible for the PCCA+ step.
    sequences : list of array-like
        State-label sequences.
    lag_times : sequence of int
        Lag times; the first is the model under test.
    n_metastable_sets : int
        Number of PCCA+ metastable sets.
    include_lag0 : bool, default=True
        Include lag time 0 (the identity).
    err_est : bool, default=False
        Passed to deeptime.

    Returns
    -------
    test : deeptime.util.validation.ChapmanKolmogorovTest
        With ``lagtimes``, ``predictions`` and ``estimates`` (arrays of shape
        ``(n_lag_times, n_metastable_sets, n_metastable_sets)``) and a
        ``plot`` helper (``deeptime.plots.plot_ck_test``).
    """
    _deeptime()
    lag_times = [int(t) for t in lag_times]
    if len(lag_times) < 2:
        raise ValueError('ck_test needs at least two lag times')
    if sorted(lag_times) != lag_times or lag_times[0] < 1:
        raise ValueError('lag_times must be increasing positive integers')
    encoded, _ = _encode(sequences)
    models = []
    for lag in lag_times:
        msm = clone(estimator).set_params(lag_time=lag)
        if 'verbose' in msm.get_params():
            msm.set_params(verbose=False)
        models.append(to_deeptime(msm.fit(encoded)))
    return models[0].ck_test(models, n_metastable_sets, include_lag0=include_lag0,
                             err_est=err_est)
