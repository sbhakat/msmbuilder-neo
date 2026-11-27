
import numpy as np
from sklearn.utils import check_random_state

from ..msm import MarkovStateModel


class PCCA(MarkovStateModel):
    """Perron Cluster Cluster Analysis (PCCA) for coarse-graining (lumping)
    microstates into macrostates.

    Parameters
    ----------
    n_macrostates : int
        The desired number of macrostates in the lumped model.
    kwargs : optional
        Additional keyword arguments to be passed to MarkovStateModel.  See
        msmbuilder.msm.MarkovStateModel for possible options.

    Attributes
    ----------
    microstate_mapping_ : array, shape (n_states_,)
        Macrostate of each microstate (internal MSM index); also available
        as ``metastable_assignments_``.
    memberships_ : array, shape (n_states_, n_macrostates)
        Membership of each microstate in each macrostate (one-hot for PCCA,
        fuzzy for PCCA+); rows sum to one.
    metastable_sets_ : list of arrays
        Microstates (internal MSM indices) assigned to each macrostate.
    metastable_label_sets_ : list of arrays
        The same, as the microstate labels of the input sequences.
    metastable_distributions_ : array, shape (n_macrostates, n_states_)
        Probability of each microstate within each macrostate,
        ``pi_i * m_ik / sum_j pi_j * m_jk``; rows sum to one.
    coarse_grained_populations_ : array, shape (n_macrostates,)
        Equilibrium population of each macrostate, ``memberships_.T @ pi``.
    coarse_grained_transmat_ : array, shape (n_macrostates, n_macrostates)
        Macrostate transition matrix ``(M^T M)^-1 M^T T M`` of the
        memberships ``M``, as PyEMMA and deeptime define it.

    Notes
    -----
    PCCA is a subclass of MarkovStateModel.  However, the MSM properties
    and attributes on PCCA refer to the MICROSTATE properties--e.g.
    pcca.transmat_ is the microstate transition matrix. The macrostate
    transition matrix is ``coarse_grained_transmat_``; alternatively, fit a
    new MarkovStateModel on the output (assignments) of PCCA().

    Use :meth:`draw_macrostate_samples` to pick representative frames of
    each macrostate, and :func:`msmbuilder.lumping.save_macrostate_samples`
    to write them as structures (e.g. one PDB file per macrostate).
    """

    def __init__(self, n_macrostates, pcca_tolerance=1e-5, **kwargs):
        self.n_macrostates = n_macrostates
        self.pcca_tolerance = pcca_tolerance
        super().__init__(**kwargs)

    def fit(self, sequences, y=None):
        """Fit a PCCA lumping model using a sequence of cluster assignments.

        Parameters
        ----------
        sequences : list(np.ndarray(dtype='int'))
            List of arrays of cluster assignments
        y : None
            Unused, present for sklearn compatibility only.

        Returns
        -------
        self
        """
        super().fit(sequences, y=y)
        self._do_lumping()
        return self

    def _do_lumping(self):
        """Do the PCCA lumping.

        Notes
        -------
        1.  Iterate over the eigenvectors, starting with the slowest.
        2.  Calculate the spread of that eigenvector within each existing
            macrostate.
        3.  Pick the macrostate with the largest eigenvector spread.
        4.  Split the macrostate based on the sign of the eigenvector.
        """

        # Extract non-perron eigenvectors
        right_eigenvectors = self.right_eigenvectors_[:, 1:]

        assert self.n_states_ > 0
        microstate_mapping = np.zeros(self.n_states_, dtype=int)

        def spread(x):
            return x.max() - x.min()

        for i in range(self.n_macrostates - 1):
            v = right_eigenvectors[:, i]
            all_spreads = np.array([spread(v[microstate_mapping == k])
                                    for k in range(i + 1)])
            state_to_split = np.argmax(all_spreads)
            inds = ((microstate_mapping == state_to_split) &
                    (v >= self.pcca_tolerance))
            microstate_mapping[inds] = i + 1

        self.microstate_mapping_ = microstate_mapping
        memberships = np.zeros((self.n_states_, self.n_macrostates))
        memberships[np.arange(self.n_states_), microstate_mapping] = 1.0
        self._finish_lumping(memberships)

    def _finish_lumping(self, memberships):
        """Macrostate quantities derived from the memberships (as PyEMMA's
        and deeptime's PCCA define them)."""
        # PCCA+ memberships can be negative at round-off level (~1e-14):
        # clip and renormalize, as deeptime does
        chi = np.clip(np.asarray(memberships, dtype=float), 0.0, None)
        chi /= chi.sum(axis=1, keepdims=True)
        pi = np.asarray(self.populations_, dtype=float)
        T = np.asarray(self.transmat_, dtype=float)
        self.memberships_ = chi
        pi_coarse = chi.T @ pi
        self.coarse_grained_populations_ = pi_coarse
        with np.errstate(divide='ignore', invalid='ignore'):
            dist = (chi * pi[:, None]).T / pi_coarse[:, None]
        self.metastable_distributions_ = np.nan_to_num(dist)
        self.coarse_grained_transmat_ = np.linalg.pinv(chi.T @ chi) @ (chi.T @ T @ chi)
        self.metastable_sets_ = [np.flatnonzero(self.microstate_mapping_ == k)
                                 for k in range(self.n_macrostates)]
        labels = np.empty(self.n_states_, dtype=object)
        for label, i in self.mapping_.items():
            labels[i] = label
        try:
            labels = np.array(labels.tolist())
        except (TypeError, ValueError):
            pass
        self.metastable_label_sets_ = [labels[s] for s in self.metastable_sets_]

    @property
    def metastable_assignments_(self):
        """Macrostate of each microstate (same as ``microstate_mapping_``)."""
        return self.microstate_mapping_

    def _internal_sequences(self, sequences):
        """Microstate label sequences -> internal MSM indices (-1 for
        labels that are not in the model)."""
        from ..utils import list_of_1d
        out = []
        if isinstance(sequences, (list, tuple)) and len(sequences) == 0:
            return out
        keys = list(self.mapping_)
        int_keys = all(isinstance(k, (int, np.integer)) and k >= 0 for k in keys)
        if int_keys:
            lut = np.full(max(keys) + 1, -1, dtype=np.int64)
            lut[np.asarray(keys, dtype=np.int64)] = list(self.mapping_.values())
        for seq in list_of_1d(sequences):
            seq = np.asarray(seq)
            if int_keys and seq.dtype.kind in 'iu':
                idx = np.full(len(seq), -1, dtype=np.int64)
                ok = (seq >= 0) & (seq < len(lut))
                idx[ok] = lut[seq[ok]]
            else:
                idx = np.array([self.mapping_.get(x.item() if hasattr(x, 'item') else x, -1)
                                for x in seq], dtype=np.int64)
            out.append(idx)
        return out

    def draw_macrostate_samples(self, sequences, n_samples, weighting='membership',
                                random_state=None):
        """Pick frames that represent each macrostate.

        Parameters
        ----------
        sequences : list of array-like
            Microstate label sequences (e.g. the cluster labels the MSM was
            fitted to).
        n_samples : int
            Frames to draw per macrostate (with replacement, so a frame can
            appear more than once).
        weighting : {'membership', 'assignment'}, default='membership'
            'membership' (PyEMMA's ``sample_by_distributions``): draw a
            microstate from ``metastable_distributions_[k]``, then a frame
            of that microstate, so frames follow the equilibrium distribution
            within the macrostate. 'assignment': draw uniformly from the
            frames whose microstate is assigned to the macrostate.
        random_state : int, RandomState or None

        Returns
        -------
        samples : list of arrays, one per macrostate
            ``samples[k]`` has shape ``(n_samples, 2)``: (sequence index,
            frame index) pairs; shape ``(0, 2)`` if no frame can represent
            macrostate ``k``.
        """
        if not hasattr(self, 'memberships_'):
            raise ValueError('This %s is not fitted yet.' % type(self).__name__)
        if weighting not in ('membership', 'assignment'):
            raise ValueError("weighting must be 'membership' or 'assignment', got %r"
                             % weighting)
        n_samples = int(n_samples)
        if n_samples < 1:
            raise ValueError('n_samples must be at least 1')
        rs = check_random_state(random_state)
        idx = self._internal_sequences(sequences)
        if not idx or sum(len(s) for s in idx) == 0:
            return [np.zeros((0, 2), dtype=np.int64) for _ in range(self.n_macrostates)]
        traj = np.concatenate([np.full(len(s), i, dtype=np.int64) for i, s in enumerate(idx)])
        frame = np.concatenate([np.arange(len(s), dtype=np.int64) for s in idx])
        state = np.concatenate(idx) if idx else np.zeros(0, dtype=np.int64)
        keep = state >= 0
        traj, frame, state = traj[keep], frame[keep], state[keep]
        order = np.argsort(state, kind='stable')
        traj, frame, state = traj[order], frame[order], state[order]
        counts = np.bincount(state, minlength=self.n_states_)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

        samples = []
        for k in range(self.n_macrostates):
            if weighting == 'membership':
                p = self.metastable_distributions_[k] * (counts > 0)
            else:
                p = (self.microstate_mapping_ == k) * counts.astype(float)
            if p.sum() <= 0:
                samples.append(np.zeros((0, 2), dtype=np.int64))
                continue
            micro = rs.choice(self.n_states_, size=n_samples, p=p / p.sum())
            pick = starts[micro] + (rs.random_sample(n_samples) * counts[micro]).astype(np.int64)
            samples.append(np.column_stack([traj[pick], frame[pick]]))
        return samples

    def partial_transform(self, sequence, mode='clip'):
        trimmed_sequence = super().partial_transform(sequence, mode)
        if mode == 'clip':
            return [self.microstate_mapping_[seq] for seq in trimmed_sequence]
        elif mode == 'fill':
            def nan_get(x):
                try:
                    x = int(x)
                    return self.microstate_mapping_[x]
                except ValueError:
                    return np.nan

            return np.asarray([nan_get(x) for x in trimmed_sequence])
        else:
            raise ValueError

    @classmethod
    def from_msm(cls, msm, n_macrostates, **kwargs):
        """Create and fit lumped model from pre-existing MSM.

        Parameters
        ----------
        msm : MarkovStateModel
            The input microstate msm to use.
        n_macrostates : int
            The number of macrostates
        **kwargs
            Other parameters of the lumping (e.g. ``random_state`` or
            ``objective_function`` for PCCA+).

        Returns
        -------
        lumper : cls
            The fit PCCA(+) object.
        """
        params = msm.get_params()
        params.update(kwargs)
        lumper = cls(n_macrostates, **params)

        lumper.transmat_ = msm.transmat_
        lumper.populations_ = msm.populations_
        lumper.mapping_ = msm.mapping_
        lumper.countsmat_ = msm.countsmat_
        lumper.n_states_ = msm.n_states_

        lumper._do_lumping()

        return lumper
