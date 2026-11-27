.. _msm:
.. currentmodule:: msmbuilder.msm

Markov state models (MSMs)
==========================

Markov state models (MSMs) are a class of models for modeling the
long-timescale dynamics of molecular systems. They model the dynamics of a
system as a series of memoryless, probabilistic jumps between a set of
states. Practically, the model consists of (1) a set of conformational
states, and (2) a matrix of transition probabilities between each pair of
states.

In MSMBuilder, you can use :class:`MarkovStateModel` to build MSMs from
"labeled" trajectories -- that is, sequences of integers that are the
result of :ref:`clustering<cluster>`.

Algorithms
----------

.. autosummary::

    MarkovStateModel
    BayesianMarkovStateModel
    OOMReweightedMSM
    TRAM

Maximum Likelihood and Bayesian Estimation
------------------------------------------

There are two steps in constructing an MSM

#. Count the number of observed transitions between states. That is,
   construct :math:`\mathbf{C}` such that :math:`C_{ij}` is the number of
   observed transitions from state :math:`i` at time :math:`t` to state
   :math:`j` at time :math:`t+\tau`, summed over all times :math:`t`.

#. Estimate the transition probability matrix, :math:`\mathbf{T}`

   .. math ::
       T_{ij} = P( s_{t+\tau} = j | s_t = i)
    
   where :math:`S = (s_t)` is a trajectory in state-index space of length
   :math:`N`, and :math:`s_t \in \{1, \ldots, k\}` the state-index of the
   trajectory at time :math:`t`.

The probability that a given transition probability matrix would generate
some observed trajectory (the likelihood) is

.. math ::
  \mathcal{L}(\mathbf{T}) = P(S | \mathbf{T}) =
  \prod_{t=0}^{N-\tau} T_{s_t, s_{t+\tau}} = \prod_{i,j}^{k} T_{ij}^{C_{ij}}.

Assuming a prior distribution on :math:`T` of the form
:math:`P(T)=\prod_{ij} T_{ij}^{B_{ij}}`, we then have a posterior
distribution

.. math ::
   P(\mathbf{T} | S) \propto \prod_{i,j}^{k} T_{ij}^{B_{ij} + C_{ij}}.


MSMBuilder implements two MSM estimators.

- :class:`MarkovStateModel` performs maximum likelihood estimation.  It
  estimates a single transition matrix, :math:`\mathbf{T}`, to maximimize
  :math:`\mathcal{L}(\mathbf{T})`.

- :class:`BayesianMarkovStateModel` uses Metropolis Markov chain Monte
  Carlo to (approximately) draw a sample of transition matrices from the
  posterior distribution :math:`P(\mathbf{T} | S)`. This sampler is
  described in Metzner et al. [#f5]_ This can be used to estimate the
  sampling uncertainty in functions of the transition matrix (e.g.
  relaxation timescales).

.. note::

   The uncertainty in the transition matrix (and functions of the
   transition matrix) that can be estimated from
   :class:`BayesianMarkovStateModel` do not **fully** account for all
   sources of error. In particular, the discretization induced by
   clustering produces a negative bias on  the eigenvalues of the
   transition matrix -- they asymptotically underestimate the eigenvalues
   of the propagator / transfer operator in the limit of infinite sampling.
   [#f6]_ See section 3D (Quantifying the discretization error) of Prinz et
   al. for more discussion on the discretization error. [#f1]_


Tradeoffs and Parameter Selection
---------------------------------

The most important tradeoff with MSMs is a `bias-variance dilemma
<http://en.wikipedia.org/wiki/Bias%E2%80%93variance_dilemma>`_ on the
number of states. We know analytically that the expected value of the
relaxation timescales is below the true value when using a finite number of
states, and that the magnitude of this bias decreases as the number of
states goes up. On the other hand, the statistical error in the MSM
(variance) goes up as the number of states increases with a fixed data set,
because there are fewer transitions (data) per element of the transition
probability matrix.

There are no existing algorithms in the MSM literature which fully balance
these competing sources of error in an automatic and practical way,
although some partially satisfactory algorithms are available. [#f3]_
[#f4]_

Macrostates and representative structures
-----------------------------------------

:class:`msmbuilder.lumping.PCCAPlus` (PCCA+) groups the microstates of a
fitted MSM into metastable macrostates, as ``pyemma.msm.PCCA`` does, and
:func:`msmbuilder.lumping.save_macrostate_samples` writes sampled
structures of each macrostate to disk, one multi-model PDB file per
macrostate, like PyEMMA's ``sample_by_distributions`` and ``save_trajs``:

.. code-block:: python

    from msmbuilder.lumping import PCCAPlus, save_macrostate_samples

    pcca = PCCAPlus.from_msm(msm, n_macrostates=4, method='deeptime')
    pcca.memberships_                 # fuzzy memberships (n_states, 4)
    pcca.metastable_assignments_      # crisp macrostate of each microstate
    pcca.coarse_grained_transmat_     # 4 x 4 macrostate transition matrix
    pcca.coarse_grained_populations_  # macrostate populations
    macro_labels = pcca.transform(labels)

    save_macrostate_samples(pcca, labels, trajectory_files, n_samples=100,
                            top='topology.pdb', stride=1, superpose='backbone',
                            out_dir='macrostates', random_state=0)
    # -> macrostates/macrostate_0.pdb ... macrostate_3.pdb and
    #    macrostates/macrostate_samples.csv (trajectory, frame, microstate
    #    and membership of every structure)

Frames are drawn as in PyEMMA: a microstate from the macrostate's
``metastable_distributions_`` (equilibrium-weighted memberships), then a
frame of that microstate (``weighting='assignment'`` instead samples
uniformly from the frames crisply assigned to the macrostate). Pass the
``stride`` the trajectories were featurized with, so that frames map back to
the trajectory files.

``method='deeptime'`` runs PyEMMA's PCCA+ algorithm (through deeptime,
``pip install msmbuilder-neo[deeptime]``) and reproduces PyEMMA's
memberships. The default ``method='msmbuilder'`` (seed it with
``random_state``) finds equally metastable macrostates, but its optimizer
can assign the states near barriers differently, especially when asking for
more macrostates than the system has clearly separated basins.

Methods from deeptime (optional)
--------------------------------

With deeptime installed (``pip install msmbuilder-neo[deeptime]``), these
deeptime methods [#f7]_ work on MSMBuilder state-label sequences and return
MSMBuilder objects:

* :class:`TRAM` [#f8]_ combines biased simulations (umbrella sampling,
  replica exchange, ...) into unbiased free energies of the states
  (``markov_state_energies_``, ``populations_``), free energies of each
  thermodynamic state, and one :class:`MarkovStateModel` per thermodynamic
  state (``msms_``). ``fit(sequences, bias_matrices, ttrajs=None)`` takes the
  bias energy (in kT) of every frame in every thermodynamic state;
  ``pmf`` and ``sample_weights`` reweight frames to the unbiased (or any)
  ensemble.
* :class:`OOMReweightedMSM` [#f9]_ is a :class:`MarkovStateModel` corrected
  for simulations started out of equilibrium and analysed on a
  not-quite-Markovian discretization, the usual situation for many short
  trajectories.
* :func:`ck_test` runs the Chapman-Kolmogorov test: models estimated at
  longer lag times are compared with the predictions of the model at the
  shortest one, on PCCA+ metastable sets.
* :func:`to_deeptime` and :func:`from_deeptime` convert fitted models, to
  use the rest of deeptime's analysis (or bring its estimators' results
  back). Note that deeptime orders eigenvalues by modulus and MSMBuilder by
  value, so a negative eigenvalue can shift deeptime's list of timescales.

.. code-block:: python

    from msmbuilder.msm import MarkovStateModel, TRAM, ck_test

    tram = TRAM(lag_time=10).fit(labels, bias_matrices)
    print(tram.populations_)           # unbiased state populations
    print(tram.msms_[0].timescales_)   # kinetics in thermodynamic state 0

    test = ck_test(MarkovStateModel(), labels, [10, 20, 40, 80], 3)
    test.plot()                        # needs matplotlib

.. todo: talk about lag time

.. todo: analysis page


References
----------
.. [#f1] Prinz, J.-H., et al. `Markov models of molecular kinetics: Generation and validation <http://dx.doi.org/10.1063/1.3565032>`_ J. Chem. Phys. 134.17 (2011): 174105.
.. [#f2] Pande, V. S., K. A. Beauchamp, and G. R. Bowman. `Everything you wanted to know about Markov State Models but were afraid to ask <http://dx.doi.org/10.1016/j.ymeth.2010.06.002>`_ Methods 52.1 (2010): 99-105.
.. [#f3] McGibbon, R. T., C. R. Schwantes, and Vijay S. Pande. `Statistical Model Selection for Markov Models of Biomolecular Dynamics. <http://dx.doi.org/10.1021/jp411822r>`_ J. Phys. Chem. B (2014).
.. [#f4] Kellogg, E. H., O. F. Lange, and D. Baker. `Evaluation and optimization of discrete state models of protein folding. <http://dx.doi.org//10.1021/jp3044303>`_ J. Phys. Chem. B 116.37 (2012): 11405-11413.
.. [#f5] Metzner, P., F. Noe, and C. Schutte. `Estimating the sampling error: Distribution of transition matrices and functions of transition matrices for given trajectory data. <http://journals.aps.org/pre/abstract/10.1103/PhysRevE.80.021106>`_ Phys. Rev. E 80.2 (2009): 021106.
.. [#f6] Nuske, F., et al. `Variational approach to molecular kinetics. <http://pubs.acs.org/doi/abs/10.1021/ct4009156>`_ J. Chem. Theory Comput.10.4 (2014): 1739-1752.

.. [#f7] Hoffmann, M., et al. `Deeptime: a Python library for machine learning dynamical models from time series data. <https://doi.org/10.1088/2632-2153/ac3de0>`_ Mach. Learn.: Sci. Technol. 3 (2022): 015009.
.. [#f8] Wu, H., et al. `Multiensemble Markov models of molecular thermodynamics and kinetics. <https://doi.org/10.1073/pnas.1525092113>`_ Proc. Natl. Acad. Sci. USA 113 (2016): E3221-E3230.
.. [#f9] Nuske, F., et al. `Markov state models from short non-equilibrium simulations - Analysis and correction of estimation bias. <https://doi.org/10.1063/1.4976518>`_ J. Chem. Phys. 146 (2017): 094104.

.. vim: tw=75
