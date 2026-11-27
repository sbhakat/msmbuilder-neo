.. _migration:

Migration guide
===============

For users of MSMBuilder 3.8 and msmbuilder2022.

Most code runs unchanged: the package is still imported as ``msmbuilder``,
the ``msmb`` command keeps all its subcommands, and no public class,
function or parameter was removed or renamed. This page lists everything
that *did* change, from comparing the public API of msmbuilder2022 with
MSMBuilder Neo and from the development history.

Installation
------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - MSMBuilder 3.8 / msmbuilder2022
     - MSMBuilder Neo
   * - Install
     - ``conda install -c omnia msmbuilder`` / build from source with a C
       compiler
     - ``pip install`` (pure Python; see :ref:`installation`)
   * - Python, NumPy
     - Python 3.9-3.12 (msmbuilder2022; older for 3.8), NumPy 1.x
     - Python 3.10-3.13, NumPy 1.24+ and NumPy 2
   * - Platforms
     - Linux, macOS, Windows
     - Linux and macOS (Windows not supported for now)
   * - Storage dependency
     - PyTables (``tables``)
     - h5py
   * - Bayesian rate-matrix sampling
     - pyhmc
     - pyhmc, installed separately (it is not maintained on PyPI)

Install MSMBuilder Neo in a fresh environment: it provides the same
``msmbuilder`` package and ``msmb`` command as the original.

Results that differ
-------------------

These differences were found by comparing MSMBuilder Neo with the
original compiled code (``benchmarks/*_parity.py`` in the repository):

* **Hidden Markov model log-likelihoods.** ``GaussianHMM`` and
  ``VonMisesHMM`` passed their start probabilities to the forward-backward
  code as if they were log-probabilities. With the fix, ``score()``,
  ``fit_logprob_`` and the log-probability returned by ``predict()`` shift by
  exactly ``n_trajectories * (log(1/n_states) - 1/n_states)``. Fitted means,
  variances, transition matrices, populations and Viterbi paths are
  unchanged.
* **RMSD of a frame with itself** is about 1e-4 nm instead of exactly 0 (RMSD
  now uses MDTraj's kernel; all other RMSDs match).
* **Bayesian MSM sampling.** Parallel MCMC chains of the Metzner sampler
  shared one state array; each chain now has its own, so samples differ
  from before (and are now correct).
* **Random seeds are honoured** where they were ignored before:
  ``VonMisesHMM(random_state=...)``, and the new ``random_state`` of
  ``PCCAPlus`` (its optimizer was unseeded).
* **PCCA+** results are unchanged by default. ``PCCAPlus(method='deeptime')``
  gives PyEMMA's memberships, which can assign barrier states differently.

Changed or added parameters
---------------------------

No parameter was removed or renamed. New, optional parameters (the defaults
keep the old behaviour):

* ``MarkovStateModel(score_method='gmrq')`` -- also ``'vamp1'`` and
  ``'vamp2'``; new method ``score_vamp``.
* ``PCCAPlus(random_state=None, method='msmbuilder')``; new attributes
  ``memberships_``, ``metastable_sets_``, ``metastable_label_sets_``,
  ``metastable_distributions_``, ``coarse_grained_transmat_``,
  ``coarse_grained_populations_``, also on ``PCCA``; new method
  ``draw_macrostate_samples``.
* ``PCCA.from_msm`` accepts further parameters for the lumper.
* HDF5 datasets: ``dataset(path, mode='a', fmt='hdf5')`` appends, and
  ``compression='gzip'`` compresses new arrays.

Files and data
--------------

* **HDF5 datasets** (``.h5`` written by ``msmb`` commands or
  ``dataset(..., fmt='hdf5')``) keep their layout: files from earlier
  versions open unchanged, and files written by MSMBuilder Neo open in them.
* **Project metadata** is saved as ``meta.parquet`` (with pyarrow) or
  ``meta.csv`` instead of the pickle ``meta.pandas.pickl``. ``load_meta()``
  still reads old pickles, and ``save_meta(meta, 'meta.pandas.pickl')``
  still writes one.
* **Pickled estimators** saved by msmbuilder2022 generally load, but
  pickles are tied to the code that wrote them; refitting is the safe path.

Removed internals
-----------------

Only internal modules were removed, as part of replacing compiled code:
``msmbuilder.src.cy_blas``, ``msmbuilder.src.triu_utils``, the Cython
modules behind ``msmbuilder.libdistance``, ``msmbuilder.msm._ratematrix_*``
and the bundled ``msmbuilder.utils.progressbar`` (now ``tqdm``). The public
functions they served (for example ``msmbuilder.libdistance.cdist``,
``pdist`` and ``assign_nearest``) keep their names and arguments.

Example notebooks
-----------------

The example notebooks no longer use MSMExplorer (unmaintained) or OpenMM:
plots use matplotlib directly, and the toy simulation in *tICA vs. PCA* is a
short NumPy Langevin integrator. Notebooks that used
``sklearn.cross_validation`` or ``sklearn.grid_search`` now use
``sklearn.model_selection``.

Coming from PyEMMA
------------------

PyEMMA is no longer maintained. Rough equivalents in MSMBuilder Neo:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - PyEMMA
     - MSMBuilder Neo
   * - ``pyemma.coordinates.featurizer`` + ``load``/``source``
     - :ref:`featurizers <featurization>` on MDTraj trajectories, or
       ``msmbuilder.io.mdanalysis.featurize_mdanalysis``
   * - ``coordinates.tica``, ``coordinates.vamp``
     - :class:`~msmbuilder.decomposition.tICA`,
       :class:`~msmbuilder.decomposition.VAMP`
   * - ``coordinates.cluster_kmeans``
     - :class:`~msmbuilder.cluster.KMeans`,
       :class:`~msmbuilder.cluster.MiniBatchKMeans`
   * - ``msm.estimate_markov_model``
     - :class:`~msmbuilder.msm.MarkovStateModel`
   * - ``msm.bayesian_markov_model``
     - :class:`~msmbuilder.msm.BayesianMarkovStateModel`
   * - ``msm.its``, ``mplt.plot_implied_timescales``
     - :func:`~msmbuilder.msm.implied_timescales`
   * - ``msm.cktest``
     - :func:`~msmbuilder.msm.ck_test`
   * - ``msm.pcca``, ``metastable_memberships``, ``metastable_sets``
     - :class:`~msmbuilder.lumping.PCCAPlus` (``method='deeptime'``)
   * - ``sample_by_distributions`` + ``save_trajs``
     - :func:`~msmbuilder.lumping.save_macrostate_samples`
   * - ``msm.mfpt``, ``msm.tpt``
     - :func:`~msmbuilder.tpt.mfpts`, :func:`~msmbuilder.tpt.net_fluxes`,
       :func:`~msmbuilder.tpt.committors` (on sets of states)
   * - ``thermo.tram``
     - :class:`~msmbuilder.msm.TRAM`
