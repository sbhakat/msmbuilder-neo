.. _whats_new:

What's new
==========

MSMBuilder Neo starts from msmbuilder2022 (itself based on MSMBuilder 3.8)
and modernizes it in six steps. The :ref:`migration guide <migration>` lists
what this means for existing code.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Installs everywhere, pure Python

      * Python 3.10 to 3.13, NumPy 2, current SciPy, scikit-learn and pandas.
      * All Cython, C and C++ code replaced by `Numba
        <https://numba.pydata.org>`_ kernels (results checked against the
        old code: 363 of 364 parity checks identical) -- one universal
        wheel, no compiler needed.
      * Tested on Linux x86_64 and aarch64 and macOS (Apple Silicon and
        Intel) on every change.

   .. grid-item-card:: Model selection with VAMP

      * :class:`~msmbuilder.decomposition.VAMP`, the variational approach for
        Markov processes, with VAMP-1, VAMP-2 and VAMP-E scores.
      * ``MarkovStateModel(score_method='vamp2')`` and ``score_vamp`` for
        cross-validating Markov state models.

   .. grid-item-card:: Deep collective variables (optional PyTorch)

      * VAMPnets, state-free reversible VAMPnets (SRV), time-lagged
        autoencoders.
      * DeepTICA and DeepLDA, following `mlcolvar
        <https://mlcolvar.readthedocs.io>`_, including training on
        reweighted biased simulations.
      * Export any of them to TorchScript for PLUMED's ``PYTORCH_MODEL``.

   .. grid-item-card:: Methods from deeptime (optional)

      * TRAM for umbrella sampling and other multi-ensemble data.
      * OOM-reweighted MSMs for short off-equilibrium simulations.
      * Chapman-Kolmogorov tests, and conversion of models to and from
        deeptime.

   .. grid-item-card:: Macrostates like PyEMMA

      * PCCA+ memberships, metastable sets and distributions, coarse-grained
        transition matrix and populations; ``method='deeptime'`` reproduces
        ``pyemma.msm.PCCA``.
      * :func:`~msmbuilder.lumping.save_macrostate_samples` writes sampled
        structures of every macrostate as PDB files.

   .. grid-item-card:: Data, pipelines and provenance

      * Datasets written with h5py (old PyTables files still open), lazy
        Dask loading, Parquet or CSV project metadata.
      * Read any MDAnalysis trajectory format, featurized in chunks.
      * ``msmb Pipeline``: a whole analysis from one YAML file, with
        parameters, package versions and checksums recorded, and
        ``--verify`` to check a rerun reproduces it.

Development of MSMBuilder Neo is tracked on GitHub; each step was a pull
request with its own tests: `pull requests
<https://github.com/sbhakat/msmbuilder-neo/pulls?q=is%3Apr+is%3Amerged>`_.
The changes of the original MSMBuilder releases are in the :ref:`changelog`.

.. toctree::
   :hidden:

   changelog
