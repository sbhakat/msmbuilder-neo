.. _msmbuilder:

:html_theme.sidebar_secondary.remove:

.. raw:: html

   <div class="msmb-hero">

MSMBuilder Neo
==============

.. rst-class:: msmb-tagline

   Statistical models for biomolecular dynamics: featurize simulations, find
   their slow processes, and build validated Markov state models -- on
   current Python, NumPy 2 and every major platform.

.. raw:: html

   </div>

.. code-block:: bash

   pip install msmbuilder-neo
   msmb Pipeline --template > pipeline.yaml    # describe your analysis
   msmb Pipeline pipeline.yaml                 # run it, with provenance

MSMBuilder Neo is a modernized fork of `MSMBuilder
<https://github.com/msmbuilder/msmbuilder>`_ (via `msmbuilder2022
<https://github.com/msmbuilder/msmbuilder2022>`_). It keeps the MSMBuilder
API -- ``import msmbuilder`` and the ``msmb`` command line work as before --
and adds the methods of current Markov state model papers: VAMP scoring,
VAMPnets and other deep collective variables, deeptime's TRAM, PyEMMA-style
PCCA+ macrostates, and reproducible YAML pipelines.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket` Getting started
      :link: getting_started
      :link-type: ref

      Install MSMBuilder Neo and build your first Markov state model, from
      trajectories to macrostates, in one notebook.

   .. grid-item-card:: :octicon:`book` User guide
      :link: user_guide
      :link-type: ref

      Featurization, tICA and VAMP, clustering, Markov state models,
      macrostates, transition path theory, and the theory behind them.

   .. grid-item-card:: :octicon:`beaker` Examples
      :link: examples
      :link-type: ref

      Notebooks, executed every time the documentation is built, so every
      output you see comes from the current code.

   .. grid-item-card:: :octicon:`code` API reference
      :link: api
      :link-type: ref

      Every estimator, featurizer and function, with parameters and
      attributes.

   .. grid-item-card:: :octicon:`sparkle-fill` What's new in Neo
      :link: whats_new
      :link-type: ref

      Pure Python on Numba, NumPy 2, VAMP and deep learning, deeptime
      adapters, h5py storage, pipelines with provenance.

   .. grid-item-card:: :octicon:`arrow-switch` Migrating
      :link: migration
      :link-type: ref

      Coming from MSMBuilder 3.8 or msmbuilder2022: what changed, what was
      removed, and the few results that differ.

What it does
------------

.. list-table::
   :widths: 25 75
   :class: msmb-features

   * - **Featurize**
     - Dihedrals, contacts, distances, RMSD, SASA, ligand features and more
       on MDTraj trajectories -- or any format MDAnalysis reads, in
       memory-bounded chunks.
   * - **Reduce**
     - tICA, sparse and kernel tICA, PCA, VAMP; VAMPnets, state-free
       reversible VAMPnets, time-lagged autoencoders, DeepTICA and DeepLDA
       (optional PyTorch), exportable to PLUMED.
   * - **Cluster**
     - k-means, k-centers, k-medoids, regular spatial, landmark and
       scikit-learn clusterers, all on lists of trajectories.
   * - **Model**
     - Maximum-likelihood, Bayesian and continuous-time Markov state models,
       hidden Markov models, TRAM for biased simulations, and
       OOM-reweighted MSMs for short off-equilibrium runs.
   * - **Validate**
     - GMRQ and VAMP scores with cross-validation, implied timescales,
       Chapman-Kolmogorov tests.
   * - **Interpret**
     - PCCA and PCCA+ macrostates with representative structures written
       as PDB files, transition path theory (fluxes, committors, pathways,
       mean first passage times).
   * - **Reproduce**
     - One YAML file runs the whole analysis and records parameters,
       package versions and checksums; ``msmb Pipeline --verify`` checks a
       rerun reproduces it.

Citing
------

If you use MSMBuilder Neo in published work, please cite the MSMBuilder
papers listed under :ref:`publications`, and the papers of the methods you
use (each page lists them).

MSMBuilder Neo is licensed under the GNU LGPL, version 2.1 or later.
Development happens on `GitHub <https://github.com/sbhakat/msmbuilder-neo>`_.

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started
   user_guide
   examples/index
   api
   whats_new
   migration
   publications
   contributing
