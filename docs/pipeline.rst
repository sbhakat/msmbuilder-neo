.. _pipeline:
.. currentmodule:: msmbuilder.pipeline

Pipelines from a YAML file
==========================

A whole analysis, from trajectories to a validated Markov state model, can
be described in one YAML file and run with one command::

    msmb Pipeline --template > pipeline.yaml   # an example to edit
    msmb Pipeline pipeline.yaml

or from Python with :func:`run_pipeline`. Rerunning it reproduces the same
model, and every run records what was done.

The pipeline file
-----------------

.. code-block:: yaml

    # MSMBuilder pipeline: run with  msmb Pipeline pipeline.yaml
    version: 1
    random_state: 42            # seed for every step with a random_state parameter
    output: pipeline_output     # relative paths are relative to this file

    data:
      trajectories: "trajectories/*.xtc"   # glob, or a list of globs/files
      topology: topology.pdb
      stride: 1
      # features: features.h5    # or start from an existing dataset (no featurizer step)

    steps:
      - name: features
        featurizer: DihedralFeaturizer
        params: {types: [phi, psi]}
      - name: tica
        estimator: tICA
        params: {n_components: 4, lag_time: 10, kinetic_mapping: true}
      - name: clusters
        estimator: MiniBatchKMeans
        params: {n_clusters: 100}
      - name: msm
        estimator: MarkovStateModel
        params: {lag_time: 10, verbose: false}

    validation:
      implied_timescales: {lag_times: [1, 2, 5, 10, 20, 50], n_timescales: 5}
      # ck_test: {lag_times: [10, 20, 40, 80], n_metastable_sets: 3}   # needs deeptime

``data`` gives either trajectory files (a glob or a list, read with MDTraj
in chunks, with an optional ``topology``, ``stride`` and ``chunk``) and a
first ``featurizer`` step, or an existing ``features`` dataset.

Each step names an MSMBuilder class: a featurizer, or any estimator from
``preprocessing``, ``decomposition``, ``cluster``, ``msm``, ``lumping`` or
``hmm`` (or a dotted path inside ``msmbuilder``), with its parameters. Each
step is fitted to the output of the previous one; steps that transform their
data save it, while a Markov state model is only fitted (set
``transform: true`` to also save its output). Unknown keys are errors, so
typos are caught.

``random_state`` seeds every step that has a ``random_state`` parameter and
does not set one. ``validation`` computes implied timescales over a range of
lag times and, with deeptime installed, a Chapman-Kolmogorov test, using the
pipeline's Markov state model (or a default one on the final labels).

Outputs and provenance
----------------------

The output directory holds, for each step, the fitted object
(``<step>.pkl``) and, for steps that transform their data, the output
dataset (``<step>.h5``); plus ``implied_timescales.csv``, ``ck_test.npz``
and ``provenance.yaml``, which records:

* the configuration and its checksum;
* each step's class and *complete* parameters (defaults included), and its
  run time;
* the Python, platform and package versions (MSMBuilder, NumPy, SciPy,
  scikit-learn, MDTraj, Numba, pandas, h5py, and PyTorch, deeptime,
  MDAnalysis or Dask when used);
* SHA-256 checksums of the input files (or of the input dataset's content)
  and of every step's output, the Markov state model's slowest timescales
  and transition-matrix checksum, and one ``fingerprint`` of all results.

Rerunning and verifying
-----------------------

A rerun reuses every step whose input data, class and parameters are
unchanged, after checking its saved output against the recorded checksum,
and recomputes the rest: changing the MSM lag time refits only the MSM,
while changing tICA refits tICA and everything after it. ``--force``
recomputes everything.

``msmb Pipeline --verify pipeline.yaml`` reruns the pipeline from scratch in
a temporary directory and compares every output with ``provenance.yaml``; it
reports any difference and exits with status 1. Results are expected to be
bit-for-bit identical with the same package versions on the same platform;
other versions or platforms can change the last digits, which ``--verify``
will show.

API
---

.. autosummary::

    run_pipeline
    verify_pipeline
    load_config
    PipelineResult

.. vim: tw=75
