.. _gmrq:
.. _model_selection:
.. currentmodule:: msmbuilder

Model selection: GMRQ and VAMP scores
=====================================

Building a Markov state model involves many choices: which features, how many
tICA components, how many clusters, which clustering algorithm. Different
choices give different models, and they cannot all be right. MSMBuilder gives
every model a single **score** that measures how well it captures the slow
dynamics of the system, so these choices can be made objectively, with
cross-validation, instead of by eye.

Two families of scores are available:

* the **GMRQ** (generalized matrix Rayleigh quotient) [#f1]_, the classic
  MSMBuilder score, for reversible (equilibrium) dynamics;
* the **VAMP scores** (VAMP-1, VAMP-2 and, for features, VAMP-E) [#f2]_, from
  the variational approach for Markov processes, which do not assume
  reversibility. VAMP-2 is also the score used by PyEMMA and deeptime.

Both are computed the same way and used the same way; the sections below
explain the idea, which score to use where, and a complete recipe.

The idea in one paragraph
-------------------------

The slow processes of a molecular system -- folding, binding, conformational
change -- appear as the eigenfunctions of its transfer operator with
eigenvalues close to one. A Markov state model, or a tICA or VAMP projection,
can only *approximate* these eigenfunctions with the features and states it
is given, and the **variational principle** guarantees that any approximation
can only underestimate the true eigenvalues (or, for VAMP, singular values).
So among models built from the same data, the one whose leading eigenvalues
are larger describes the slow dynamics better. GMRQ and VAMP turn this into a
single number per model: higher is better.

Why cross-validation is needed
------------------------------

Scored on the data it was fitted to, a model with more states always looks
better, because it can fit statistical noise. The remedy is to score each
model on data it has **not** seen:

1. split the trajectories into a training set and a test set,
2. fit the model on the training trajectories,
3. score it on the test trajectories,
4. repeat over several splits (k-fold cross-validation) and average.

The model with the best *test* score generalizes best. Always split by
**trajectory**, never by frame: frames of one trajectory are correlated.
MSMBuilder estimators take lists of trajectories, so scikit-learn's ``KFold``
and ``GridSearchCV`` split by trajectory automatically.

Which score to use
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - GMRQ
     - VAMP-2 (or VAMP-1)
   * - Assumes
     - reversible, equilibrium dynamics
     - nothing beyond Markovian dynamics
   * - Good for
     - equilibrium simulations; reproducing earlier MSMBuilder analyses
     - any data, including non-equilibrium and driven dynamics; comparing
       with PyEMMA and deeptime results
   * - For reversible MSMs
     - VAMP-1 equals GMRQ on the training data. On test data the scores
       agree on clear differences but can rank close models differently
     -

If in doubt, use **VAMP-2**: it is valid in more situations and is the most
widely used score today.

Where the scores are
--------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Estimator
     - Score
   * - :class:`~msmbuilder.msm.MarkovStateModel`
     - ``score()`` returns the GMRQ by default; with
       ``MarkovStateModel(score_method='vamp2')`` (or ``'vamp1'``) it returns
       that VAMP score. :meth:`~msmbuilder.msm.MarkovStateModel.score_vamp`
       computes a VAMP score directly.
   * - :class:`~msmbuilder.decomposition.tICA`
     - ``score()``: GMRQ of the tICA components.
   * - :class:`~msmbuilder.decomposition.VAMP`
     - ``score()``: VAMP-2 by default; ``VAMP(score_method='VAMP1')`` or
       ``'VAMPE'`` for the others.
   * - :class:`~msmbuilder.msm.ContinuousTimeMSM`
     - ``score()``: GMRQ.
   * - VAMPnet, SRV, TimeLaggedAutoencoder, DeepTICA
     - ``score()``: VAMP-2 of the learned coordinates (see
       :ref:`decomposition`).
   * - a scikit-learn ``Pipeline``
     - ``score()`` of its last step, applied to the transformed data -- so a
       whole featurize-tICA-cluster-MSM pipeline can be scored at once.

Recipe: choose the number of states
-----------------------------------

The pipeline below goes from dihedral features to an MSM. ``GridSearchCV``
fits it for each number of clusters on four fifths of the trajectories and
scores it on the remaining fifth, five times, with the VAMP-2 score of the
MSM:

.. code-block:: python

    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import Pipeline

    from msmbuilder.cluster import MiniBatchKMeans
    from msmbuilder.decomposition import tICA
    from msmbuilder.featurizer import DihedralFeaturizer
    from msmbuilder.msm import MarkovStateModel

    pipeline = Pipeline([
        ('features', DihedralFeaturizer()),
        ('tica', tICA(n_components=4, lag_time=10)),
        ('cluster', MiniBatchKMeans(random_state=0)),
        ('msm', MarkovStateModel(lag_time=10, n_timescales=3,
                                 score_method='vamp2', verbose=False)),
    ])
    search = GridSearchCV(pipeline, {'cluster__n_clusters': [20, 50, 100, 200]},
                          cv=KFold(n_splits=5, shuffle=True, random_state=0))
    search.fit(trajectories)          # a list of MDTraj trajectories

    print(search.best_params_)
    print(search.cv_results_['mean_test_score'])   # one per number of clusters
    print(search.cv_results_['std_test_score'])    # spread across the 5 folds

On the alanine dipeptide example dataset this takes a few seconds; the test
score rises from 20 to 100 clusters and then levels off. Differences smaller
than the spread across folds are not meaningful: prefer the simplest model
among those within that spread.

The same search works for any other parameter of any step (for example
``'tica__n_components'`` or the featurizer), and with GMRQ by leaving out
``score_method``. The :doc:`GMRQ model selection example
<examples/notebooks/GMRQ-Model-Selection>` walks through a complete
cross-validation by hand and compares GMRQ with VAMP-2.

Things to watch
---------------

* **Score only a few slow modes.** The number of modes scored is
  ``n_timescales`` for MSMs (``dim`` for ``score_vamp``) and ``n_components``
  for tICA and VAMP. Keep it small (a handful) and fixed across the models
  you compare. If every mode of an MSM is scored, the score on test data
  depends only on the test data, not on the model, and cannot tell models
  apart.
* **Compare like with like.** Compare scores computed with the same score,
  the same number of modes and the same lag time. Scores at different lag
  times are not comparable (eigenvalues shrink as the lag time grows), so the
  lag time is chosen separately, from the implied timescales (see
  :ref:`msm`).
* **Know the scale.** MSM scores (GMRQ, VAMP-1, VAMP-2) and the ``VAMP``
  estimator include the constant (stationary) mode, which contributes 1:
  with ``n_timescales=k`` an MSM's score is at most ``k + 1``. ``tICA.score``
  leaves that mode out, so it is at most ``n_components``. The numbers
  matter only relative to other models scored the same way.
* **VAMP-E** is available for features (``VAMP(score_method='VAMPE')``) but
  not for MSMs, where it would depend on the amount of data.
* **Fix random seeds** (``random_state``) of the clustering so that
  differences in score come from the parameters, not from chance.

Algorithms
----------

.. autosummary::

    msm.MarkovStateModel.score
    msm.MarkovStateModel.score_vamp
    decomposition.tICA.score
    decomposition.VAMP.score
    msm.ContinuousTimeMSM.score

References
----------

.. [#f1] McGibbon, Robert T., and Vijay S. Pande. `Variational
   cross-validation of slow dynamical modes in molecular kinetics
   <https://doi.org/10.1063/1.4916292>`_ J. Chem. Phys. 142, 124105 (2015).
.. [#f2] Wu, Hao, and Frank Noé. `Variational approach for learning Markov
   processes from time series data <https://doi.org/10.1007/s00332-019-09567-y>`_
   J. Nonlinear Sci. 30, 23-66 (2020).

.. vim: tw=75
