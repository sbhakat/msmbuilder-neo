.. _featurization:
.. currentmodule:: msmbuilder.featurizer


Featurization
=============

Many algorithms require that the input data be vectors in a (euclidean)
vector space. This includes :class:`~msmbuilder.cluster.KMeans` clustering,
:class:`~msmbuilder.decomposition.tICA`, and others.

Since there's usually no special rotational or translational reference
frame in an MD simulation, it's often desirable to remove rotational and
translational motion via featurization that is insensitive to rotations and
translations. 

Featurizations
--------------

.. autosummary::

    AtomPairsFeaturizer
    ContactFeaturizer
    DRIDFeaturizer
    DihedralFeaturizer
    GaussianSolventFeaturizer
    RMSDFeaturizer
    RawPositionsFeaturizer
    SuperposeFeaturizer


Reading trajectories with MDAnalysis
------------------------------------

Featurizers work on MDTraj trajectories. To featurize trajectories that
MDTraj cannot read, such as a GROMACS ``.tpr`` or AMBER ``prmtop`` topology,
or a trajectory too large to fit in memory, read them with `MDAnalysis
<https://www.mdanalysis.org>`_ and featurize them chunk by chunk. Install the
optional dependency with ``pip install msmbuilder-neo[mdanalysis]``.

.. code-block:: python

    import MDAnalysis as mda
    from msmbuilder.featurizer import DihedralFeaturizer
    from msmbuilder.io.mdanalysis import featurize_mdanalysis

    universes = [mda.Universe('topol.tpr', 'run%d.xtc' % i) for i in range(10)]
    features = featurize_mdanalysis(DihedralFeaturizer(), universes, chunk=1000)

Only ``chunk`` frames are held in memory at a time, and ``features`` is a list
with one array per trajectory, as from ``featurizer.transform``. Pass an
``AtomGroup`` (for example ``u.select_atoms('protein')``) to featurize a
subset of atoms. Coordinates are converted from Angstrom to nm, and the
topology is converted so that atom names, residues, elements and bonds match
what ``md.load`` gives for the same system.

``iter_mdanalysis`` yields the MDTraj chunks themselves, and
``load_mdanalysis`` loads a whole trajectory at once.

Chunking does not change the features, because every featurizer computes
each frame independently. The one small exception is
:class:`~msmbuilder.featurizer.SASAFeaturizer`: MDTraj's SASA calculation can
differ by about 0.1% for the first frames of a chunk.

.. autosummary::

    msmbuilder.io.mdanalysis.featurize_mdanalysis
    msmbuilder.io.mdanalysis.iter_mdanalysis
    msmbuilder.io.mdanalysis.load_mdanalysis
    msmbuilder.io.mdanalysis.mdanalysis_topology


Alternative to Featurization
----------------------------

Many algorithms require vectorizable data. Other algorithms only require a
pairwise distance metric, e.g. RMSD between two protein conformations. In
general, you can define a pairwise distance among vectorized data, but you
cannot embed data into a vector space only from pairwise distance.

Some :ref:`clustering <cluster>` methods let you use an arbitrary distance
metric, including RMSD. In this case, the input to ``fit()`` may be a list
of MD trajectories instead of a list of numpy arrays. Clustering methods
that allow this currently include :class:`~msmbuilder.cluster.KCenters` and
:class:`~msmbuilder.cluster.KMedoids`.

.. vim: tw=75
