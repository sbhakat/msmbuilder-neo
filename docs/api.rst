.. _api:

API reference
=============

Every public class and function, by module. The :ref:`user guide <user_guide>`
explains when to use them.

.. _api_msmbuilder_featurizer:

Featurizers
-----------

Turn trajectories into feature arrays.

.. currentmodule:: msmbuilder.featurizer

.. autosummary::
   :toctree: generated/

   AlphaAngleFeaturizer
   AtomPairsFeaturizer
   BinaryContactFeaturizer
   BinaryLigandContactFeaturizer
   CommonContactFeaturizer
   ContactFeaturizer
   DihedralFeaturizer
   DRIDFeaturizer
   FeatureUnion
   FirstSlicer
   FunctionFeaturizer
   GaussianSolventFeaturizer
   KappaAngleFeaturizer
   LandMarkRMSDFeaturizer
   LigandContactFeaturizer
   LigandFeaturizer
   LigandRMSDFeaturizer
   LogisticContactFeaturizer
   RawPositionsFeaturizer
   RMSDFeaturizer
   SASAFeaturizer
   Slicer
   SuperposeFeaturizer
   VonMisesFeaturizer
   dict_maker
   featurize_all
   get_atompair_indices
   zippy_maker

.. _api_msmbuilder_preprocessing:

Preprocessing
-------------

Scale, filter and impute feature time series.

.. currentmodule:: msmbuilder.preprocessing

.. autosummary::
   :toctree: generated/

   Binarizer
   Butterworth
   DoubleEWMA
   EWMA
   FunctionTransformer
   Imputer
   KernelCenterer
   LabelBinarizer
   MaxAbsScaler
   MinMaxScaler
   MultiLabelBinarizer
   Normalizer
   PolynomialFeatures
   RobustScaler
   StandardScaler

.. _api_msmbuilder_decomposition:

Decomposition
-------------

Linear and nonlinear dimensionality reduction for time series (tICA, VAMP, deep collective variables).

.. currentmodule:: msmbuilder.decomposition

.. autosummary::
   :toctree: generated/

   DeepLDA
   DeepTICA
   FactorAnalysis
   FastICA
   KernelTICA
   KSparseTICA
   MiniBatchSparsePCA
   PCA
   SparsePCA
   SparseTICA
   SRV
   tICA
   TimeLaggedAutoencoder
   VAMP
   VAMPNet

.. _api_msmbuilder_cluster:

Clustering
----------

Assign frames to discrete states.

.. currentmodule:: msmbuilder.cluster

.. autosummary::
   :toctree: generated/

   AffinityPropagation
   AgglomerativeClustering
   APM
   GMM
   KCenters
   KMeans
   KMedoids
   LandmarkAgglomerative
   MeanShift
   MiniBatchKMeans
   MiniBatchKMedoids
   NDGrid
   RegularSpatial
   SpectralClustering

.. _api_msmbuilder_msm:

Markov state models
-------------------

Estimate, validate and convert Markov state models.

.. currentmodule:: msmbuilder.msm

.. autosummary::
   :toctree: generated/

   BayesianContinuousTimeMSM
   BayesianMarkovStateModel
   ContinuousTimeMSM
   MarkovStateModel
   OOMReweightedMSM
   TRAM
   ck_test
   from_deeptime
   implied_timescales
   to_deeptime

.. _api_msmbuilder_lumping:

Lumping
-------

Coarse-grain microstates into macrostates.

.. currentmodule:: msmbuilder.lumping

.. autosummary::
   :toctree: generated/

   BACE
   MVCA
   PCCA
   PCCAPlus
   save_macrostate_samples

.. _api_msmbuilder_tpt:

Transition path theory
----------------------

Fluxes, committors, pathways and first passage times.

.. currentmodule:: msmbuilder.tpt

.. autosummary::
   :toctree: generated/

   committors
   conditional_committors
   fluxes
   fraction_visited
   hub_scores
   mfpts
   net_fluxes
   paths
   top_path

.. _api_msmbuilder_hmm:

Hidden Markov models
--------------------

Continuous-emission hidden Markov models.

.. currentmodule:: msmbuilder.hmm

.. autosummary::
   :toctree: generated/

   GaussianHMM
   VonMisesHMM

.. _api_msmbuilder_feature_selection:

Feature selection
-----------------

Select and combine features.

.. currentmodule:: msmbuilder.feature_selection

.. autosummary::
   :toctree: generated/

   FeatureSelector
   MultiSequenceFeatureSelectionMixin
   VarianceThreshold

.. _api_msmbuilder_dataset:

Datasets
--------

Lists of sequences stored on disk.

.. currentmodule:: msmbuilder.dataset

.. autosummary::
   :toctree: generated/

   dataset

.. _api_msmbuilder_io:

Input and output
----------------

Project metadata, trajectories and saved objects.

.. currentmodule:: msmbuilder.io

.. autosummary::
   :toctree: generated/

   GenericParser
   HierarchyParser
   NumberedRunsParser
   ParseWarning
   TemplateProject
   backup
   gather_metadata
   itertrajs
   load_generic
   load_meta
   load_trajs
   preload_top
   preload_tops
   render_meta
   save_generic
   save_meta
   save_trajs

.. _api_msmbuilder_io_mdanalysis:

MDAnalysis
----------

Read MDAnalysis trajectories as MDTraj chunks.

.. currentmodule:: msmbuilder.io.mdanalysis

.. autosummary::
   :toctree: generated/

   featurize_mdanalysis
   iter_mdanalysis
   load_mdanalysis
   mdanalysis_topology

.. _api_msmbuilder_pipeline:

Pipelines
---------

Run a whole analysis from a YAML file.

.. currentmodule:: msmbuilder.pipeline

.. autosummary::
   :toctree: generated/

   PipelineError
   PipelineResult
   EXAMPLE_CONFIG
   load_config
   run_pipeline
   verify_pipeline

.. _api_msmbuilder_example_datasets:

Example datasets
----------------

Small datasets for tutorials and tests.

.. currentmodule:: msmbuilder.example_datasets

.. autosummary::
   :toctree: generated/

   AlanineDipeptide
   DoubleWell
   FsPeptide
   MetEnkephalin
   QuadWell
   clear_data_home
   doublewell_eigs
   fetch_alanine_dipeptide
   fetch_fs_peptide
   fetch_met_enkephalin
   get_data_home
   has_msmb_data
   load_doublewell
   load_muller
   load_quadwell
   quadwell_eigs

.. _api_msmbuilder_utils:

Utilities
---------

Helpers.

.. currentmodule:: msmbuilder.utils

.. autosummary::
   :toctree: generated/

   ExperimentalWarning
   KDTree
   Subsampler
   array2d
   categorical
   check_iter_of_sequences
   dump
   experimental
   list_of_1d
   load
   map_drawn_samples
   param_sweep
   printoptions
   verbosedump
   verboseload
