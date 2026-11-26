MSMBuilder Neo
==============

MSMBuilder Neo is a modernized fork of [msmbuilder2022](https://github.com/msmbuilder/msmbuilder2022),
which is itself based on msmbuilder 3.8.0. It keeps the MSMBuilder API
(`import msmbuilder`) while updating the build so the package installs on
current Python and NumPy.

> **Status:** early development (Phase 1 of the roadmap: packaging). APIs may change.

Installation
------------

MSMBuilder Neo requires Python 3.10 or newer and works with NumPy 1.24+ and NumPy 2.

Install from a local clone:

```bash
git clone https://github.com/sbhakat/msmbuilder-neo.git
python -m pip install ./msmbuilder-neo
```

Prebuilt wheels for Linux, macOS and Windows are built by CI and will be
published to PyPI once the first release is cut.

Building from source needs a C/C++ compiler. OpenMP is used when found and
skipped otherwise (pass `-Csetup-args=-Dopenmp=disabled` to turn it off). On
macOS, install it with `brew install libomp`.

To run the tests:

```bash
python -m pip install "./msmbuilder-neo[test]"
python msmbuilder-neo/msmbuilder/tests/data_init.py   # downloads example datasets
pytest --pyargs msmbuilder.tests
```

Credits
-------

MSMBuilder was created by the Pande lab at Stanford University.
msmbuilder2022 is developed and maintained by Xuhui Huang's group at the
University of Wisconsin-Madison. MSMBuilder Neo keeps their copyright notices
and the LGPL-2.1-or-later license.

About MSMBuilder
----------------

[![PyPi version](https://badge.fury.io/py/msmbuilder.svg)](https://pypi.python.org/pypi/msmbuilder/)
[![License](https://img.shields.io/badge/license-LGPLv2.1+-red.svg?style=flat)](https://pypi.python.org/pypi/msmbuilder/)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg?style=flat)](http://msmbuilder.org)

MSMBuilder is a python package which implements a series of statistical
models for high-dimensional time-series. It is particularly focused on the
analysis of atomistic simulations of biomolecular dynamics. For example,
MSMBuilder has been used to model protein folding and conformational change
from molecular dynamics (MD) simulations. MSMBuilder is available under the
LGPL (v2.1 or later).

Capabilities include:

- Feature extraction into dihedrals, contact maps, and more
- Geometric clustering with a variety of algorithms.
- Dimensionality reduction using time-structure independent component
  analysis (tICA) and principal component analysis (PCA).
- Markov state model (MSM) construction
- Rate-matrix MSM construction
- Hidden markov model (HMM) construction
- Timescale and transition path analysis.

Check out the documentation at [msmbuilder.org](http://msmbuilder.org) and
join the [mailing list](https://mailman.stanford.edu/mailman/listinfo/msmbuilder-user).
For a broader overview of MSMBuilder, take a look at our [slide deck](http://rawgit.com/msmbuilder/talk/master/index.html).

Workflow
--------

An example workflow might be as follows:

1. Set up a system for molecular dynamics, and run one or more simulations
   for as long as you can on as many CPUs or GPUs as you have access to.
   There are a lot of great software packages for running MD, e.g [OpenMM]
   (https://simtk.org/home/openmm), [Gromacs](http://www.gromacs.org/),
   [Amber](http://ambermd.org/), [CHARMM](http://www.charmm.org/), and
   many others. MSMBuilder is not one of them.

2. Transform your MD coordinates into an appropriate set of features.

3. Perform some sort of dimensionality reduction with tICA or PCA.
   Reduce your data into discrete states by using clustering.

4. Fit an MSM, rate matrix MSM, or HMM. Perform model selection using
   cross-validation with the [generalized matrix Rayleigh quotient](http://arxiv.org/abs/1407.8083)
