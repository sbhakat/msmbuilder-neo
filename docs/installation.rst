.. _installation:

Installation
============

MSMBuilder Neo needs Python 3.10 or newer and runs on Linux and macOS (Intel
and Apple Silicon). It is pure Python: no compiler is needed. Numerical
kernels are compiled by `Numba <https://numba.pydata.org>`_ the first time
they are used (a few seconds, once; the result is cached).

.. code-block:: bash

    pip install msmbuilder-neo

The package is imported as ``msmbuilder`` and provides the ``msmb`` command,
like the original MSMBuilder, so the two should not be installed in the same
environment. To try the latest development version instead:

.. code-block:: bash

    pip install git+https://github.com/sbhakat/msmbuilder-neo.git

Optional features
-----------------

Install extras with ``pip install "msmbuilder-neo[torch,deeptime]"``:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Extra
     - Enables
   * - ``torch``
     - VAMPnets, state-free reversible VAMPnets, time-lagged autoencoders,
       DeepTICA and DeepLDA (:ref:`decomposition`). Install a CPU or CUDA
       build of PyTorch as you prefer.
   * - ``deeptime``
     - TRAM, OOM-reweighted MSMs, the Chapman-Kolmogorov test, conversion
       to deeptime models, and PyEMMA's PCCA+ algorithm (:ref:`msm`).
   * - ``mdanalysis``
     - Reading any trajectory format MDAnalysis supports, in chunks
       (:ref:`featurization`).
   * - ``dask``
     - Lazily loaded datasets larger than memory (:ref:`datasets`).
   * - ``parquet``
     - Project metadata stored as Parquet (:ref:`io`).
   * - ``docs``
     - Building this documentation.

From a clone
------------

.. code-block:: bash

    git clone https://github.com/sbhakat/msmbuilder-neo.git
    cd msmbuilder-neo
    pip install -e ".[test]"
    cd /tmp && pytest --pyargs msmbuilder.tests -n auto

(Run the tests from outside the source tree, against the installed package.)

Platforms
---------

Continuous integration tests every change on Linux x86_64 and aarch64 and
macOS on Apple Silicon and Intel, with Python 3.10 and 3.13. Windows is not
supported for now. On Intel Macs, current Numba and PyTorch publish no
wheels, so Numba is pinned below 0.63 there and the PyTorch features are
unavailable.
