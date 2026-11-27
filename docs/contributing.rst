.. _contributing:

Contributing
============

MSMBuilder Neo is developed on `GitHub
<https://github.com/sbhakat/msmbuilder-neo>`_. Bug reports, questions and
pull requests are welcome.

Reporting a problem
-------------------

Open an `issue <https://github.com/sbhakat/msmbuilder-neo/issues>`_ with the
MSMBuilder Neo version (``python -c "import msmbuilder.version as v;
print(v.full_version)"``), your platform, and a short example that shows the
problem.

Development setup
-----------------

.. code-block:: bash

    git clone https://github.com/sbhakat/msmbuilder-neo.git
    cd msmbuilder-neo
    python -m pip install -e ".[test,docs]"
    pre-commit install          # optional: run ruff on every commit

Run the tests from **outside** the source tree, against the installed
package:

.. code-block:: bash

    cd /tmp && python -m pytest --pyargs msmbuilder.tests -n auto

and the linters from the source tree:

.. code-block:: bash

    ruff check msmbuilder examples benchmarks
    mypy

Pull requests
-------------

* Add tests that fail without your change and pass with it. Tests live in
  ``msmbuilder/tests/test_*.py``; prefer small, deterministic data and seed
  every random number generator.
* When a change reimplements or replaces existing numerical code, compare
  against the previous results (see the ``benchmarks/*_parity.py`` scripts)
  and explain any difference in the pull request.
* New estimators follow the scikit-learn API on *lists* of sequences
  (``fit``, ``transform``/``partial_transform``, ``score``), with
  numpydoc docstrings that include references.
* Continuous integration runs the tests on Linux (x86_64 and aarch64) and
  macOS (Apple Silicon and Intel) with Python 3.10 and 3.13, the linters,
  and the documentation build.

Documentation
-------------

The documentation is built with Sphinx, the PyData theme and MyST-NB:

.. code-block:: bash

    make -C docs html                         # executes every notebook
    MSMB_DOCS_EXECUTE=off make -C docs html   # faster: skip execution

Example notebooks live in ``examples/`` (and ``examples/advanced/``) and are
copied into the documentation and executed on every build, so they must run
from top to bottom on a clean machine: use the example datasets from
``msmbuilder.example_datasets``, keep them to a few minutes, and commit them
without outputs. Add a new notebook to the toctree in
``docs/examples/index.rst``.

Every merge to ``main`` publishes the documentation to GitHub Pages.
