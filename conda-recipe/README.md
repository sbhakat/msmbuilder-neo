# conda-forge recipe for MSMBuilder Neo

This is the recipe to submit to [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes)
once `msmbuilder-neo` is on PyPI. After it is accepted, conda-forge creates a
`msmbuilder-neo-feedstock` repository, and its bot opens pull requests for
every new PyPI release.

## Submitting

1. `meta.yaml` already has the checksum of the 1.0.0 sdist on PyPI. For a
   later version, update `version` and `sha256`:

   ```bash
   curl -sL https://pypi.org/pypi/msmbuilder-neo/<version>/json \
     | python -c "import json,sys; print([u['digests']['sha256'] for u in json.load(sys.stdin)['urls'] if u['packagetype']=='sdist'][0])"
   ```

   (After the feedstock exists, conda-forge's bot does this automatically.)

2. Fork `conda-forge/staged-recipes`, copy this directory to
   `recipes/msmbuilder-neo/`, and open a pull request. Its CI builds the
   package and runs the tests in `meta.yaml` (imports, `pip check`,
   `msmb -h`).

3. The recipe maintainer listed in `meta.yaml` (`sbhakat`) must comment on
   the pull request to confirm being a maintainer.

The recipe was built and tested locally with conda-build against
conda-forge (osx-arm64): it builds the noarch package from the PyPI sdist
and passes its import, `pip check` and `msmb -h` tests.

Notes

- The package is `noarch: python` (pure Python); Numba compiles the kernels
  at run time.
- The PyPI wheel pins `numba<0.63` on Intel macOS because PyPI has no newer
  Intel-macOS wheels; conda-forge builds Numba for `osx-64`, so the recipe
  does not need that pin.
