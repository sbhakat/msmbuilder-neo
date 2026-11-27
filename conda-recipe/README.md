# conda-forge recipe for MSMBuilder Neo

This is the recipe to submit to [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes)
once `msmbuilder-neo` is on PyPI. After it is accepted, conda-forge creates a
`msmbuilder-neo-feedstock` repository, and its bot opens pull requests for
every new PyPI release.

## Submitting

1. Get the checksum of the released sdist and put it into `meta.yaml`:

   ```bash
   curl -sL https://pypi.org/pypi/msmbuilder-neo/1.0.0/json \
     | python -c "import json,sys; print([u['digests']['sha256'] for u in json.load(sys.stdin)['urls'] if u['packagetype']=='sdist'][0])"
   ```

2. Fork `conda-forge/staged-recipes`, copy this directory to
   `recipes/msmbuilder-neo/`, and open a pull request. Its CI builds the
   package and runs the tests in `meta.yaml` (imports, `pip check`,
   `msmb -h`).

3. The recipe maintainer listed in `meta.yaml` (`sbhakat`) must comment on
   the pull request to confirm being a maintainer.

Notes

- The package is `noarch: python` (pure Python); Numba compiles the kernels
  at run time.
- The PyPI wheel pins `numba<0.63` on Intel macOS because PyPI has no newer
  Intel-macOS wheels; conda-forge builds Numba for `osx-64`, so the recipe
  does not need that pin.
