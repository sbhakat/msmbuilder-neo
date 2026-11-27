# conda-forge recipe for MSMBuilder Neo

`recipe.yaml` is the conda-forge recipe (v1 format, as conda-forge requires for
new recipes). It was submitted to
[conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes);
once accepted, conda-forge creates a `msmbuilder-neo-feedstock` repository,
whose bot opens a pull request for every new PyPI release, and the package
installs with `conda install -c conda-forge msmbuilder-neo`.

## Checking it locally

With [rattler-build](https://rattler.build) and conda-smithy (both on
conda-forge):

```bash
conda-smithy recipe-lint --conda-forge conda-recipe     # conda-forge's linter
printf 'python_min:\n  - "3.10"\n' > variants.yaml       # conda-forge's global pin
rattler-build build --recipe conda-recipe/recipe.yaml -c conda-forge -m variants.yaml
```

The recipe passes the linter, builds the noarch package from the PyPI
sdist, and passes its tests: imports on Python 3.10 and the newest Python,
`pip check`, and `msmb -h`.

## Notes

- `noarch: python`: the package is pure Python; Numba compiles the kernels at
  run time.
- `run_constraints: msmbuilder <0.0a0`: the original `msmbuilder` package on
  conda-forge installs the same `msmbuilder` module and `msmb` command, so
  conda refuses to install the two together.
- Two patches are applied to the 1.0.1 sdist:
  `0001-...` drops the PyPI-only `numba<0.63` pin for Intel macOS (conda-forge
  has current Numba for `osx-64`; the pin also made `pip check` fail there), and
  `0002-...` makes `tools/list_package_files.py` print `/`-separated paths, so
  Windows builds install files into their packages (fixed upstream after
  1.0.1; drop this patch for the next release).
- `pip check` runs as a script test with `netcdf4` added: conda-forge's
  `mdtraj` declares netcdf4 in its Python metadata without depending on it.
- For a new release: update `version` and `sha256` (from
  `https://pypi.org/pypi/msmbuilder-neo/<version>/json`), or let the
  feedstock's bot do it.
