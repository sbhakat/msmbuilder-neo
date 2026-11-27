MSMBuilder Neo
==============

[![PyPI](https://img.shields.io/pypi/v/msmbuilder-neo.svg)](https://pypi.org/project/msmbuilder-neo/)
[![Python](https://img.shields.io/pypi/pyversions/msmbuilder-neo.svg)](https://pypi.org/project/msmbuilder-neo/)
[![CI](https://github.com/sbhakat/msmbuilder-neo/actions/workflows/ci.yml/badge.svg)](https://github.com/sbhakat/msmbuilder-neo/actions/workflows/ci.yml)
[![Docs](https://github.com/sbhakat/msmbuilder-neo/actions/workflows/docs.yml/badge.svg)](https://sbhakat.github.io/msmbuilder-neo/)
[![License: LGPL v2.1+](https://img.shields.io/badge/license-LGPL--2.1--or--later-blue.svg)](https://github.com/sbhakat/msmbuilder-neo/blob/main/LICENSE)

**Statistical models for biomolecular dynamics.** MSMBuilder Neo featurizes
molecular dynamics simulations, finds their slow processes (tICA, VAMP,
VAMPnets), clusters them into states, and builds validated Markov state
models and hidden Markov models -- on current Python, NumPy 2, Linux and
macOS.

It is a modernized fork of [MSMBuilder](https://github.com/msmbuilder/msmbuilder)
(via [msmbuilder2022](https://github.com/msmbuilder/msmbuilder2022)) that keeps
the MSMBuilder API: `import msmbuilder` and the `msmb` command work as before.

**Documentation:** <https://sbhakat.github.io/msmbuilder-neo/>

Installation
------------

```bash
pip install msmbuilder-neo
```

Python 3.10 or newer, on Linux (x86_64, aarch64) or macOS (Apple Silicon,
Intel). MSMBuilder Neo is pure Python: numerical kernels are compiled by
[Numba](https://numba.pydata.org) the first time they are used (a few
seconds, once). Install it in its own environment: it provides the same
`msmbuilder` package and `msmb` command as the original MSMBuilder.

Optional extras, e.g. `pip install "msmbuilder-neo[torch,deeptime]"`:

| Extra | Enables |
|---|---|
| `torch` | VAMPnets, state-free reversible VAMPnets, time-lagged autoencoders, DeepTICA, DeepLDA |
| `deeptime` | TRAM, OOM-reweighted MSMs, Chapman-Kolmogorov tests, PyEMMA's PCCA+ |
| `mdanalysis` | any trajectory format MDAnalysis reads, featurized in chunks |
| `dask` | lazily loaded datasets larger than memory |
| `parquet` | project metadata stored as Parquet |

Quick start
-----------

```python
from msmbuilder.example_datasets import AlanineDipeptide
from msmbuilder.featurizer import DihedralFeaturizer
from msmbuilder.cluster import KCenters
from msmbuilder.msm import MarkovStateModel
from msmbuilder.lumping import PCCAPlus, save_macrostate_samples

trajectories = AlanineDipeptide().get().trajectories
features = DihedralFeaturizer().fit_transform(trajectories)
labels = KCenters(n_clusters=100, random_state=0).fit_transform(features)

msm = MarkovStateModel(lag_time=10).fit(labels)
print(msm.timescales_[:3])                       # slowest relaxation timescales (frames)

pcca = PCCAPlus.from_msm(msm, n_macrostates=2, random_state=0)
save_macrostate_samples(pcca, labels, trajectories, n_samples=20,
                        superpose='backbone', out_dir='macrostates')   # one PDB per macrostate
```

Or describe the whole analysis in one YAML file and run it reproducibly:

```bash
msmb Pipeline --template > pipeline.yaml   # edit the data paths and steps
msmb Pipeline pipeline.yaml                # results + provenance.yaml
msmb Pipeline --verify pipeline.yaml       # rerun from scratch and compare
```

What it does
------------

- **Featurize**: dihedrals, contacts, distances, RMSD, SASA, ligand features,
  on MDTraj trajectories or anything MDAnalysis reads.
- **Reduce**: tICA (also sparse and kernel), PCA, VAMP; VAMPnets, SRVs,
  time-lagged autoencoders, DeepTICA and DeepLDA, exportable to PLUMED.
- **Cluster**: k-means, k-centers, k-medoids, regular spatial, landmark and
  scikit-learn clusterers.
- **Model**: maximum-likelihood, Bayesian and continuous-time MSMs, hidden
  Markov models, TRAM for biased simulations, OOM-reweighted MSMs.
- **Validate**: GMRQ and VAMP scores with cross-validation, implied
  timescales, Chapman-Kolmogorov tests.
- **Interpret**: PCCA+ macrostates (like PyEMMA) with representative
  structures as PDB files; transition path theory (fluxes, committors,
  pathways, mean first passage times).

Coming from MSMBuilder 3.8, msmbuilder2022 or PyEMMA? See the
[migration guide](https://sbhakat.github.io/msmbuilder-neo/migration.html).

Development
-----------

```bash
git clone https://github.com/sbhakat/msmbuilder-neo.git
pip install -e "./msmbuilder-neo[test]"
cd /tmp && pytest --pyargs msmbuilder.tests -n auto
```

See the [contributing guide](https://sbhakat.github.io/msmbuilder-neo/contributing.html).
Bug reports and questions: [GitHub issues](https://github.com/sbhakat/msmbuilder-neo/issues).

Citing
------

If you use MSMBuilder Neo in published work, please cite the MSMBuilder
papers (listed under [Publications](https://sbhakat.github.io/msmbuilder-neo/publications.html))
and the papers of the methods you use; each documentation page lists them.

Credits and license
-------------------

MSMBuilder was created by the Pande lab at Stanford University.
msmbuilder2022 is developed and maintained by Xuhui Huang's group at the
University of Wisconsin-Madison. MSMBuilder Neo keeps their copyright notices
and is licensed under the GNU LGPL, version 2.1 or later.
