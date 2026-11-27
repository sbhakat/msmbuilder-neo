"""The YAML pipeline runner: results, provenance, reuse of unchanged steps,
reproducibility and verification."""
import os
import subprocess

import numpy as np
import pandas as pd
import pytest
import yaml

from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.dataset import dataset
from msmbuilder.decomposition import tICA
from msmbuilder.msm import MarkovStateModel, implied_timescales
from msmbuilder.pipeline import (EXAMPLE_CONFIG, PipelineError, load_config,
                                 resolve_class, run_pipeline, verify_pipeline)


def _double_well(n, rs):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] - 0.02 * (4 * x[t - 1] ** 3 - 4 * x[t - 1]) + 0.25 * rs.randn()
    return np.c_[x, 0.5 * rs.randn(n), 0.2 * x + 0.3 * rs.randn(n)]


@pytest.fixture
def project(tmp_path):
    rs = np.random.RandomState(0)
    with dataset(str(tmp_path / "features.h5"), "w", "hdf5") as ds:
        for i in range(4):
            ds[i] = _double_well(1500, rs)
    config = {
        "version": 1, "random_state": 7, "output": "out",
        "data": {"features": "features.h5"},
        "steps": [
            {"name": "tica", "estimator": "tICA", "params": {"n_components": 2, "lag_time": 5}},
            {"name": "clusters", "estimator": "MiniBatchKMeans", "params": {"n_clusters": 12}},
            {"name": "msm", "estimator": "MarkovStateModel",
             "params": {"lag_time": 5, "verbose": False}},
        ],
        "validation": {"implied_timescales": {"lag_times": [1, 2, 5, 10], "n_timescales": 3}},
    }
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return tmp_path, path, config


def _write(path, config):
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def test_runs_and_records_provenance(project):
    tmp, path, _ = project
    res = run_pipeline(str(path), verbose=False)
    out = tmp / "out"
    for fn in ("tica.h5", "tica.pkl", "clusters.h5", "clusters.pkl", "msm.pkl",
               "implied_timescales.csv", "provenance.yaml"):
        assert (out / fn).exists(), fn
    assert not (out / "msm.h5").exists()          # MSMs are fitted, not used to transform
    assert isinstance(res.models["tica"], tICA)
    assert isinstance(res.models["clusters"], MiniBatchKMeans)
    assert isinstance(res.models["msm"], MarkovStateModel)
    assert res.models["clusters"].random_state == 7          # the pipeline seed
    prov = yaml.safe_load((out / "provenance.yaml").read_text())
    assert prov == res.provenance
    assert prov["msmbuilder_pipeline"] == 1
    assert {"python", "msmbuilder", "numpy", "scipy", "sklearn", "mdtraj", "h5py"} \
        <= set(prov["environment"]["versions"])
    assert prov["inputs"]["features"]["n_sequences"] == 4
    steps = {s["name"]: s for s in prov["steps"]}
    assert steps["clusters"]["params"]["n_clusters"] == 12
    assert steps["clusters"]["params"]["random_state"] == 7
    assert steps["tica"]["params"]["n_components"] == 2
    assert len(steps["msm"]["timescales"]) == 5 and steps["msm"]["n_states"] == 12
    assert not any(s["cached"] for s in prov["steps"])
    # the outputs are what the estimators give directly
    with dataset(str(tmp / "features.h5")) as ds:
        X = [ds[k] for k in ds.keys()]
    Y = tICA(n_components=2, lag_time=5).fit_transform(X)
    with dataset(str(out / "tica.h5")) as ds:
        for k, y in zip(ds.keys(), Y):
            np.testing.assert_allclose(ds[k], y)
    labels = MiniBatchKMeans(n_clusters=12, random_state=7).fit_predict(Y)
    with dataset(str(out / "clusters.h5")) as ds:
        for k, lab in zip(ds.keys(), labels):
            np.testing.assert_array_equal(ds[k], lab)
    its = pd.read_csv(out / "implied_timescales.csv")
    ref = implied_timescales(labels, [1, 2, 5, 10], n_timescales=3,
                             msm=MarkovStateModel(lag_time=5, verbose=False))
    np.testing.assert_array_equal(its["lag_time"], [1, 2, 5, 10])
    np.testing.assert_allclose(its[["timescale_1", "timescale_2", "timescale_3"]], ref)


def test_rerun_reuses_unchanged_steps(project):
    tmp, path, config = project
    first = run_pipeline(str(path), verbose=False)
    again = run_pipeline(str(path), verbose=False)
    assert all(s["cached"] for s in again.provenance["steps"])
    assert again.fingerprint == first.fingerprint
    # a new MSM lag time: only the MSM is refitted
    config["steps"][2]["params"]["lag_time"] = 10
    _write(path, config)
    third = run_pipeline(str(path), verbose=False)
    assert [s["cached"] for s in third.provenance["steps"]] == [True, True, False]
    assert third.models["msm"].lag_time == 10
    # new tICA parameters: everything downstream is refitted
    config["steps"][0]["params"]["n_components"] = 1
    _write(path, config)
    fourth = run_pipeline(str(path), verbose=False)
    assert [s["cached"] for s in fourth.provenance["steps"]] == [False, False, False]


def test_force_recomputes_identically(project):
    _, path, _ = project
    first = run_pipeline(str(path), verbose=False)
    forced = run_pipeline(str(path), force=True, verbose=False)
    assert not any(s["cached"] for s in forced.provenance["steps"])
    assert forced.fingerprint == first.fingerprint
    for a, b in zip(first.provenance["steps"], forced.provenance["steps"]):
        assert a.get("output_sha256") == b.get("output_sha256")


def test_changed_outputs_are_recomputed(project):
    tmp, path, _ = project
    first = run_pipeline(str(path), verbose=False)
    with dataset(str(tmp / "out" / "clusters.h5"), "a", "hdf5") as ds:
        ds[0] = np.zeros_like(ds[0])                       # tamper with a result
    again = run_pipeline(str(path), verbose=False)
    cached = {s["name"]: s["cached"] for s in again.provenance["steps"]}
    # the clusters are recomputed; their (restored) output is the original one,
    # so the MSM fitted to it is still valid and is reused
    assert cached == {"tica": True, "clusters": False, "msm": True}
    assert again.fingerprint == first.fingerprint
    with dataset(str(tmp / "out" / "clusters.h5")) as ds:
        assert ds[0].any()


def test_changed_input_invalidates_everything(project):
    tmp, path, _ = project
    run_pipeline(str(path), verbose=False)
    with dataset(str(tmp / "features.h5"), "a", "hdf5") as ds:
        ds[0] = ds[0][:-10]
    again = run_pipeline(str(path), verbose=False)
    assert not any(s["cached"] for s in again.provenance["steps"])


def test_verify(project):
    tmp, path, _ = project
    run_pipeline(str(path), verbose=False)
    assert verify_pipeline(str(path), verbose=False) == []
    prov_path = tmp / "out" / "provenance.yaml"
    prov = yaml.safe_load(prov_path.read_text())
    prov["steps"][1]["output_sha256"] = "0" * 64
    prov_path.write_text(yaml.safe_dump(prov, sort_keys=False))
    assert verify_pipeline(str(path), verbose=False) == ["step clusters: output_sha256 differs"]
    os.remove(prov_path)
    with pytest.raises(PipelineError):
        verify_pipeline(str(path), verbose=False)


def test_ck_test_validation(project):
    pytest.importorskip("deeptime")
    tmp, path, config = project
    config["validation"]["ck_test"] = {"lag_times": [5, 10, 20], "n_metastable_sets": 2}
    _write(path, config)
    res = run_pipeline(str(path), verbose=False)
    ck = np.load(tmp / "out" / "ck_test.npz")
    assert ck["predictions"].shape == ck["estimates"].shape == (4, 2, 2)
    np.testing.assert_array_equal(ck["lag_times"], [0, 5, 10, 20])
    assert res.provenance["validation"]["ck_test"]["max_deviation"] < 0.2


def test_featurizes_trajectories(tmp_path):
    from msmbuilder.example_datasets import AlanineDipeptide, get_data_home
    from msmbuilder.featurizer import DihedralFeaturizer
    import mdtraj as md
    AlanineDipeptide().cache()
    src = os.path.join(get_data_home(), "alanine_dipeptide")
    config = {
        "random_state": 0, "output": str(tmp_path / "out"),
        "data": {"trajectories": [os.path.join(src, "trajectory-1.dcd"),
                                  os.path.join(src, "trajectory-0.dcd")],
                 "topology": os.path.join(src, "ala2.pdb"), "stride": 10, "chunk": 300},
        "steps": [{"name": "dihedrals", "featurizer": "DihedralFeaturizer",
                   "params": {"types": ["phi", "psi"]}},
                  {"name": "clusters", "estimator": "KCenters", "params": {"n_clusters": 5}}],
    }
    res = run_pipeline(config, verbose=False)
    f = DihedralFeaturizer(types=["phi", "psi"])
    with dataset(res.outputs["dihedrals"]) as ds:
        assert list(ds.keys()) == [0, 1]
        for k, name in zip(ds.keys(), ["trajectory-1.dcd", "trajectory-0.dcd"]):
            t = md.load(os.path.join(src, name), top=os.path.join(src, "ala2.pdb"), stride=10)
            np.testing.assert_allclose(ds[k], f.partial_transform(t), atol=1e-6)
    traj_inputs = res.provenance["inputs"]["trajectories"]
    assert [os.path.basename(t["path"]) for t in traj_inputs] == ["trajectory-1.dcd",
                                                                   "trajectory-0.dcd"]
    assert len(traj_inputs[0]["sha256"]) == 64 and "topology" in res.provenance["inputs"]
    assert res.provenance["validation"] == {}


def test_resolve_class():
    assert resolve_class("tICA") is tICA
    assert resolve_class("msmbuilder.decomposition.tICA") is tICA
    with pytest.raises(PipelineError):
        resolve_class("NoSuchEstimator")
    with pytest.raises(PipelineError):
        resolve_class("os.path.join")                  # only msmbuilder classes
    with pytest.raises(PipelineError):
        resolve_class("msmbuilder.decomposition.nothing")


def test_example_config_is_valid():
    cfg = load_config(yaml.safe_load(EXAMPLE_CONFIG))
    assert [s["name"] for s in cfg["steps"]] == ["features", "tica", "clusters", "msm"]
    assert cfg["validation"]["implied_timescales"]["lag_times"] == [1, 2, 5, 10, 20, 50]


@pytest.mark.parametrize("change,match", [
    (lambda c: c.update(colour="red"), "unknown key"),
    (lambda c: c.update(version=2), "version"),
    (lambda c: c.update(random_state="x"), "random_state"),
    (lambda c: c.pop("data"), "data"),
    (lambda c: c["data"].update(trajectories="*.xtc"), "exactly one"),
    (lambda c: c.update(steps=[]), "steps"),
    (lambda c: c["steps"][1].update(name="tica"), "unique name"),
    (lambda c: c["steps"][1].update(name="provenance"), "unique name"),
    (lambda c: c["steps"][0].update(featurizer="DihedralFeaturizer"), "exactly one"),
    (lambda c: c["steps"][1].pop("estimator"), "exactly one"),
    (lambda c: c["steps"][1].update(estimator="Nope"), "no MSMBuilder class"),
    (lambda c: c["steps"][1].update(params=[1]), "mapping"),
    (lambda c: c["steps"][1].update(transform="yes"), "true or false"),
    (lambda c: c["steps"].insert(1, {"name": "f", "featurizer": "DihedralFeaturizer"}),
     "first step"),
    (lambda c: c["validation"].update(ck_test={"lag_times": [10, 5], "n_metastable_sets": 2}),
     "increasing"),
    (lambda c: c["validation"].update(ck_test={"lag_times": [5, 10], "n_metastable_sets": 1}),
     "n_metastable_sets"),
    (lambda c: c["validation"]["implied_timescales"].update(lag_times=[0, 1]), "positive"),
])
def test_config_errors(project, change, match):
    _, _, config = project
    change(config)
    with pytest.raises(PipelineError, match=match):
        load_config(config)


def test_run_errors(project):
    tmp, path, config = project
    bad = dict(config, steps=[{"name": "tica", "estimator": "tICA",
                               "params": {"n_componentz": 2}}])
    with pytest.raises(PipelineError, match="n_componentz"):
        run_pipeline(dict(bad, _base_dir=str(tmp)), verbose=False)
    continuous = dict(config, steps=config["steps"][:1])   # validation on features
    with pytest.raises(PipelineError, match="integer state labels"):
        run_pipeline(dict(continuous, _base_dir=str(tmp)), verbose=False)
    missing = dict(config, data={"trajectories": "nothing/*.xtc"},
                   steps=[{"name": "f", "featurizer": "DihedralFeaturizer"}])
    with pytest.raises(PipelineError, match="no trajectory files"):
        run_pipeline(dict(missing, _base_dir=str(tmp)), verbose=False)


def test_command_line(project):
    tmp, path, _ = project
    out = subprocess.run(["msmb", "Pipeline", "--template"], capture_output=True, text=True)
    assert out.returncode == 0 and yaml.safe_load(out.stdout)["version"] == 1
    out = subprocess.run(["msmb", "Pipeline", "-q", str(path)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert (tmp / "out" / "provenance.yaml").exists()
    out = subprocess.run(["msmb", "Pipeline", "--verify", str(path)], capture_output=True,
                         text=True)
    assert out.returncode == 0 and "Reproduced" in out.stdout, out.stdout + out.stderr
    alt = tmp / "elsewhere"
    out = subprocess.run(["msmb", "Pipeline", "-q", "-o", str(alt), str(path)],
                         capture_output=True, text=True)
    assert out.returncode == 0 and (alt / "msm.pkl").exists()
    path.write_text("steps: 3\n")
    out = subprocess.run(["msmb", "Pipeline", str(path)], capture_output=True, text=True)
    assert out.returncode == 1 and "pipeline error" in out.stderr
