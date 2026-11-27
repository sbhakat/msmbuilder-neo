"""HDF5 datasets on h5py (compatible with PyTables files), Dask loading and
Parquet/CSV project metadata."""
import os
import pickle

import numpy as np
import pandas as pd
import pytest

from msmbuilder.dataset import HDF5Dataset, dataset
from msmbuilder.io import load_meta, load_trajs, save_meta, save_trajs
from msmbuilder.io import io as msmio


def _arrays():
    rs = np.random.RandomState(0)
    return {0: rs.randn(7, 3).astype(np.float32), 1: rs.randint(0, 5, size=11),
            2: rs.randn(4), 10: np.arange(6, dtype=np.int32).reshape(3, 2),
            3: np.zeros((0, 2))}


# --- compatibility with datasets written by PyTables ------------------------------

def _write_like_old_msmbuilder(path, arrays, provenance):
    """What HDF5Dataset did before MSMBuilder Neo (PyTables)."""
    tables = pytest.importorskip("tables")
    with tables.open_file(path, mode="w", filters=tables.Filters(complevel=0)) as f:
        for k, x in arrays.items():
            if x.size:
                f.create_carray("/", "arr_%d" % k, obj=x)
            else:
                f.create_array("/", "arr_%d" % k, obj=x)
        f.root._v_attrs["provenance"] = provenance


def test_reads_pytables_datasets(tmp_path):
    path = str(tmp_path / "old.h5")
    arrays = _arrays()
    _write_like_old_msmbuilder(path, arrays, "MSMBuilder Dataset:\n  made by PyTables")
    with dataset(path) as ds:
        assert isinstance(ds, HDF5Dataset)
        assert list(ds.keys()) == [0, 1, 2, 3, 10]
        for k, x in arrays.items():
            got = ds[k]
            assert got.dtype == x.dtype and got.shape == x.shape
            np.testing.assert_array_equal(got, x)
        assert ds.provenance == "MSMBuilder Dataset:\n  made by PyTables"
        assert len(ds) == 5


def test_pytables_reads_new_datasets(tmp_path):
    tables = pytest.importorskip("tables")
    path = str(tmp_path / "new.h5")
    arrays = _arrays()
    with dataset(path, "w", "hdf5") as ds:
        for k, x in arrays.items():
            ds[k] = x
    with tables.open_file(path) as f:
        for k, x in arrays.items():
            np.testing.assert_array_equal(f.get_node("/", "arr_%d" % k)[:], x)
        assert "MSMBuilder Dataset" in str(f.root._v_attrs["provenance"])


# --- the h5py backend ---------------------------------------------------------------

def test_modes(tmp_path):
    path = str(tmp_path / "ds.h5")
    with dataset(path, "w", "hdf5") as ds:
        ds[0] = np.arange(3)
    with pytest.raises(ValueError):
        dataset(path, "w", "hdf5")                  # exists
    with dataset(path, "a", "hdf5") as ds:          # append, overwrite
        ds[1] = np.arange(4)
        ds[0] = np.arange(5)
        provenance = ds.provenance
    with dataset(path) as ds:
        assert list(ds.keys()) == [0, 1]
        np.testing.assert_array_equal(ds[0], np.arange(5))
        assert ds.provenance == provenance          # not rewritten by 'a'
        with pytest.raises(OSError):
            ds[2] = np.arange(2)
        with pytest.raises(IndexError):
            ds.get(7)
    fresh = str(tmp_path / "fresh.h5")
    with dataset(fresh, "a", "hdf5") as ds:         # 'a' creates a new file
        ds[4] = np.ones(2)
        assert "MSMBuilder Dataset" in ds.provenance
    with pytest.raises(ValueError):
        dataset(fresh, "x", "hdf5")


def test_compression(tmp_path):
    import h5py
    path = str(tmp_path / "z.h5")
    x = np.tile(np.arange(100.0), (200, 1))
    with dataset(path, "w", "hdf5", compression="gzip") as ds:
        ds[0] = x
        ds[1] = np.zeros(0)
    with h5py.File(path) as f:
        assert f["arr_0"].compression == "gzip"
    assert os.path.getsize(path) < x.nbytes / 5
    with dataset(path) as ds:
        np.testing.assert_array_equal(ds[0], x)


def test_pickle_reopens(tmp_path):
    path = str(tmp_path / "p.h5")
    ds = dataset(path, "w", "hdf5")
    ds[0] = np.arange(3)
    ds.flush()
    clone = pickle.loads(pickle.dumps(ds))
    np.testing.assert_array_equal(clone[0], np.arange(3))
    ds.close()
    clone.close()
    ro = dataset(path)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(ro))[0], np.arange(3))
    ro.close()


def test_derived_datasets_keep_provenance(tmp_path):
    from msmbuilder.preprocessing import StandardScaler
    src = str(tmp_path / "in.h5")
    with dataset(src, "w", "hdf5") as ds:
        for k, x in enumerate(np.random.RandomState(1).randn(3, 50, 2)):
            ds[k] = x
    with dataset(src) as ds:
        out = ds.fit_transform_with(StandardScaler(), str(tmp_path / "out.h5"))
        assert "Derived from" in out.provenance
        allx = np.concatenate(list(out))
        np.testing.assert_allclose(allx.mean(axis=0), 0, atol=1e-12)
        out.close()


# --- Dask ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt,name", [("hdf5", "d.h5"), ("dir-npy", "d")])
def test_to_dask(tmp_path, fmt, name):
    da = pytest.importorskip("dask.array")
    path = str(tmp_path / name)
    rs = np.random.RandomState(2)
    arrays = [rs.randn(1000, 4), rs.randn(300, 4)]
    with dataset(path, "w", fmt) as ds:
        for k, x in enumerate(arrays):
            ds[k] = x
    with dataset(path) as ds:
        lazy = ds.to_dask(chunks=(128, 4))
        assert len(lazy) == 2 and all(isinstance(a, da.Array) for a in lazy)
        assert lazy[0].chunks[0][0] == 128
        np.testing.assert_allclose(lazy[0].mean(axis=0).compute(), arrays[0].mean(axis=0))
        np.testing.assert_array_equal(lazy[1][10:20].compute(), arrays[1][10:20])
        total = da.concatenate(lazy).sum(axis=0).compute()
        np.testing.assert_allclose(total, np.concatenate(arrays).sum(axis=0))


def test_to_dask_not_for_trajectories(tmp_path):
    pytest.importorskip("dask.array")
    ds = dataset(str(tmp_path / "*.xtc"), fmt="mdtraj")
    assert list(ds.to_dask()) == []                 # no files: nothing to load
    ds.glob_matches = ["a.xtc"]
    with pytest.raises(NotImplementedError):
        ds.to_dask()


# --- project metadata ----------------------------------------------------------------

def _meta():
    meta = pd.DataFrame({
        "run": [0, 0, 1, 1], "clone": [0, 1, 0, 1],
        "traj_fn": ["r0/c0.xtc", "r0/c1.xtc", "r1/c0.xtc", "r1/c1.xtc"],
        "top_fn": ["top.pdb"] * 4, "nframes": [100, 120, 90, 110],
        "step_ps": [10.0, 10.0, 20.0, 20.0]})
    return meta.set_index(["run", "clone"]).sort_index()


@pytest.mark.parametrize("fn", ["meta.parquet", "meta.csv", "meta.pandas.pickl"])
def test_metadata_round_trip(tmp_path, monkeypatch, fn):
    if fn.endswith(".parquet"):
        pytest.importorskip("pyarrow")
    monkeypatch.chdir(tmp_path)
    meta = _meta()
    assert save_meta(meta, fn) == fn
    loaded = load_meta(fn)
    pd.testing.assert_frame_equal(loaded, meta)
    pd.testing.assert_frame_equal(load_meta(), meta)    # found by default


def test_metadata_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    meta = _meta()
    monkeypatch.setattr(msmio, "_parquet_available", lambda: False)
    assert save_meta(meta) == "meta.csv"
    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)
    if msmio._parquet_available():
        assert save_meta(meta) == "meta.parquet"
        assert load_meta().equals(meta)                 # parquet preferred over csv
    # a legacy pickle is still found when it is the only file
    for fn in ("meta.parquet", "meta.csv"):
        if os.path.exists(fn):
            os.remove(fn)
    meta.to_pickle("meta.pandas.pickl")
    pd.testing.assert_frame_equal(load_meta(), meta)
    os.remove("meta.pandas.pickl")
    with pytest.raises(FileNotFoundError):
        load_meta()
    with pytest.raises(ValueError):
        save_meta(meta, "meta.txt")


def test_metadata_csv_simple_index(tmp_path):
    meta = pd.DataFrame({"traj_fn": ["a.xtc", "b.xtc"], "nframes": [5, 6]})
    fn = str(tmp_path / "m.csv")
    save_meta(meta, fn)
    pd.testing.assert_frame_equal(load_meta(fn), meta)
    named = meta.rename_axis("key")
    save_meta(named, fn)
    pd.testing.assert_frame_equal(load_meta(fn), named)


def test_load_trajs_finds_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    meta = _meta()
    trajs = {k: np.full((3, 2), i, dtype=float) for i, k in enumerate(meta.index)}
    save_meta(meta)
    save_trajs(trajs, "ftrajs", meta)
    meta2, loaded = load_trajs("ftrajs")
    pd.testing.assert_frame_equal(meta2, meta)
    for k in meta.index:
        np.testing.assert_array_equal(loaded[k], trajs[k])
