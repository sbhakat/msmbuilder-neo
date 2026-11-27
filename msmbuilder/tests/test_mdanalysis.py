import os
import warnings

import mdtraj as md
import numpy as np
import pytest

import msmbuilder.featurizer as F
from msmbuilder.example_datasets import AlanineDipeptide, FsPeptide, get_data_home

mda = pytest.importorskip("MDAnalysis")

from msmbuilder.io.mdanalysis import (featurize_mdanalysis, iter_mdanalysis,  # noqa: E402
                                      load_mdanalysis, mdanalysis_topology)


def _ala2_paths():
    AlanineDipeptide().cache()
    d = os.path.join(get_data_home(), "alanine_dipeptide")
    return os.path.join(d, "ala2.pdb"), os.path.join(d, "trajectory-0.dcd")


def _universe(pdb, trj):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mda.Universe(pdb, trj)


@pytest.fixture(scope="module")
def ala2():
    pdb, dcd = _ala2_paths()
    return md.load(dcd, top=pdb), _universe(pdb, dcd)


def test_topology_matches_mdtraj(ala2):
    t, u = ala2
    top = mdanalysis_topology(u)
    assert [a.name for a in top.atoms] == [a.name for a in t.topology.atoms]
    assert [a.element for a in top.atoms] == [a.element for a in t.topology.atoms]
    assert ([(r.name, r.resSeq) for r in top.residues]
            == [(r.name, r.resSeq) for r in t.topology.residues])
    assert top.n_chains == t.topology.n_chains
    assert top.n_bonds == t.topology.n_bonds


def test_coordinates_match_mdtraj(ala2):
    t, u = ala2
    loaded = load_mdanalysis(u)
    assert loaded.n_frames == t.n_frames
    # one float32 rounding step from the Angstrom -> nm conversion
    np.testing.assert_allclose(loaded.xyz, t.xyz, atol=2e-7, rtol=0)


@pytest.mark.parametrize("chunk", [1, 7, 1000, 100000])
def test_features_match_and_do_not_depend_on_chunk(ala2, chunk):
    t, u = ala2
    f = F.DihedralFeaturizer(["phi", "psi"])
    expected = f.partial_transform(t[:400])
    got = featurize_mdanalysis(f, u, chunk=chunk, stop=400)
    assert len(got) == 1
    np.testing.assert_allclose(got[0], expected, atol=1e-5, rtol=0)
    # different chunk sizes give exactly the same features
    np.testing.assert_array_equal(got[0], featurize_mdanalysis(f, u, chunk=13, stop=400)[0])


def test_start_stop_step(ala2):
    t, u = ala2
    sub = load_mdanalysis(u, start=10, stop=200, step=7)
    np.testing.assert_allclose(sub.xyz, t[10:200:7].xyz, atol=2e-7, rtol=0)
    chunks = list(iter_mdanalysis(u, chunk=5, start=10, stop=200, step=7))
    assert [c.n_frames for c in chunks] == [5] * 5 + [3]   # 28 frames


def test_atomgroup_subset_keeps_its_order(ala2):
    t, u = ala2
    order = [9, 1, 4, 14]
    ag = u.atoms[order]
    sub = load_mdanalysis(ag, stop=50)
    assert [a.name for a in sub.topology.atoms] == [t.topology.atom(i).name for i in order]
    np.testing.assert_allclose(sub.xyz, t.xyz[:50, order], atol=2e-7, rtol=0)


def test_box_is_converted_to_nm():
    u = mda.Universe.empty(3, trajectory=True)
    u.add_TopologyAttr("names", ["C1", "C2", "C3"])
    u.add_TopologyAttr("resnames", ["UNK"])
    u.add_TopologyAttr("resids", [1])
    u.atoms.positions = np.arange(9, dtype=np.float32).reshape(3, 3)
    u.dimensions = [20.0, 30.0, 40.0, 90.0, 90.0, 120.0]
    t = load_mdanalysis(u)
    np.testing.assert_allclose(t.unitcell_lengths, [[2.0, 3.0, 4.0]], rtol=1e-6)
    np.testing.assert_allclose(t.unitcell_angles, [[90.0, 90.0, 120.0]])
    np.testing.assert_allclose(t.xyz[0], u.atoms.positions / 10.0)


def test_bonds_are_copied_when_present(ala2):
    pdb, dcd = _ala2_paths()
    u = _universe(pdb, dcd)
    u.atoms.guess_bonds(vdwradii={"C": 1.7, "H": 1.1, "N": 1.6, "O": 1.5})
    top = mdanalysis_topology(u)
    assert top.n_bonds == len(u.bonds)
    mda_pairs = {tuple(sorted((b.atoms[0].index, b.atoms[1].index))) for b in u.bonds}
    md_pairs = {tuple(sorted((b[0].index, b[1].index))) for b in top.bonds}
    assert md_pairs == mda_pairs


def test_list_of_universes(ala2):
    t, u = ala2
    f = F.AtomPairsFeaturizer(np.array([[0, 10], [4, 15]]))
    out = featurize_mdanalysis(f, [u, u.atoms], chunk=500, stop=100)
    assert len(out) == 2
    np.testing.assert_array_equal(out[0], out[1])


def test_errors(ala2):
    t, u = ala2
    with pytest.raises(ValueError):
        next(iter_mdanalysis(u, chunk=0))
    with pytest.raises(ValueError):
        next(iter_mdanalysis(u.atoms[:5], topology=t.topology))


def _featurizers(traj):
    ca = traj.topology.select("name CA")
    ref = traj[0]
    return {
        "Dihedral": F.DihedralFeaturizer(),
        "VonMises": F.VonMisesFeaturizer(),
        "AlphaAngle": F.AlphaAngleFeaturizer(),
        "KappaAngle": F.KappaAngleFeaturizer(),
        "Contact": F.ContactFeaturizer(scheme="ca"),
        "BinaryContact": F.BinaryContactFeaturizer(scheme="ca"),
        "LogisticContact": F.LogisticContactFeaturizer(scheme="ca"),
        "AtomPairs": F.AtomPairsFeaturizer(np.array([[ca[0], ca[5]], [ca[2], ca[9]]])),
        "RMSD": F.RMSDFeaturizer(ref),
        "LandMarkRMSD": F.LandMarkRMSDFeaturizer(traj[[0, 50, 100]]),
        "Superpose": F.SuperposeFeaturizer(ca, ref),
        "RawPositions": F.RawPositionsFeaturizer(atom_indices=ca, ref_traj=ref),
        "DRID": F.DRIDFeaturizer(atom_indices=ca),
        "GaussianSolvent": F.GaussianSolventFeaturizer(
            solute_indices=ca[:5], solvent_indices=ca[5:], sigma=0.5),
    }


def test_featurizers_are_frame_local():
    # Chunked featurization relies on each frame's features not depending
    # on the other frames. (SASAFeaturizer is left out: MDTraj's
    # shrake_rupley varies by ~0.1% for the first frames of a batch.)
    traj = FsPeptide().get_cached().trajectories[0][:120]
    for name, f in _featurizers(traj).items():
        whole = f.partial_transform(traj)
        chunked = np.concatenate([f.partial_transform(traj[i:i + 37])
                                  for i in range(0, traj.n_frames, 37)])
        np.testing.assert_allclose(chunked, whole, atol=1e-6, err_msg=name)
