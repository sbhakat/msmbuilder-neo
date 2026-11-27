"""PCCA/PCCA+ macrostate quantities (checked against deeptime, PyEMMA's
successor), macrostate sampling, and writing macrostate structures."""
import itertools
import os

import mdtraj as md
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from msmbuilder.lumping import PCCA, PCCAPlus, save_macrostate_samples
from msmbuilder.msm import MarkovStateModel

N = 30


def _three_wells():
    """Metropolis chain on 30 lattice states with three wells; the barrier
    tops fall between states, so the macrostates are unambiguous."""
    x = np.arange(N)
    U = 2.4 * np.cos(2 * np.pi * (x + 0.5) / 10)
    T = np.zeros((N, N))
    for i in range(N):
        for j in (i - 1, i + 1):
            if 0 <= j < N:
                T[i, j] = 0.5 * min(1.0, np.exp(-(U[j] - U[i])))
        T[i, i] = 1 - T[i].sum()
    pi = np.exp(-U) / np.exp(-U).sum()
    return T, pi


def _exact_msm(T, pi, labels=None):
    msm = MarkovStateModel(verbose=False)
    msm.transmat_, msm.populations_, msm.n_states_ = T, pi, len(pi)
    msm.countsmat_ = np.ones_like(T)
    labels = range(len(pi)) if labels is None else labels
    msm.mapping_ = {lab: i for i, lab in enumerate(labels)}
    msm._is_dirty = True
    return msm


def _sample(T, n, x0, rs):
    cum = np.cumsum(T, axis=1)
    out = np.empty(n, dtype=np.int64)
    s = x0
    for t, r in enumerate(rs.rand(n)):
        out[t] = s
        s = min(int(np.searchsorted(cum[s], r)), len(T) - 1)
    return out


def _match(ours, ref):
    """Column permutation of ours that best matches ref."""
    m = ref.shape[1]
    return list(min(itertools.permutations(range(m)),
                    key=lambda p: np.abs(ours[:, list(p)] - ref).max()))


# --- the macrostate quantities ----------------------------------------------------

def _random_blocks(seed):
    """A reversible chain on 25 states in 4 weakly coupled blocks."""
    rs = np.random.RandomState(seed)
    n = 25
    blocks = np.repeat(np.arange(4), [6, 7, 6, 6])
    C = rs.rand(n, n) * 0.01 + (blocks[:, None] == blocks[None, :]) * rs.rand(n, n)
    C = (C + C.T) * 1000
    return C / C.sum(axis=1, keepdims=True), C.sum(axis=1) / C.sum()


@pytest.mark.parametrize("system", ["three_wells", "blocks0", "blocks1"])
@pytest.mark.parametrize("m", [2, 3, 4])
def test_deeptime_method_is_pyemma_pcca(system, m):
    dm = pytest.importorskip("deeptime.markov")
    T, pi = _three_wells() if system == "three_wells" else _random_blocks(int(system[-1]))
    ours = PCCAPlus.from_msm(_exact_msm(T, pi), m, method="deeptime")
    ref = dm.pcca(T, m, pi)
    np.testing.assert_allclose(ours.memberships_, ref.memberships, atol=1e-12)
    np.testing.assert_array_equal(ours.metastable_assignments_, ref.assignments)
    np.testing.assert_allclose(ours.coarse_grained_transmat_,
                               ref.coarse_grained_transition_matrix, atol=1e-10)
    np.testing.assert_allclose(ours.coarse_grained_populations_,
                               ref.coarse_grained_stationary_probability, atol=1e-12)
    np.testing.assert_allclose(ours.metastable_distributions_, ref.metastable_distributions,
                               atol=1e-12)
    for a, b in zip(ours.metastable_sets_, ref.sets):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("m", [2, 3])
def test_default_method_finds_equally_metastable_macrostates(m):
    dm = pytest.importorskip("deeptime.markov")
    T, pi = _three_wells()
    ours = PCCAPlus.from_msm(_exact_msm(T, pi), m, random_state=0)
    ref = dm.pcca(T, m, pi)
    np.testing.assert_allclose(np.trace(ours.coarse_grained_transmat_),
                               np.trace(ref.coarse_grained_transition_matrix), atol=1e-3)
    if m == 2:      # one clear barrier: the same split
        perm = _match(ours.memberships_, ref.memberships)
        np.testing.assert_allclose(ours.memberships_[:, perm], ref.memberships, atol=1e-6)


def test_derived_quantities_follow_pyemma_definitions():
    T, pi = _three_wells()
    ours = PCCAPlus.from_msm(_exact_msm(T, pi), 3, random_state=0)
    chi = ours.memberships_
    assert np.all(chi >= 0)
    np.testing.assert_allclose(chi.sum(axis=1), 1.0)
    np.testing.assert_allclose(ours.coarse_grained_populations_, chi.T @ pi)
    np.testing.assert_allclose(ours.metastable_distributions_.sum(axis=1), 1.0)
    np.testing.assert_allclose(ours.metastable_distributions_,
                               (chi * pi[:, None]).T / (chi.T @ pi)[:, None])
    np.testing.assert_allclose(ours.coarse_grained_transmat_,
                               np.linalg.inv(chi.T @ chi) @ chi.T @ T @ chi)
    np.testing.assert_array_equal(ours.metastable_assignments_, np.argmax(chi, axis=1))


def test_pcca_crisp_memberships_and_labels():
    T, pi = _three_wells()
    labels = list(range(100, 100 + 10 * N, 10))
    pcca = PCCA.from_msm(_exact_msm(T, pi, labels), 3)
    chi = pcca.memberships_
    assert set(np.unique(chi)) <= {0.0, 1.0}
    np.testing.assert_array_equal(np.argmax(chi, 1), pcca.microstate_mapping_)
    assert sorted(np.concatenate(pcca.metastable_sets_)) == list(range(N))
    for s, lab in zip(pcca.metastable_sets_, pcca.metastable_label_sets_):
        np.testing.assert_array_equal(lab, np.array(labels)[s])
    np.testing.assert_allclose(pcca.coarse_grained_transmat_.sum(axis=1), 1.0)


def test_pcca_plus_random_state():
    T, pi = _three_wells()
    msm = _exact_msm(T, pi)
    a = PCCAPlus.from_msm(msm, 4, random_state=3)
    b = PCCAPlus.from_msm(msm, 4, random_state=3)
    np.testing.assert_array_equal(a.chi_, b.chi_)
    assert clone(PCCAPlus(2, random_state=5)).random_state == 5
    # fitting on sequences works as before
    rs = np.random.RandomState(0)
    seqs = [_sample(T, 3000, 5, rs) for _ in range(3)]
    fitted = PCCAPlus(3, random_state=0, verbose=False).fit(seqs)
    assert fitted.memberships_.shape == (fitted.n_states_, 3)
    with pytest.raises(ValueError):
        PCCAPlus(2, method="pyemma")


# --- sampling -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def sampled():
    T, _ = _three_wells()
    rs = np.random.RandomState(1)
    seqs = [_sample(T, 20000, x0, rs) for x0 in (5, 15, 25)]
    msm = MarkovStateModel(lag_time=1, verbose=False).fit(seqs)
    return seqs, PCCAPlus.from_msm(msm, 3, random_state=0)


def test_assignment_weighting_draws_from_the_macrostate(sampled):
    seqs, pcca = sampled
    draws = pcca.draw_macrostate_samples(seqs, 200, weighting="assignment", random_state=0)
    assert len(draws) == 3
    for k, pairs in enumerate(draws):
        assert pairs.shape == (200, 2)
        micro = np.array([pcca.mapping_[seqs[t][f]] for t, f in pairs])
        assert np.all(pcca.microstate_mapping_[micro] == k)


def test_membership_weighting_follows_metastable_distributions(sampled):
    seqs, pcca = sampled
    n = 20000
    draws = pcca.draw_macrostate_samples(seqs, n, random_state=1)
    for k, pairs in enumerate(draws):
        micro = np.array([pcca.mapping_[seqs[t][f]] for t, f in pairs])
        freq = np.bincount(micro, minlength=pcca.n_states_) / n
        np.testing.assert_allclose(freq, pcca.metastable_distributions_[k], atol=0.012)
    again = pcca.draw_macrostate_samples(seqs, n, random_state=1)
    for a, b in zip(draws, again):
        np.testing.assert_array_equal(a, b)


def test_sampling_uses_labels_not_indices():
    # non-contiguous labels, and a state removed by ergodic trimming
    T, _ = _three_wells()
    rs = np.random.RandomState(2)
    seqs = [_sample(T, 5000, x0, rs) * 7 + 3 for x0 in (5, 15, 25)]
    seqs.append(np.array([999, 999, 999]))                 # not connected: trimmed
    msm = MarkovStateModel(verbose=False).fit(seqs)
    assert 999 not in msm.mapping_
    pcca = PCCAPlus.from_msm(msm, 3, random_state=0)
    for k, pairs in enumerate(pcca.draw_macrostate_samples(seqs, 300, weighting="assignment",
                                                           random_state=0)):
        labels = np.array([seqs[t][f] for t, f in pairs])
        assert 999 not in labels
        assert set(labels) <= set(pcca.metastable_label_sets_[k])


def test_sampling_errors_and_empty_macrostates(sampled):
    seqs, pcca = sampled
    with pytest.raises(ValueError):
        pcca.draw_macrostate_samples(seqs, 5, weighting="uniform")
    with pytest.raises(ValueError):
        pcca.draw_macrostate_samples(seqs, 0)
    with pytest.raises(ValueError):
        PCCAPlus(2).draw_macrostate_samples(seqs, 5)
    # sequences that only visit one macrostate: the others have no frames
    members = pcca.metastable_label_sets_[1]
    one = [s[np.isin(s, members)] for s in seqs]
    draws = pcca.draw_macrostate_samples(one, 10, weighting="assignment")
    assert [len(d) for d in draws] == [0, 10, 0]


# --- writing structures --------------------------------------------------------------

def _toy_trajectories(seqs, tmp_path, stride=1):
    """Three-atom 'trajectories' whose x coordinate is the lattice state,
    saved as .h5 files with every frame repeated ``stride`` times."""
    top = md.Topology()
    chain = top.add_chain()
    res = top.add_residue("ALA", chain)
    for name in ("N", "CA", "CB"):     # three atoms: superposition is well defined
        top.add_atom(name, md.element.carbon, res)
    files, trajs = [], []
    for i, s in enumerate(seqs):
        full = np.repeat(s, stride)
        n = len(full)
        xyz = np.zeros((n, 3, 3), dtype=np.float32)
        xyz[:, 0, 0] = full * 0.1
        xyz[:, 1, 0] = full * 0.1 + 0.15
        xyz[:, 1, 1] = np.arange(n) * 1e-4                   # frame-specific
        angle = 0.3 * np.sin(np.arange(n))                     # and a varying geometry
        xyz[:, 2, 0] = full * 0.1 + 0.15 * np.cos(angle)
        xyz[:, 2, 2] = 0.15 * np.sin(angle) + 0.1
        t = md.Trajectory(xyz, top)
        path = str(tmp_path / ("traj%d.h5" % i))
        t.save(path)
        files.append(path)
        trajs.append(t)
    return files, trajs, top


def test_save_macrostate_samples_from_files(sampled, tmp_path):
    seqs, pcca = sampled
    short = [s[:3000] for s in seqs]
    files, trajs, _ = _toy_trajectories(short, tmp_path, stride=2)
    out = tmp_path / "out"
    res = save_macrostate_samples(pcca, short, files, n_samples=25, stride=2, out_dir=str(out),
                                  fmt="h5", random_state=0)
    assert [os.path.basename(f) for f in res["files"]] == ["macrostate_%d.h5" % k
                                                          for k in range(3)]
    index = pd.read_csv(out / "macrostate_samples.csv")
    pd.testing.assert_frame_equal(index, res["index"], check_dtype=False)
    assert len(index) == 75 and set(index.macrostate) == {0, 1, 2}
    for k, f in enumerate(res["files"]):
        saved = md.load(f)
        assert saved.n_frames == 25
        rows = index[index.macrostate == k]
        for j, r in enumerate(rows.itertuples()):
            src = trajs[r.trajectory]
            np.testing.assert_array_equal(saved.xyz[j], src.xyz[r.frame])   # the listed frame
            assert r.frame % 2 == 0 and r.microstate == short[r.trajectory][r.frame // 2]
            assert r.membership == pytest.approx(
                pcca.memberships_[pcca.mapping_[r.microstate], k])


def test_save_macrostate_samples_pdb(sampled, tmp_path):
    seqs, pcca = sampled
    short = [s[:2000] for s in seqs]
    _, trajs, _ = _toy_trajectories(short, tmp_path)
    res = save_macrostate_samples(pcca, short, trajs, n_samples=8, out_dir=str(tmp_path),
                                  prefix="pcca", weighting="assignment", random_state=1)
    for k, f in enumerate(res["files"]):
        assert f.endswith("pcca_%d.pdb" % k)
        text = open(f).read()
        assert text.count("MODEL") == 8
        pdb = md.load(f)
        rows = res["index"][res["index"].macrostate == k]
        for j, r in enumerate(rows.itertuples()):
            np.testing.assert_allclose(pdb.xyz[j], trajs[r.trajectory].xyz[r.frame], atol=2e-4)
            assert pcca.microstate_mapping_[pcca.mapping_[r.microstate]] == k


def test_superpose(sampled, tmp_path):
    seqs, pcca = sampled
    short = [s[:2000] for s in seqs]
    _, trajs, _ = _toy_trajectories(short, tmp_path)
    res = save_macrostate_samples(pcca, short, trajs, n_samples=10, out_dir=str(tmp_path),
                                  fmt="h5", superpose="all", random_state=0)
    t = md.load(res["files"][0])
    # superposed: the plain coordinate RMSD to the first frame is the optimal RMSD
    plain = np.sqrt(((t.xyz - t.xyz[0]) ** 2).sum(axis=2).mean(axis=1))
    np.testing.assert_allclose(plain, md.rmsd(t, t, 0), atol=1e-4)


def test_save_errors(sampled, tmp_path):
    seqs, pcca = sampled
    short = [s[:500] for s in seqs]
    files, trajs, _ = _toy_trajectories(short, tmp_path)
    with pytest.raises(ValueError):
        save_macrostate_samples(pcca, short, files[:2], out_dir=str(tmp_path))
    with pytest.raises(TypeError):
        save_macrostate_samples(MarkovStateModel(), short, files, out_dir=str(tmp_path))
    with pytest.raises(IndexError, match="stride"):
        save_macrostate_samples(pcca, short, files, stride=5, n_samples=50,
                                out_dir=str(tmp_path), random_state=0)
    with pytest.raises(IndexError, match="stride"):
        save_macrostate_samples(pcca, short, trajs, stride=5, n_samples=50,
                                out_dir=str(tmp_path), random_state=0)
    with pytest.raises(ValueError, match="matches no atoms"):
        save_macrostate_samples(pcca, short, trajs, superpose="name XX", n_samples=2,
                                out_dir=str(tmp_path))
    members = pcca.metastable_label_sets_[1]
    keep = [i for i, s in enumerate(short) if np.isin(s, members).any()]
    one = [short[i][np.isin(short[i], members)] for i in keep]
    with pytest.warns(UserWarning, match="no frames"):
        res = save_macrostate_samples(pcca, one, [trajs[i][:len(o)] for i, o in zip(keep, one)],
                                      weighting="assignment", n_samples=3,
                                      out_dir=str(tmp_path))
    assert [f is None for f in res["files"]] == [True, False, True]


def test_alanine_dipeptide_macrostates(tmp_path):
    """End to end on real data: PCCA+ macrostates of alanine dipeptide,
    written as PDB files with the frames they list."""
    from msmbuilder.cluster import KCenters
    from msmbuilder.example_datasets import AlanineDipeptide, get_data_home
    from msmbuilder.featurizer import DihedralFeaturizer
    AlanineDipeptide().cache()
    src = os.path.join(get_data_home(), "alanine_dipeptide")
    top = os.path.join(src, "ala2.pdb")
    files = [os.path.join(src, "trajectory-%d.dcd" % i) for i in range(3)]
    trajs = [md.load(f, top=top, stride=5) for f in files]
    X = DihedralFeaturizer(types=["phi", "psi"], sincos=False).fit_transform(trajs)
    labels = KCenters(n_clusters=30, random_state=0).fit_predict(X)
    msm = MarkovStateModel(lag_time=2, verbose=False).fit(labels)
    try:
        import deeptime.markov as dm
    except ImportError:
        dm = None
    if dm is not None:
        for m in (2, 3, 4):
            ours = PCCAPlus.from_msm(msm, m, method="deeptime")
            np.testing.assert_allclose(ours.memberships_,
                                       dm.pcca(msm.transmat_, m, msm.populations_).memberships,
                                       atol=1e-12)
    pcca = PCCAPlus.from_msm(msm, 2, random_state=0)
    res = save_macrostate_samples(pcca, labels, files, n_samples=10, top=top, stride=5,
                                  superpose="backbone", out_dir=str(tmp_path), random_state=0)
    for k, f in enumerate(res["files"]):
        pdb = md.load(f)
        assert pdb.n_frames == 10 and pdb.n_atoms == trajs[0].n_atoms
        rows = res["index"][res["index"].macrostate == k]
        for j, r in enumerate(rows.itertuples()):
            ref = md.load_frame(r.trajectory_file, r.frame, top=top)
            assert md.rmsd(pdb[j], ref)[0] < 1e-3                   # the listed frame
            np.testing.assert_allclose(md.compute_phi(pdb[j])[1], md.compute_phi(ref)[1],
                                       atol=2e-3)
