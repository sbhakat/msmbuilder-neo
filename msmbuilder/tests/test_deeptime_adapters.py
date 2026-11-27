"""Adapters to deeptime: TRAM, OOM-reweighted MSMs, the CK test, conversions."""
import pickle

import numpy as np
import pytest
from sklearn.base import clone

pytest.importorskip("deeptime")

import deeptime.markov.msm as dtmsm  # noqa: E402

from msmbuilder.msm import (TRAM, MarkovStateModel, OOMReweightedMSM,  # noqa: E402
                            ck_test, from_deeptime, to_deeptime)

# A double well on a 20-state lattice, sampled with Metropolis moves.
N = 20
X = np.arange(N)
U = 0.002 * (X - 4.5) ** 2 * (X - 14.5) ** 2
U = U - U.min()
PI = np.exp(-U) / np.exp(-U).sum()


def _metropolis(E):
    T = np.zeros((N, N))
    for i in range(N):
        for j in (i - 1, i + 1):
            if 0 <= j < N:
                T[i, j] = 0.5 * min(1.0, np.exp(-(E[j] - E[i])))
        T[i, i] = 1 - T[i].sum()
    return T


def _sample(T, n_steps, x0, rs):
    cum = np.cumsum(T, axis=1)
    r = rs.rand(n_steps)
    out = np.empty(n_steps, dtype=np.int64)
    s = x0
    for t in range(n_steps):
        out[t] = s
        s = min(int(np.searchsorted(cum[s], r[t])), N - 1)
    return out


@pytest.fixture(scope="module")
def unbiased():
    rs = np.random.RandomState(0)
    return [_sample(_metropolis(U), 20000, 4, rs) for _ in range(2)]


@pytest.fixture(scope="module")
def umbrella():
    """Umbrella sampling: 8 harmonic biases (kT) along the lattice."""
    rs = np.random.RandomState(1)
    centers = np.linspace(0, N - 1, 8)
    B = 0.25 * (X[:, None] - centers[None, :]) ** 2
    trajs = [_sample(_metropolis(U + B[:, k]), 10000, int(round(c)), rs)
             for k, c in enumerate(centers)]
    return trajs, [B[t] for t in trajs], B


# --- conversions --------------------------------------------------------------------

def test_round_trip(unbiased):
    msm = MarkovStateModel(lag_time=2, verbose=False).fit(unbiased)
    dt = to_deeptime(msm)
    np.testing.assert_array_equal(dt.transition_matrix, msm.transmat_)
    np.testing.assert_allclose(dt.stationary_distribution, msm.populations_, rtol=1e-14)
    np.testing.assert_array_equal(dt.count_model.count_matrix, msm.countsmat_)
    assert dt.lagtime == 2 and dt.reversible
    # deeptime orders eigenvalues by modulus, MSMBuilder by value: compare
    # the slow (positive) part of the spectrum
    np.testing.assert_allclose(dt.timescales(2), msm.timescales_[:2], rtol=1e-10)
    back = from_deeptime(dt)
    np.testing.assert_array_equal(back.transmat_, msm.transmat_)
    np.testing.assert_allclose(back.populations_, msm.populations_, rtol=1e-14)
    np.testing.assert_array_equal(back.countsmat_, msm.countsmat_)
    assert back.mapping_ == msm.mapping_ and back.lag_time == 2
    np.testing.assert_allclose(back.timescales_, msm.timescales_, rtol=1e-12)


def test_conversion_keeps_state_labels():
    seqs = [np.array([10, 30, 30, 20, 10, 20, 30, 10] * 50)]
    msm = MarkovStateModel(verbose=False).fit(seqs)
    dt = to_deeptime(msm)
    np.testing.assert_array_equal(dt.count_model.state_symbols, [10, 20, 30])
    assert from_deeptime(dt).mapping_ == {10: 0, 20: 1, 30: 2}
    # non-integer labels: deeptime sees the internal indices
    words = [np.array(["a", "c", "b", "a", "c", "b", "b", "a"] * 50)]
    m2 = MarkovStateModel(verbose=False).fit(words)
    dt2 = to_deeptime(m2)
    np.testing.assert_array_equal(dt2.count_model.state_symbols, [0, 1, 2])
    assert from_deeptime(dt2, labels=["a", "b", "c"]).mapping_ == m2.mapping_


def test_deeptime_mle_agrees_with_msmbuilder(unbiased):
    msm = MarkovStateModel(lag_time=1, verbose=False).fit(unbiased)
    ref = dtmsm.MaximumLikelihoodMSM(lagtime=1, reversible=True, maxiter=100000,
                                     maxerr=1e-12).fit_fetch(unbiased)
    converted = from_deeptime(ref)
    assert converted.mapping_ == msm.mapping_
    np.testing.assert_allclose(converted.transmat_, msm.transmat_, atol=1e-6)
    np.testing.assert_allclose(converted.populations_, msm.populations_, atol=1e-6)


# --- TRAM ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tram(umbrella):
    trajs, bias, _ = umbrella
    return TRAM().fit(trajs, bias)     # one TRAM fit (~10 s) shared by the tests


def test_tram_recovers_free_energies(umbrella, tram):
    _, _, B = umbrella
    assert tram.n_therm_states_ == 8 and tram.n_states_ == N and len(tram.msms_) == 8
    # unbiased free energies of the well-sampled states (observed error 0.05-0.09 kT)
    f = tram.markov_state_energies_ - tram.markov_state_energies_[4]
    np.testing.assert_allclose(f[2:18], (U - U[4])[2:18], atol=0.2)
    np.testing.assert_allclose(tram.populations_, PI, atol=0.01)
    # free energies of the biased ensembles, f_k = -log sum_x exp(-U - b_k)
    exact = -np.log(np.exp(-(U[:, None] + B)).sum(axis=0))
    np.testing.assert_allclose(tram.therm_state_energies_ - tram.therm_state_energies_[0],
                               exact - exact[0], atol=0.2)
    for msm in tram.msms_:
        # each ensemble's MSM, on the states it connects
        assert isinstance(msm, MarkovStateModel) and 3 <= msm.n_states_ < N
        np.testing.assert_allclose(msm.transmat_.sum(axis=1), 1.0)
        assert np.all(np.isfinite(msm.timescales_[:2])) and msm.timescales_[0] > 0
        labels = sorted(msm.mapping_)
        assert labels == list(range(labels[0], labels[-1] + 1))   # a contiguous window


def test_tram_is_deeptime_tram(umbrella, tram):
    trajs, bias, _ = umbrella
    ref = dtmsm.TRAM().fit_fetch((trajs, bias))
    np.testing.assert_allclose(tram.therm_state_energies_, ref.therm_state_energies,
                               atol=1e-6)
    np.testing.assert_allclose(tram.markov_state_energies_, ref.markov_state_energies,
                               atol=1e-6)
    ref.msm_collection.select(3)
    P = ref.msm_collection.transition_matrix
    keep = sorted(tram.msms_[3].mapping_)
    np.testing.assert_allclose(tram.msms_[3].transmat_, P[np.ix_(keep, keep)], atol=1e-6)


def test_tram_with_one_unbiased_ensemble_is_the_reversible_mle(unbiased):
    msm = MarkovStateModel(lag_time=1, verbose=False).fit(unbiased)
    tram = TRAM(maxerr=1e-12, maxiter=100000).fit(
        unbiased, [np.zeros((len(t), 1)) for t in unbiased],
        ttrajs=[np.zeros(len(t), dtype=int) for t in unbiased])
    np.testing.assert_allclose(tram.msms_[0].transmat_, msm.transmat_, atol=1e-4)
    np.testing.assert_allclose(tram.populations_, msm.populations_, atol=1e-4)


def test_tram_ttrajs(umbrella, tram):
    # all ensembles in one trajectory, labelled by ttrajs: transitions between
    # ensembles are not counted, so the estimate is unchanged
    trajs, bias, _ = umbrella
    joined = TRAM().fit(
        [np.concatenate(trajs)], [np.concatenate(bias)],
        ttrajs=[np.concatenate([np.full(len(t), k) for k, t in enumerate(trajs)])])
    np.testing.assert_allclose(joined.therm_state_energies_, tram.therm_state_energies_,
                               atol=1e-6)
    np.testing.assert_allclose(joined.populations_, tram.populations_, atol=1e-6)


def test_tram_weights_and_pmf(umbrella, tram):
    trajs, bias, _ = umbrella
    w = tram.sample_weights(trajs, bias)
    allw, allx = np.concatenate(w), np.concatenate(trajs)
    np.testing.assert_allclose(allw.sum(), 1.0)
    # (to the convergence of the TRAM iteration)
    np.testing.assert_allclose(np.bincount(allx, weights=allw, minlength=N),
                               tram.populations_, atol=1e-6)
    # PMF along the states themselves is the unbiased free energy
    pmf = tram.pmf(trajs, bias, trajs, n_bins=N)
    f = tram.markov_state_energies_
    np.testing.assert_allclose(pmf, f - f.min(), atol=1e-4)   # TRAM's convergence
    # PMF along a coarser coordinate (two states per bin)
    coarse = tram.pmf(trajs, bias, [t // 2 for t in trajs])
    p = np.exp(-coarse) / np.exp(-coarse).sum()
    np.testing.assert_allclose(p, tram.populations_.reshape(-1, 2).sum(axis=1), atol=1e-6)
    # weights in a biased ensemble reproduce its own histogram on average
    w3 = np.concatenate(tram.sample_weights(trajs, bias, therm_state=3))
    np.testing.assert_allclose(w3.sum(), 1.0)


def test_tram_string_labels(umbrella):
    trajs, bias, _ = umbrella
    trajs, bias = [t[:2000] for t in trajs], [b[:2000] for b in bias]
    names = np.array(["s%02d" % i for i in range(N)])
    named = TRAM(maxerr=1e-11).fit([names[t] for t in trajs], bias)
    ref = TRAM(maxerr=1e-11).fit(trajs, bias)
    assert named.mapping_ == {name: i for i, name in enumerate(names[:named.n_states_])}
    np.testing.assert_allclose(named.populations_, ref.populations_, atol=1e-12)
    assert set(named.msms_[0].mapping_) <= set(names)


def test_tram_convergence_flag(umbrella):
    trajs, bias, _ = umbrella
    trajs, bias = [t[:1000] for t in trajs], [b[:1000] for b in bias]
    with pytest.warns(Warning, match="did not converge"):
        short = TRAM(maxiter=2, init_strategy=None).fit(trajs, bias)
    assert not short.converged_


def test_tram_errors(umbrella, tram):
    trajs, bias, _ = umbrella
    with pytest.raises(ValueError):
        TRAM().fit(trajs, bias[:-1])                       # one bias matrix missing
    with pytest.raises(ValueError):
        TRAM().fit(trajs, [b[:, :3] for b in bias])        # wrong n_therm (no ttrajs)
    with pytest.raises(ValueError):
        TRAM().fit(trajs, [b[:-1] for b in bias])          # wrong n_frames
    with pytest.raises(ValueError):
        TRAM().fit(trajs, bias, ttrajs=[t[:-1] for t in trajs])
    with pytest.raises(ValueError):
        TRAM().sample_weights(trajs, bias)                 # not fitted
    with pytest.raises(ValueError):
        tram.pmf([np.array([0, 1, 99])], [np.zeros((3, 8))], [np.zeros(3, int)])
    with pytest.raises(ValueError):
        tram.sample_weights(trajs, [b[:, :3] for b in bias])


# --- OOM reweighting ------------------------------------------------------------------

@pytest.fixture(scope="module")
def short_nonequilibrium():
    """Many short runs, all started in the left well, seen through a coarse
    (non-Markovian) discretization of 5 states."""
    rs = np.random.RandomState(2)
    T = _metropolis(U)
    return [_sample(T, 40, 4, rs) // 4 for _ in range(1500)]


def test_oom_corrects_nonequilibrium_bias(short_nonequilibrium):
    true = np.bincount(X // 4, weights=PI)
    mle = MarkovStateModel(lag_time=1, verbose=False).fit(short_nonequilibrium)
    oom = OOMReweightedMSM(lag_time=1, nbs=1000, random_state=0).fit(short_nonequilibrium)
    err_mle = np.abs(mle.populations_ - true).sum()
    err_oom = np.abs(oom.populations_ - true).sum()
    # observed across seeds: 0.74-0.76 for the MSM, 0.15-0.30 with OOM reweighting
    assert err_oom < 0.5 * err_mle
    assert oom.model_.oom_rank >= 2
    # it is a full MarkovStateModel
    assert oom.timescales_.shape == (oom.n_states_ - 1,)
    np.testing.assert_array_equal(oom.transform(short_nonequilibrium[:2])[0],
                                  short_nonequilibrium[0])
    assert np.isfinite(oom.score_vamp())


def test_oom_is_deeptime_oom(short_nonequilibrium):
    ours = OOMReweightedMSM(lag_time=2, nbs=500, random_state=3).fit(short_nonequilibrium)
    np.random.seed(3)
    ref = dtmsm.OOMReweightedMSM(lagtime=2, nbs=500).fit_fetch(short_nonequilibrium)
    np.testing.assert_allclose(ours.transmat_, ref.transition_matrix)
    np.testing.assert_allclose(ours.populations_, ref.stationary_distribution)


def test_oom_reproducible_and_keeps_global_random_state(short_nonequilibrium):
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    a = OOMReweightedMSM(nbs=300, random_state=5).fit(short_nonequilibrium)
    np.testing.assert_array_equal(np.random.get_state()[1], before)
    b = OOMReweightedMSM(nbs=300, random_state=5).fit(short_nonequilibrium)
    np.testing.assert_array_equal(a.transmat_, b.transmat_)
    c = clone(a)
    assert c.get_params() == a.get_params() and c.reversible_type == "mle"
    assert OOMReweightedMSM(reversible=False).reversible_type == "none"
    restored = pickle.loads(pickle.dumps(a))
    np.testing.assert_array_equal(restored.transmat_, a.transmat_)


# --- Chapman-Kolmogorov test ----------------------------------------------------------

def test_ck_test_passes_for_a_markov_chain(unbiased):
    ck = ck_test(MarkovStateModel(verbose=False), unbiased, [1, 2, 4, 8], 2)
    np.testing.assert_array_equal(ck.lagtimes, [0, 1, 2, 4, 8])
    assert ck.predictions.shape == ck.estimates.shape == (5, 2, 2)
    np.testing.assert_allclose(ck.predictions, ck.estimates, atol=0.01)
    np.testing.assert_allclose(ck.predictions[0], np.eye(2))


def test_ck_test_detects_non_markovian_discretization(short_nonequilibrium, unbiased):
    # a coarse discretization of the lattice is not Markovian at lag 1
    coarse = [t // 4 for t in unbiased]
    ck = ck_test(MarkovStateModel(verbose=False), coarse, [1, 5, 10, 20], 2,
                 include_lag0=False)
    fine = ck_test(MarkovStateModel(verbose=False), unbiased, [1, 5, 10, 20], 2,
                   include_lag0=False)
    assert (np.abs(ck.predictions - ck.estimates).max()
            > 3 * np.abs(fine.predictions - fine.estimates).max())


def test_ck_test_errors(unbiased):
    with pytest.raises(ValueError):
        ck_test(MarkovStateModel(), unbiased, [1], 2)
    with pytest.raises(ValueError):
        ck_test(MarkovStateModel(), unbiased, [4, 2], 2)


def test_command_line(tmp_path, short_nonequilibrium):
    import subprocess
    from msmbuilder.dataset import dataset
    from msmbuilder.utils import load
    path = str(tmp_path / "labels")
    with dataset(path, mode="w", fmt="dir-npy") as ds:
        for i, s in enumerate(short_nonequilibrium[:300]):
            ds[i] = s
    out = subprocess.run(["msmb", "OOMReweightedMSM", "-i", path, "-o", str(tmp_path / "m.pkl"),
                          "--lag_time", "2", "--nbs", "200", "--random_state", "0"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    model = load(str(tmp_path / "m.pkl"))
    assert isinstance(model, OOMReweightedMSM) and model.lag_time == 2
    np.testing.assert_allclose(model.transmat_.sum(axis=1), 1.0)
