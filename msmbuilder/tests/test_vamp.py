import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from msmbuilder.cluster import KCenters
from msmbuilder.decomposition import VAMP, tICA
from msmbuilder.msm import MarkovStateModel

# Linear, non-reversible dynamics: a damped rotation in features 0-1 and a
# coupled decay in features 2-3, with nonzero means.
_A = np.array([[0.95, 0.20, 0.0, 0.0],
               [-0.20, 0.95, 0.0, 0.0],
               [0.0, 0.0, 0.6, 0.0],
               [0.0, 0.0, 0.1, 0.3]])
_MEAN = np.array([1.0, -2.0, 0.5, 3.0])


def _trajectories(n_traj, length, seed):
    rs = np.random.RandomState(seed)
    out = []
    for _ in range(n_traj):
        x = np.zeros((length, 4))
        for t in range(1, length):
            x[t] = _A @ x[t - 1] + 0.3 * rs.randn(4)
        out.append(x + _MEAN)
    return out


TRAIN = _trajectories(3, 3000, seed=0)
TEST = _trajectories(2, 2000, seed=1)


@pytest.mark.parametrize("lag_time", [1, 3])
@pytest.mark.parametrize("n_components", [None, 2])
def test_matches_deeptime(lag_time, n_components):
    deeptime = pytest.importorskip("deeptime")
    DT = deeptime.decomposition.VAMP

    ours = VAMP(lag_time=lag_time, n_components=n_components).fit(TRAIN)
    ref = DT(lagtime=lag_time, dim=n_components, epsilon=1e-6).fit(TRAIN).fetch_model()
    ref_test = DT(lagtime=lag_time, dim=n_components, epsilon=1e-6).fit(TEST).fetch_model()

    k = len(ours.singular_values_)
    np.testing.assert_allclose(ours.singular_values_, ref.singular_values[:k], rtol=1e-10)
    np.testing.assert_allclose(ours.means_, ref.cov.mean_0, rtol=1e-12)

    for method, r in [("VAMP1", 1), ("VAMP2", 2), ("VAMPE", "E")]:
        ours.score_method = method
        np.testing.assert_allclose(ours.score(TRAIN), ref.score(r), rtol=1e-9)
        np.testing.assert_allclose(ours.score(TEST), ref.score(r, test_model=ref_test),
                                   rtol=1e-9)

    # projections agree up to the sign of each singular function
    ours_proj = np.concatenate(ours.transform(TEST))
    ref_proj = np.concatenate([ref.transform(X) for X in TEST])[:, :k]
    signs = np.sign(np.sum(ours_proj * ref_proj, axis=0))
    np.testing.assert_allclose(ours_proj, ref_proj * signs, atol=1e-8)


def test_scores_are_consistent():
    m = VAMP(lag_time=2).fit(TRAIN)
    assert np.all(np.diff(m.singular_values_) <= 0)
    assert np.all((m.singular_values_ > 0) & (m.singular_values_ < 1))
    # training VAMP-r score is 1 + sum(sigma_i ** r)
    np.testing.assert_allclose(m.score_, 1 + np.sum(m.singular_values_ ** 2))
    m.score_method = "VAMP1"
    np.testing.assert_allclose(m.score(TRAIN), 1 + np.sum(m.singular_values_))
    # keeping fewer components can only lower the score
    scores = [VAMP(lag_time=2, n_components=k).fit(TRAIN).score_ for k in (1, 2, 3, 4)]
    assert np.all(np.diff(scores) >= 0)
    # held-out data scores no higher than training data, but stays close
    full = VAMP(lag_time=2).fit(TRAIN)
    assert 1.0 < full.score(TEST) <= full.score_ + 0.1


def test_projection_is_whitened_and_kinetic_map_scales():
    m = VAMP(lag_time=1).fit(TRAIN)
    Y = np.concatenate([y[:-1] for y in m.transform(TRAIN)])
    # left singular functions are uncorrelated with unit variance
    np.testing.assert_allclose(np.cov(Y.T, bias=True), np.eye(Y.shape[1]), atol=1e-2)
    km = VAMP(lag_time=1, scaling="kinetic_map").fit(TRAIN)
    np.testing.assert_allclose(km.transform(TRAIN)[0],
                               m.transform(TRAIN)[0] * m.singular_values_, rtol=1e-10)


def test_timescales_and_summary():
    m = VAMP(lag_time=2).fit(TRAIN)
    np.testing.assert_allclose(m.timescales_, -2 / np.log(m.singular_values_))
    assert m.components_.shape == (4, 4)
    assert "VAMP" in m.summarize()


def test_sklearn_compatibility():
    m = VAMP(n_components=2, lag_time=3, score_method="VAMPE")
    c = clone(m)
    assert c.get_params() == m.get_params()
    pipe = Pipeline([("vamp", VAMP(n_components=2, lag_time=1)),
                     ("cluster", KCenters(n_clusters=5, random_state=0)),
                     ("msm", MarkovStateModel(lag_time=1, verbose=False))])
    pipe.fit(TRAIN)
    assert pipe.named_steps["msm"].n_states_ == 5


def test_reversible_data_agrees_with_tica():
    # For reversible equilibrium data the leading VAMP singular values and
    # tICA eigenvalues estimate the same quantity.
    rs = np.random.RandomState(3)
    sym = np.array([[0.9, 0.05], [0.05, 0.5]])
    X = []
    for _ in range(4):
        x = np.zeros((20000, 2))
        for t in range(1, len(x)):
            x[t] = sym @ x[t - 1] + rs.randn(2)
        X.append(x)
    v = VAMP(lag_time=1).fit(X)
    t = tICA(lag_time=1).fit(X)
    np.testing.assert_allclose(v.singular_values_, np.sort(t.eigenvalues_)[::-1], atol=0.02)


def test_errors():
    with pytest.raises(ValueError):
        VAMP(lag_time=0).fit(TRAIN)
    with pytest.raises(ValueError):
        VAMP(score_method="VAMP3").fit(TRAIN)
    with pytest.raises(ValueError):
        VAMP(lag_time=10).fit([np.zeros((5, 2))])
    with pytest.raises(ValueError):
        VAMP().score(TRAIN)
