import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV, KFold

from msmbuilder.msm import MarkovStateModel

# A non-reversible 5-state chain: a cyclic drift 0 -> 1 -> 2 -> 3 -> 4 -> 0.
P = np.array([[.80, .15, .00, .00, .05],
              [.00, .80, .15, .05, .00],
              [.05, .00, .80, .15, .00],
              [.00, .05, .00, .80, .15],
              [.15, .00, .05, .00, .80]])


def _chains(n_traj, length, seed):
    rs = np.random.RandomState(seed)
    out = []
    for k in range(n_traj):
        s = [k % 5]
        for _ in range(length - 1):
            s.append(rs.choice(5, p=P[s[-1]]))
        out.append(np.array(s))
    return out


TRAIN = _chains(3, 3000, seed=0)
TEST = _chains(2, 2000, seed=1)
# label 9 never appears in the training data
TEST_UNSEEN = [np.where(TEST[0] == 4, 9, TEST[0]), TEST[1]]


@pytest.mark.parametrize("lag_time", [1, 2])
@pytest.mark.parametrize("reversible_type", ["mle", "none"])
@pytest.mark.parametrize("dim", [None, 3])
def test_matches_deeptime(lag_time, reversible_type, dim):
    pytest.importorskip("deeptime")
    from deeptime.markov import TransitionCountModel
    from deeptime.markov.msm import MarkovStateModel as DeeptimeMSM

    m = MarkovStateModel(lag_time=lag_time, reversible_type=reversible_type,
                         verbose=False).fit(TRAIN)
    ref = DeeptimeMSM(m.transmat_,
                      count_model=TransitionCountModel(m.countsmat_, lagtime=lag_time))

    def to_states(seqs):
        unknown = m.n_states_ + 1
        return [np.array([m.mapping_.get(x, unknown) for x in s]) for s in seqs]

    for r in (1, 2):
        for data in (None, TEST, TEST_UNSEEN):
            ours = m.score_vamp(data, method="vamp%d" % r, dim=dim)
            theirs = ref.score(None if data is None else to_states(data), r=r, dim=dim)
            np.testing.assert_allclose(ours, theirs, rtol=1e-9)


def test_default_score_is_still_gmrq():
    m = MarkovStateModel(lag_time=1, verbose=False).fit(TRAIN)
    assert m.score_method == "gmrq"
    gmrq = m.score(TEST)
    m.score_method = "vamp2"
    assert m.score(TEST) == m.score_vamp(TEST)
    assert m.score(TEST) != gmrq


def test_score_method_switches_score_and_score_():
    m = MarkovStateModel(lag_time=1, score_method="vamp2", verbose=False).fit(TRAIN)
    assert m.score_ == m.score_vamp()
    assert m.score(TEST) == m.score_vamp(TEST, method="vamp2")
    m1 = MarkovStateModel(lag_time=1, score_method="VAMP1", verbose=False).fit(TRAIN)
    assert m1.score(TEST) == m1.score_vamp(TEST, method="vamp1")


def test_score_properties():
    # For a non-reversible fit the transition matrix is the row-normalized
    # count matrix, so the training VAMP-2 with all singular functions is the
    # sum of squared singular values, including the constant function (1).
    # (For reversible fits the score is measured against the raw counts.)
    m = MarkovStateModel(lag_time=1, reversible_type="none", verbose=False).fit(TRAIN)
    _, s, _, _ = m._empirical_koopman()
    np.testing.assert_allclose(s[0], 1.0, atol=1e-10)
    np.testing.assert_allclose(m.score_vamp(), np.sum(s ** 2), rtol=1e-10)
    # fewer singular functions can only lower the score
    scores = [m.score_vamp(dim=k) for k in range(1, m.n_states_ + 1)]
    assert np.all(np.diff(scores) >= -1e-12)
    np.testing.assert_allclose(scores[0], 1.0, rtol=1e-10)
    # n_timescales sets the default number of singular functions
    m3 = MarkovStateModel(lag_time=1, n_timescales=2, verbose=False).fit(TRAIN)
    assert m3.score_vamp(TEST) == m3.score_vamp(TEST, dim=3)


def test_dynamics_matter():
    # Shuffling the training data destroys its kinetics; that model should
    # score clearly worse on real held-out dynamics, when scored with fewer
    # singular functions than states.
    rs = np.random.RandomState(5)
    shuffled = [rs.permutation(s) for s in TRAIN]
    real = MarkovStateModel(lag_time=1, reversible_type="none", verbose=False).fit(TRAIN)
    fake = MarkovStateModel(lag_time=1, reversible_type="none", verbose=False).fit(shuffled)
    assert real.score_vamp(TEST, dim=3) > fake.score_vamp(TEST, dim=3) + 0.5


def test_full_dimension_test_score_ignores_the_model():
    # With all n_states singular functions the model's singular vectors span
    # the whole state space, so the held-out score is just the test data's
    # own maximum and cannot tell models apart. (GMRQ behaves the same way.)
    rs = np.random.RandomState(5)
    shuffled = [rs.permutation(s) for s in TRAIN]
    real = MarkovStateModel(lag_time=1, reversible_type="none", verbose=False).fit(TRAIN)
    fake = MarkovStateModel(lag_time=1, reversible_type="none", verbose=False).fit(shuffled)
    np.testing.assert_allclose(real.score_vamp(TEST), fake.score_vamp(TEST), rtol=1e-10)


def test_grid_search_with_vamp():
    # sklearn model selection calls score(), so score_method="vamp2" lets
    # VAMP-2 choose hyperparameters; for this chain, lag 1 beats lag 20.
    search = GridSearchCV(
        MarkovStateModel(score_method="vamp2", n_timescales=3, verbose=False),
        {"lag_time": [1, 20]}, cv=KFold(n_splits=2))
    search.fit(TRAIN + TEST)
    assert search.best_params_["lag_time"] == 1


def test_errors():
    m = MarkovStateModel(lag_time=1, verbose=False).fit(TRAIN)
    with pytest.raises(ValueError, match="VAMP-E"):
        m.score_vamp(TEST, method="vampe")
    m.score_method = "vampe"
    with pytest.raises(ValueError):
        m.score(TEST)
