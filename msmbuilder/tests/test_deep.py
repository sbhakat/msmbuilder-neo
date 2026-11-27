import pickle

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from msmbuilder.decomposition import (SRV, VAMP, TimeLaggedAutoencoder,  # noqa: E402
                                      VAMPNet, tICA)
from msmbuilder.decomposition.deep import vamp_score_torch  # noqa: E402

FAST = dict(hidden_layers=(32, 32), n_epochs=20, batch_size=512, random_state=0)


def _ring(n, rs):
    """Three metastable states on a ring, seen through (cos, sin) + noise:
    the slow process is a nonlinear function of the features."""
    P = np.array([[.98, .01, .01], [.01, .98, .01], [.01, .01, .98]])
    s = [rs.randint(3)]
    for _ in range(n - 1):
        s.append(rs.choice(3, p=P[s[-1]]))
    ang = 2 * np.pi * np.array(s) / 3 + 0.35 * rs.randn(n)
    return np.c_[np.cos(ang), np.sin(ang)] + 0.05 * rs.randn(n, 2)


def _ou(n, rs):
    """Reversible linear (Ornstein-Uhlenbeck-like) process."""
    A = np.array([[0.95, 0.03], [0.03, 0.6]])
    x = np.zeros((n, 2))
    for t in range(1, n):
        x[t] = A @ x[t - 1] + rs.randn(2)
    return x


@pytest.fixture(scope="module")
def ring():
    rs = np.random.RandomState(0)
    return [_ring(3000, rs) for _ in range(3)], [_ring(3000, rs) for _ in range(2)]


@pytest.fixture(scope="module")
def ou():
    rs = np.random.RandomState(1)
    return [_ou(5000, rs) for _ in range(3)]


@pytest.mark.parametrize("method", ["VAMP1", "VAMP2", "VAMPE"])
@pytest.mark.parametrize("mode", ["trunc", "regularize", "clamp"])
def test_loss_matches_deeptime(method, mode):
    deep = pytest.importorskip("deeptime.decomposition.deep")
    rs = np.random.RandomState(2)
    x = rs.randn(800, 6)
    y = x @ (0.3 * rs.randn(6, 6)) + 0.5 * rs.randn(800, 6)
    for dtype in (torch.float32, torch.float64):
        X, Y = torch.tensor(x, dtype=dtype), torch.tensor(y, dtype=dtype)
        ours = float(vamp_score_torch(X, Y, method, 1e-6, mode))
        ref = float(deep.vamp_score(X, Y, method, 1e-6, mode))
        np.testing.assert_allclose(ours, ref, rtol=1e-10)


def test_loss_is_differentiable():
    X = torch.tensor(np.random.RandomState(3).randn(300, 4), dtype=torch.float32,
                     requires_grad=True)
    vamp_score_torch(X[:-1], X[1:]).backward()
    assert torch.isfinite(X.grad).all()


@pytest.mark.parametrize("output", ["softmax", "linear"])
def test_vampnet_beats_linear_vamp_on_nonlinear_data(ring, output):
    train, test = ring
    n = 3 if output == "softmax" else 2
    net = VAMPNet(n_components=n, lag_time=5, output=output, **FAST).fit(train)
    linear = VAMP(n_components=2, lag_time=5).fit(train)
    assert net.score(test) > linear.score(test) + 0.15
    assert net.loss_history_[-1] < net.loss_history_[0]
    assert len(net.singular_values_) == len(net.timescales_)


def test_vampnet_softmax_outputs_are_probabilities(ring):
    train, _ = ring
    Y = VAMPNet(n_components=3, lag_time=5, **FAST).fit(train).transform(train)[0]
    assert Y.shape == (3000, 3)
    assert np.all(Y >= 0)
    np.testing.assert_allclose(Y.sum(axis=1), 1.0, atol=1e-5)


def test_srv_finds_nonlinear_eigenfunction(ou):
    # For this linear process the slowest eigenfunction is linear (tICA finds
    # it) and the next one is its square, with eigenvalue lambda_1 ** 2 --
    # which tICA cannot represent but the SRV can.
    srv = SRV(n_components=2, lag_time=1, hidden_layers=(16,), n_epochs=20,
              batch_size=512, random_state=0).fit(ou)
    lam1 = np.sort(tICA(lag_time=1).fit(ou).eigenvalues_)[::-1][0]
    np.testing.assert_allclose(srv.eigenvalues_[0], lam1, atol=0.01)
    np.testing.assert_allclose(srv.eigenvalues_[1], lam1 ** 2, atol=0.02)
    assert srv.eigenvalues_[0] >= srv.eigenvalues_[1]
    np.testing.assert_allclose(srv.timescales_, -1 / np.log(srv.eigenvalues_))
    # outputs are the ordered eigenfunctions: uncorrelated, unit variance
    Y = np.concatenate(srv.transform(ou))
    np.testing.assert_allclose(np.cov(Y.T), np.eye(2), atol=0.05)


def test_tae_recovers_slow_coordinate():
    rs = np.random.RandomState(4)
    slow = [np.cumsum(rs.randn(5000)) * 0.02 for _ in range(3)]
    data = [np.c_[np.sin(s) + 0.1 * rs.randn(5000), rs.randn(5000, 3)] for s in slow]
    tae = TimeLaggedAutoencoder(n_components=1, lag_time=5, hidden_layers=(16,),
                                n_epochs=20, batch_size=512, random_state=0).fit(data)
    z = np.concatenate(tae.transform(data))[:, 0]
    target = np.concatenate([np.sin(s) for s in slow])
    X = np.concatenate(data)
    # the latent variable follows the slow coordinate, not the fast noise
    # (observed: 0.95-0.99 vs <= 0.02 across data and initialization seeds)
    assert abs(np.corrcoef(z, target)[0, 1]) > 0.9
    assert max(abs(np.corrcoef(z, X[:, k])[0, 1]) for k in (1, 2, 3)) < 0.1
    assert tae.loss_history_[-1] < tae.loss_history_[0]
    assert tae.reconstruction_error(data) < tae.loss_history_[0]
    assert tae.score(data) > 1.0


@pytest.mark.parametrize("cls", [VAMPNet, SRV, TimeLaggedAutoencoder])
def test_reproducible_picklable_and_cloneable(ring, cls):
    train, test = ring
    kwargs = dict(lag_time=5, hidden_layers=(8,), n_epochs=2, batch_size=256, random_state=7)
    a = cls(**kwargs).fit(train)
    b = cls(**kwargs).fit(train)
    np.testing.assert_array_equal(a.transform(test)[0], b.transform(test)[0])
    restored = pickle.loads(pickle.dumps(a))
    np.testing.assert_array_equal(restored.transform(test)[0], a.transform(test)[0])
    c = clone(a)
    assert c.get_params() == a.get_params()
    assert not hasattr(c, "loss_history_")


def test_errors(ring):
    train, _ = ring
    with pytest.raises(ValueError):
        VAMPNet(output="sigmoid").fit(train)
    with pytest.raises(ValueError):
        SRV(activation="swish").fit(train)
    with pytest.raises(ValueError):
        TimeLaggedAutoencoder(lag_time=10000).fit(train)
    with pytest.raises(ValueError):
        VAMPNet().transform(train)


def test_command_line(tmp_path):
    import subprocess
    from msmbuilder.dataset import dataset
    from msmbuilder.utils import load
    rs = np.random.RandomState(5)
    seqs = [_ou(400, rs) for _ in range(2)]
    feats = str(tmp_path / "features")
    with dataset(feats, mode="w", fmt="dir-npy") as ds:
        for i, s in enumerate(seqs):
            ds[i] = s
    out = subprocess.run(
        ["msmb", "SRV", "-i", feats, "-o", str(tmp_path / "srv.pkl"),
         "-t", str(tmp_path / "srv.h5"), "--n_components", "1", "--lag_time", "2",
         "--hidden_layers", "8,8", "--n_epochs", "2", "--batch_size", "128",
         "--random_state", "0"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    model = load(str(tmp_path / "srv.pkl"))
    assert model.hidden_layers == (8, 8)
    np.testing.assert_allclose(dataset(str(tmp_path / "srv.h5"))[0], model.transform(seqs)[0],
                               rtol=1e-6)
