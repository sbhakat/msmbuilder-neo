"""DeepTICA and DeepLDA, checked against mlcolvar where it is installed."""
import pickle

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from msmbuilder.decomposition import (SRV, VAMPNet, DeepLDA, DeepTICA,  # noqa: E402
                                      TimeLaggedAutoencoder, tICA)
from msmbuilder.decomposition.deep import (_EIG_LOSSES, _cholesky_eigh,  # noqa: E402
                                           _lda_eigen, _lorentzian_penalty,
                                           _reduce_eigenvalues, _rescaled_time,
                                           _rescaled_time_pairs, _tica_eigen)

FAST = dict(hidden_layers=(16, 16), n_epochs=20, batch_size=512, random_state=0)
D = torch.float64


def _mlcolvar():
    return pytest.importorskip("mlcolvar")


@pytest.fixture
def float64():
    # mlcolvar creates some intermediate tensors in the default dtype; run
    # the comparisons in double precision
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _ring(n, rs):
    """Three metastable states on a ring, seen through (cos, sin) + noise."""
    P = np.array([[.98, .01, .01], [.01, .98, .01], [.01, .01, .98]])
    s = [rs.randint(3)]
    for _ in range(n - 1):
        s.append(rs.choice(3, p=P[s[-1]]))
    ang = 2 * np.pi * np.array(s) / 3 + 0.35 * rs.randn(n)
    return np.c_[np.cos(ang), np.sin(ang)] + 0.05 * rs.randn(n, 2)


@pytest.fixture(scope="module")
def ring():
    rs = np.random.RandomState(0)
    return [_ring(3000, rs) for _ in range(3)]


def _pairs(rs, n=600, d=5):
    x = rs.randn(n, d) @ rs.randn(d, d)
    y = 0.8 * x + 0.5 * rs.randn(n, d)
    return torch.tensor(x, dtype=D), torch.tensor(y, dtype=D)


# --- statistics: identical to mlcolvar -----------------------------------------

@pytest.mark.parametrize("weighted", [False, True])
def test_tica_matches_mlcolvar(weighted, float64):
    from mlcolvar.core.stats import TICA
    _mlcolvar()
    rs = np.random.RandomState(0)
    x, y = _pairs(rs)
    if weighted:
        w0, wt = (torch.tensor(rs.rand(len(x)) + 0.1, dtype=D) for _ in range(2))
    else:
        w0, wt = None, None
    for n_eig in (2, 5):
        tica = TICA(5, n_eig)
        ones = torch.ones(len(x), dtype=D)
        ref_val, ref_vec = tica.compute([x, y], [ones if w0 is None else w0,
                                                 ones if wt is None else wt], save_params=True)
        val, vec, mean = _tica_eigen(x, y, w0, wt, 1e-6, n_eig)
        np.testing.assert_allclose(val, ref_val, rtol=1e-10)
        np.testing.assert_allclose(vec, ref_vec, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(mean, tica.mean, rtol=1e-12)


@pytest.mark.parametrize("mode", ["standard", "harmonic"])
@pytest.mark.parametrize("n_states", [2, 4])
def test_lda_matches_mlcolvar(mode, n_states, float64):
    from mlcolvar.core.stats import LDA
    _mlcolvar()
    rs = np.random.RandomState(1)
    labels = torch.tensor(rs.randint(n_states, size=800))
    h = torch.tensor(rs.randn(800, 5), dtype=D) + labels[:, None].double() * torch.tensor(
        rs.randn(5), dtype=D)
    lda = LDA(5, n_states, mode=mode)
    lda.sw_reg = 0.05
    ref_val, ref_vec = lda.compute(h, labels)
    val, vec = _lda_eigen(h, labels, 0.05, mode, n_states - 1)
    np.testing.assert_allclose(val, ref_val, rtol=1e-10)
    np.testing.assert_allclose(vec, ref_vec, rtol=1e-8, atol=1e-12)


def test_cholesky_eigh_matches_mlcolvar(float64):
    from mlcolvar.core.stats.utils import cholesky_eigh
    _mlcolvar()
    rs = np.random.RandomState(2)
    a = rs.randn(6, 6)
    b = rs.randn(6, 6)
    A = torch.tensor(a + a.T, dtype=D)
    B = torch.tensor(b @ b.T + np.eye(6), dtype=D)
    for reg, n in [(1e-6, None), (0.1, 3), (None, 2)]:
        ref = cholesky_eigh(A, B, reg, n)
        ours = _cholesky_eigh(A, B, reg, n)
        np.testing.assert_allclose(ours[0], ref[0], rtol=1e-10)
        np.testing.assert_allclose(ours[1], ref[1], rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("mode", _EIG_LOSSES)
def test_eigenvalue_losses_match_mlcolvar(mode):
    from mlcolvar.core.loss.eigvals import reduce_eigenvalues_loss
    _mlcolvar()
    ev = torch.tensor([0.97, 0.9, 0.5, 0.2], dtype=D)
    for n_eig in (0, 1, 3):
        if n_eig == 0 and mode in ("single", "single2"):
            with pytest.raises(ValueError):
                _reduce_eigenvalues(ev, mode, n_eig)
            continue
        np.testing.assert_allclose(float(_reduce_eigenvalues(ev, mode, n_eig)),
                                   float(reduce_eigenvalues_loss(ev, mode, n_eig)), rtol=1e-12)


def test_rescaled_time_matches_mlcolvar():
    from mlcolvar.utils.timelagged import tprime_evaluation
    _mlcolvar()
    lw = np.random.RandomState(3).randn(500) * 3
    ref = tprime_evaluation(np.arange(500), lw).numpy()   # float32 in mlcolvar
    np.testing.assert_allclose(_rescaled_time(lw), ref, rtol=2e-5)


@pytest.mark.parametrize("lag", [0.5, 3.0, 17.25])
def test_rescaled_time_pairs_match_mlcolvar(lag):
    from mlcolvar.utils.timelagged import find_timelagged_configurations, tprime_evaluation
    _mlcolvar()
    rs = np.random.RandomState(4)
    X = rs.randn(400, 2)
    tprime = tprime_evaluation(np.arange(400), rs.randn(400) * 2)
    ref = find_timelagged_configurations(torch.tensor(X), tprime, lag_time=lag,
                                         progress_bar=False)
    i, j, w0, wt = _rescaled_time_pairs(tprime.double().numpy(), lag)
    assert len(i) == len(ref[0]) > 0
    np.testing.assert_array_equal(X[i], ref[0].numpy())
    np.testing.assert_array_equal(X[j], ref[1].numpy())
    # mlcolvar subtracts times in float32: allow its rounding error
    atol = 4 * np.finfo(np.float32).eps * float(tprime.max())
    np.testing.assert_allclose(w0, ref[2].numpy(), rtol=0, atol=atol)
    np.testing.assert_allclose(wt, ref[3].numpy(), rtol=0, atol=atol)


def test_rescaled_time_pairs_with_unit_steps():
    # in unbiased time, the pairs are (i, i + lag) with unit weights
    i, j, w0, wt = _rescaled_time_pairs(np.arange(1.0, 101.0), 4)
    np.testing.assert_array_equal(j - i, 4)
    np.testing.assert_array_equal(i, np.arange(len(i)))
    np.testing.assert_allclose(w0, 1.0)
    np.testing.assert_allclose(wt, 1.0)


# --- the models: same network output and loss as mlcolvar's ------------------------

def _copy_into_mlcolvar(est, model, block):
    state = {"norm_in.mean": torch.tensor(est.feature_means_, dtype=torch.float32),
             "norm_in.range": torch.tensor(est.feature_scales_, dtype=torch.float32)}
    for k, v in est.net_.state_dict().items():
        state["nn.nn." + k] = v
    state[block + ".evecs"] = torch.tensor(est._evecs, dtype=torch.float32)
    if block == "tica":
        state["tica.mean"] = torch.tensor(est._tica_mean, dtype=torch.float32)
    model.load_state_dict(state)
    return model.eval()


def test_deeptica_matches_mlcolvar_model(ring):
    from mlcolvar.cvs import DeepTICA as MLDeepTICA
    _mlcolvar()
    est = DeepTICA(n_components=2, lag_time=5, nn_outputs=3, **FAST).fit(ring)
    model = _copy_into_mlcolvar(est, MLDeepTICA([2, 16, 16, 3], n_cvs=2,
                                                options={"nn": {"activation": "tanh"}}), "tica")
    X = torch.tensor(ring[0], dtype=torch.float32)
    with torch.no_grad():
        np.testing.assert_allclose(model(X).numpy(), est.transform(ring[:1])[0], atol=2e-5)

    # the training objective on a batch of (weighted) pairs
    model = model.double()
    rs = np.random.RandomState(5)
    x0 = torch.tensor(est._standardize(ring[1][:-5]), dtype=D)
    xt = torch.tensor(est._standardize(ring[1][5:]), dtype=D)
    w0, wt = (torch.tensor(rs.rand(len(x0)), dtype=D) for _ in range(2))
    net = est.net_.double()
    ours = _reduce_eigenvalues(_tica_eigen(net(x0), net(xt), w0, wt, 1e-6, 2)[0], "sum2")
    f0 = model.nn(x0)
    ft = model.nn(xt)
    ref = model.loss_fn(model.tica.compute([f0, ft], [w0, wt])[0])
    np.testing.assert_allclose(float(ours), float(ref), rtol=1e-10)
    est.net_.float()


def test_deeplda_matches_mlcolvar_model(float64):
    from mlcolvar.cvs import DeepLDA as MLDeepLDA
    _mlcolvar()
    rs = np.random.RandomState(6)
    data = [rs.randn(300, 3) + c for c in ([2, 0, 0], [-2, 0, 0], [0, 2, 0])]
    est = DeepLDA(nn_outputs=4, **FAST).fit(data, [0, 1, 2])
    model = _copy_into_mlcolvar(est, MLDeepLDA([3, 16, 16, 4], n_states=3,
                                               options={"nn": {"activation": "tanh"}}), "lda")
    X = torch.tensor(np.concatenate(data), dtype=D)
    with torch.no_grad():
        np.testing.assert_allclose(model(X).numpy(), np.concatenate(est.transform(data)),
                                   atol=2e-5)

    # the training objective, Lorentzian regularization included
    x = torch.tensor(np.concatenate([est._standardize(d) for d in data]), dtype=D)
    y = torch.tensor(np.repeat([0, 1, 2], 300))
    h = model.nn(x)
    val, _ = model.lda.compute(h, y, save_params=True)
    ref = model.loss_fn(val) + model.regularization_lorentzian(model.lda(h))
    net = est.net_.double()
    hv = net(x)
    ev, evec = _lda_eigen(hv, y, 0.05, "standard", 2)
    ours = _reduce_eigenvalues(ev, "sum") + _lorentzian_penalty(hv @ evec, 2 / 0.05)
    np.testing.assert_allclose(float(ours), float(ref), rtol=1e-10)
    est.net_.float()


# --- behaviour ----------------------------------------------------------------

def test_deeptica_beats_tica_on_nonlinear_data(ring):
    est = DeepTICA(n_components=1, lag_time=5, nn_outputs=2, **FAST).fit(ring)
    lin = tICA(n_components=1, lag_time=5).fit(ring)
    # (observed margin 0.06-0.08 across data and initialization seeds)
    assert est.eigenvalues_[0] > np.max(lin.eigenvalues_) + 0.03
    assert est.loss_history_[-1] < est.loss_history_[0]
    np.testing.assert_allclose(est.timescales_, -5 / np.log(np.abs(est.eigenvalues_)))
    assert est.score(ring) > 1.0


def test_deeptica_reweighting(ring):
    kw = dict(n_components=2, lag_time=5, **FAST)
    plain = DeepTICA(**kw).fit(ring)
    # zero log-weights: weights_t is the unweighted estimator
    zeros = [np.zeros(len(X)) for X in ring]
    flat = DeepTICA(reweighting="weights_t", **kw).fit(ring, logweights=zeros)
    np.testing.assert_allclose(flat.transform(ring)[0], plain.transform(ring)[0], atol=1e-5)
    # a constant shift of the log-weights changes nothing
    shifted = DeepTICA(reweighting="weights_t", **kw).fit(ring, logweights=[z + 50 for z in zeros])
    np.testing.assert_allclose(shifted.transform(ring)[0], flat.transform(ring)[0], atol=1e-6)
    # rescaled time with a real bias trains and gives valid eigenvalues
    rs = np.random.RandomState(7)
    lw = [rs.rand(len(X)) for X in ring]
    est = DeepTICA(**dict(kw, lag_time=2.5)).fit(ring, logweights=lw)
    assert np.all(np.abs(est.eigenvalues_) <= 1 + 1e-6)
    assert est.transform(ring)[0].shape == (3000, 2)


def test_deeplda_separates_nonlinear_classes():
    # an inner disc and an outer ring: not linearly separable
    rs = np.random.RandomState(8)
    ang = rs.uniform(0, 2 * np.pi, (2, 800))
    rad = np.array([rs.uniform(0, 1, 800), rs.uniform(2, 3, 800)])
    data = [np.c_[r * np.cos(a), r * np.sin(a)] for r, a in zip(rad, ang)]
    est = DeepLDA(hidden_layers=(16, 16), n_epochs=150, learning_rate=1e-2, batch_size=400,
                  random_state=0).fit(data, ["in", "out"])
    linear, _ = _lda_eigen(torch.tensor(np.concatenate(data)),
                           torch.tensor(np.repeat([0, 1], 800)), 0.05, "standard", 1)
    assert est.n_components_ == 1 and list(est.classes_) == ["in", "out"]
    assert float(linear[0]) < 0.05
    # observed across data and initialization seeds: score 22-26 and class
    # means 17-63 (sum of) standard deviations apart
    assert est.score(data, ["in", "out"]) > 10
    s = [x[:, 0] for x in est.transform(data)]
    assert abs(s[0].mean() - s[1].mean()) > 5 * (s[0].std() + s[1].std())


@pytest.mark.parametrize("make", [
    lambda: (VAMPNet(n_components=3, lag_time=5, **FAST), {}),
    lambda: (SRV(n_components=2, lag_time=5, **FAST), {}),
    lambda: (TimeLaggedAutoencoder(n_components=2, lag_time=5, **FAST), {}),
    lambda: (DeepTICA(n_components=2, lag_time=5, **FAST), {}),
    lambda: (DeepLDA(**FAST), {"y": [0, 1, 2]}),
], ids=["VAMPNet", "SRV", "TAE", "DeepTICA", "DeepLDA"])
def test_torchscript_export(ring, make, tmp_path):
    est, kw = make()
    est.fit(ring, **kw)
    path = str(tmp_path / "model.pt")
    est.to_torchscript(path)
    loaded = torch.jit.load(path)
    with torch.no_grad():
        out = loaded(torch.tensor(ring[1], dtype=torch.float32)).numpy()
    np.testing.assert_allclose(out, est.transform(ring)[1], atol=1e-5)


@pytest.mark.parametrize("cls,kw", [(DeepTICA, {}), (DeepLDA, {"y": [0, 1, 2]})])
def test_reproducible_picklable_and_cloneable(ring, cls, kw):
    params = dict(hidden_layers=(8,), n_epochs=2, batch_size=256, random_state=3)
    a = cls(**params).fit(ring, **kw)
    b = cls(**params).fit(ring, **kw)
    np.testing.assert_array_equal(a.transform(ring)[0], b.transform(ring)[0])
    restored = pickle.loads(pickle.dumps(a))
    np.testing.assert_array_equal(restored.transform(ring)[0], a.transform(ring)[0])
    c = clone(a)
    assert c.get_params() == a.get_params()
    assert not hasattr(c, "loss_history_")


def test_deeplda_labels(ring):
    per_frame = [np.where(np.arange(len(X)) < 1500, "a", "b") for X in ring]
    est = DeepLDA(hidden_layers=(8,), n_epochs=2, random_state=0).fit(ring, per_frame)
    assert list(est.classes_) == ["a", "b"]
    assert np.isfinite(est.score(ring, per_frame))


def test_errors(ring):
    with pytest.raises(ValueError):
        DeepTICA(loss_mode="max").fit(ring)
    with pytest.raises(ValueError):
        DeepTICA(n_components=3, nn_outputs=2).fit(ring)
    with pytest.raises(ValueError):
        DeepTICA(reweighting="other").fit(ring)
    with pytest.raises(ValueError):
        DeepTICA(lag_time=2.5).fit(ring)        # fractional lag needs rescaled time
    with pytest.raises(ValueError):
        DeepTICA().fit(ring, logweights=[np.zeros(10)] * 3)
    with pytest.raises(ValueError):
        DeepLDA().fit(ring)                     # labels are required
    with pytest.raises(ValueError):
        DeepLDA().fit(ring, [0, 0, 0])          # one class
    with pytest.raises(ValueError):
        DeepLDA().fit(ring, [0, 1])             # wrong length
    with pytest.raises(ValueError):
        DeepLDA(lda_mode="other").fit(ring, [0, 1, 2])
    with pytest.raises(ValueError):
        DeepTICA().to_torchscript()             # not fitted
    est = DeepLDA(hidden_layers=(8,), n_epochs=1).fit(ring, [0, 1, 2])
    with pytest.raises(ValueError):
        est.score(ring, [0, 1, 5])              # unseen label


def test_command_line(tmp_path):
    import subprocess
    from msmbuilder.dataset import dataset
    from msmbuilder.utils import load
    rs = np.random.RandomState(9)
    seqs = [np.cumsum(rs.randn(400, 3), axis=0) * 0.1 for _ in range(2)]
    feats = str(tmp_path / "features")
    with dataset(feats, mode="w", fmt="dir-npy") as ds:
        for i, s in enumerate(seqs):
            ds[i] = s
    out = subprocess.run(
        ["msmb", "DeepTICA", "-i", feats, "-o", str(tmp_path / "m.pkl"),
         "-t", str(tmp_path / "m.h5"), "--n_components", "1", "--lag_time", "2",
         "--hidden_layers", "8,8", "--n_epochs", "2", "--batch_size", "128",
         "--random_state", "0"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    model = load(str(tmp_path / "m.pkl"))
    assert model.hidden_layers == (8, 8)
    np.testing.assert_allclose(dataset(str(tmp_path / "m.h5"))[0], model.transform(seqs)[0],
                               rtol=1e-6)
