"""Neural-network estimators of slow collective variables.

* :class:`VAMPNet` -- VAMPnets [1]_: a network trained to maximize a VAMP
  score of its outputs. With softmax outputs it learns soft metastable states.
* :class:`SRV` -- state-free reversible VAMPnets [2]_: a network trained to
  maximize the reversible VAMP-2 score; its outputs are ordered
  eigenfunctions with implied timescales.
* :class:`TimeLaggedAutoencoder` -- time-lagged autoencoders [3]_: an
  encoder-decoder trained to predict x(t + lag) from x(t) through a
  low-dimensional bottleneck.
* :class:`DeepTICA` -- Deep-TICA [4]_: a network whose outputs, after a TICA
  rotation, maximize the TICA eigenvalues; can train on biased (reweighted)
  simulations.
* :class:`DeepLDA` -- Deep-LDA [5]_: a supervised network whose outputs,
  after an LDA rotation, best separate labelled states.

DeepTICA and DeepLDA follow the definitions of the mlcolvar package [6]_
(losses, regularizations, eigenvector normalization and time-lagged pairing
of reweighted data are checked against it in the tests), without depending on
it. Every estimator can export its trained network, including input
standardization and output rotation, as TorchScript for PLUMED's PYTORCH
module (:meth:`~_DeepEstimator.to_torchscript`).

They follow the MSMBuilder API: ``fit``, ``transform`` and ``score`` take
lists of feature sequences, and they work in scikit-learn pipelines and
model selection. ``score`` has the same meaning for all three: a linear
:class:`~msmbuilder.decomposition.VAMP` model is fitted to the training-set
embedding and its VAMP score evaluated on the embedding of the new data, so
different architectures, lag times and estimators can be compared.

PyTorch is an optional dependency, imported only when an estimator is
fitted: ``pip install msmbuilder-neo[torch]``.

References
----------
.. [1] Mardt, A., Pasquali, L., Wu, H. and Noe, F. "VAMPnets for deep
   learning of molecular kinetics." Nat. Commun. 9, 5 (2018).
.. [2] Chen, W., Sidky, H. and Ferguson, A. L. "Nonlinear discovery of slow
   molecular modes using state-free reversible VAMPnets." J. Chem. Phys.
   150, 214114 (2019).
.. [3] Wehmeyer, C. and Noe, F. "Time-lagged autoencoders: Deep learning of
   slow collective variables for molecular kinetics." J. Chem. Phys. 148,
   241703 (2018).
.. [4] Bonati, L., Piccini, G. and Parrinello, M. "Deep learning the slow
   modes for rare events sampling." PNAS 118, e2113533118 (2021).
.. [5] Bonati, L., Rizzi, V. and Parrinello, M. "Data-driven collective
   variables for enhanced sampling." J. Phys. Chem. Lett. 11, 2998-3004
   (2020).
.. [6] Bonati, L., Trizio, E., Rizzi, A. and Parrinello, M. "A unified
   framework for machine learning collective variables for enhanced
   sampling simulations: mlcolvar." J. Chem. Phys. 159, 014801 (2023).
"""
import copy

import numpy as np
from numba import njit
from sklearn.base import TransformerMixin

from ..base import BaseEstimator
from ..utils import array2d, check_iter_of_sequences
from .vamp import VAMP

__all__ = ['VAMPNet', 'SRV', 'TimeLaggedAutoencoder', 'DeepTICA', 'DeepLDA',
           'vamp_score_torch']

_ACTIVATIONS = ('elu', 'relu', 'tanh', 'sigmoid', 'softplus', 'gelu', 'silu')


def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover - exercised only without torch
        raise ImportError(
            'This estimator needs PyTorch. Install it with "pip install torch" '
            'or "pip install msmbuilder-neo[torch]".') from e
    return torch


# --- losses ------------------------------------------------------------------

def _covariances(x, y):
    """Mean-free covariances with 1/(n-1) normalization (as deeptime)."""
    n = x.shape[0]
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    return x.T @ x / (n - 1), x.T @ y / (n - 1), y.T @ y / (n - 1)


def _sym_inverse_sqrt(mat, epsilon, mode):
    torch = _torch()
    eigval, eigvec = torch.linalg.eigh(mat)
    if mode == 'trunc':
        mask = eigval > epsilon
        eigval, eigvec = eigval[mask], eigvec[:, mask]
    elif mode == 'regularize':
        eigval = torch.abs(eigval) + epsilon
    elif mode == 'clamp':
        eigval = torch.clamp_min(eigval, epsilon)
    else:
        raise ValueError("mode must be 'trunc', 'regularize' or 'clamp', got %r" % mode)
    return eigvec @ torch.diag(1.0 / torch.sqrt(eigval)) @ eigvec.T


def vamp_score_torch(x, y, method='VAMP2', epsilon=1e-6, mode='trunc'):
    """Differentiable VAMP score of paired, time-lagged network outputs.

    Parameters
    ----------
    x, y : torch.Tensor, shape (n_pairs, n_features)
        Outputs at times t and t + lag.
    method : {'VAMP1', 'VAMP2', 'VAMPE'}
    epsilon : float
        Eigenvalue threshold or regularizer for the covariance inverses.
    mode : {'trunc', 'regularize', 'clamp'}
        How small covariance eigenvalues are handled: dropped, shifted by
        ``epsilon`` after taking their absolute value, or clamped.

    Returns
    -------
    score : torch.Tensor
        Scalar, including +1 for the constant singular function. Computed in
        double precision (gradients flow back to float32 inputs), as in
        deeptime's ``vamp_score``.
    """
    torch = _torch()
    method = str(method).upper().replace('-', '')
    if method not in ('VAMP1', 'VAMP2', 'VAMPE'):
        raise ValueError("method must be 'VAMP1', 'VAMP2' or 'VAMPE', got %r" % method)
    dtype = x.dtype
    x, y = x.double(), y.double()
    c00, c0t, ctt = _covariances(x, y)
    c00_inv = _sym_inverse_sqrt(c00, epsilon, mode)
    ctt_inv = _sym_inverse_sqrt(ctt, epsilon, mode)
    if method == 'VAMPE':
        koopman = (c00_inv @ c0t @ ctt_inv).T
        u, s, v = torch.linalg.svd(koopman, full_matrices=False)
        v = v.T
        mask = s > epsilon
        u = c00_inv @ u[:, mask]
        v = ctt_inv @ v[:, mask]
        S = torch.diag(s[mask])
        out = torch.trace(2.0 * S @ u.T @ c0t @ v - S @ u.T @ c00 @ u @ S @ v.T @ ctt @ v)
    else:
        koopman = (c00_inv @ c0t @ ctt_inv).T
        if method == 'VAMP1':
            out = torch.linalg.matrix_norm(koopman, ord='nuc')
        else:
            out = torch.linalg.matrix_norm(koopman, ord='fro') ** 2
    return (1 + out).to(dtype)


def _reversible_eigen(x, y, epsilon):
    """Eigenvalues/vectors of the symmetrized (reversible) generalized
    eigenproblem C01 v = lambda C00 v, eigenvalues in decreasing order."""
    torch = _torch()
    x, y = x.double(), y.double()
    c00, c0t, ctt = _covariances(x, y)
    c00 = 0.5 * (c00 + ctt)
    c01 = 0.5 * (c0t + c0t.T)
    inv = _sym_inverse_sqrt(c00, epsilon, 'trunc')
    eigval, eigvec = torch.linalg.eigh(inv @ c01 @ inv)
    order = torch.argsort(eigval, descending=True)
    return eigval[order], inv @ eigvec[:, order]


def _seed(random_state):
    """An int seed for torch from None, an int or a numpy RandomState."""
    if random_state is None:
        return None
    if isinstance(random_state, (int, np.integer)):
        return int(random_state)
    if isinstance(random_state, np.random.RandomState):
        return int(random_state.randint(2**31 - 1))
    raise ValueError('random_state must be None, an int or a numpy RandomState')


# --- mlcolvar-compatible statistics (DeepTICA / DeepLDA) ---------------------

_EIG_LOSSES = ('sum', 'sum2', 'gap', 'its', 'single', 'single2')


def _cholesky_eigh(a, b, reg=1e-6, n_eig=None):
    """Generalized symmetric eigenproblem a v = lambda b v via the Cholesky
    factor of ``b + reg * I``, as mlcolvar's ``cholesky_eigh``: eigenvalues in
    decreasing order; eigenvectors (columns) scaled to unit Euclidean norm with
    a positive first component."""
    torch = _torch()
    if reg is not None:
        b = b + reg * torch.eye(b.shape[0], dtype=b.dtype, device=b.device)
    L = torch.linalg.cholesky(b, upper=False)
    L_ti = torch.inverse(L.T)
    L_i = torch.inverse(L)
    eigval, eigvec = torch.linalg.eigh(L_i @ a @ L_ti, UPLO='L')
    eigval, order = torch.sort(eigval, 0, descending=True)
    eigvec = torch.nn.functional.normalize(L_ti @ eigvec[:, order], dim=0)
    eigvec = eigvec * torch.sign(eigvec[0, :]).unsqueeze(0)
    if n_eig is not None:
        eigval, eigvec = eigval[:n_eig], eigvec[:, :n_eig]
    return eigval, eigvec


def _tica_eigen(x, y, w_x=None, w_y=None, reg=1e-6, n_eig=None):
    """mlcolvar's (reweighted) TICA of time-lagged pairs: the mean of x(t)
    is removed from both, C(0) = <x x^T>_w(t), C(lag) = sym(<x y^T>_w(t+lag)).
    Returns eigenvalues, eigenvectors and the mean."""
    torch = _torch()
    if w_x is None:
        w_x = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
    if w_y is None:
        w_y = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
    mean = torch.einsum('ij,i->j', x, w_x) / torch.sum(w_x)
    x, y = x - mean, y - mean
    c0 = torch.einsum('ij,ik,i->jk', x, x, w_x) / torch.sum(w_x)
    ct = torch.einsum('ij,ik,i->jk', x, y, w_y) / torch.sum(w_y)
    c0, ct = 0.5 * (c0 + c0.T), 0.5 * (ct + ct.T)
    eigval, eigvec = _cholesky_eigh(ct, c0, reg, n_eig)
    return eigval, eigvec, mean


def _lda_eigen(h, labels, sw_reg=1e-6, mode='standard', n_eig=None):
    """mlcolvar's LDA: between- and within-class scatter of ``h`` and their
    generalized eigenproblem (``mode='harmonic'`` averages the inverse
    within-class scatters, as harmonic LDA)."""
    torch = _torch()
    n, d = h.shape
    states = torch.unique(labels)
    h_bar = h - torch.mean(h, 0, True)
    s_t = h_bar.T @ h_bar / (n - 1)
    s_w = torch.zeros((d, d), dtype=h.dtype, device=h.device)
    s_w_inv = torch.zeros((d, d), dtype=h.dtype, device=h.device)
    for state in states:
        h_i = h[labels == state]
        h_i_bar = h_i - torch.mean(h_i, 0, True)
        cov_i = h_i_bar.T @ h_i_bar / ((h_i.shape[0] - 1) * len(states))
        s_w = s_w + cov_i
        if mode == 'harmonic':
            s_w_inv = s_w_inv + torch.inverse(cov_i)
    if mode == 'harmonic':
        s_w = torch.inverse(s_w_inv)
    return _cholesky_eigh(s_t - s_w, s_w, sw_reg, n_eig)


def _reduce_eigenvalues(eigval, mode='sum', n_eig=0):
    """The negative of a monotonic function of the eigenvalues, as mlcolvar's
    ``reduce_eigenvalues_loss``: 'sum', 'sum2' (sum of squares), 'gap'
    (lambda_1 - lambda_2), 'its' (sum of 1/log lambda), or 'single' /
    'single2' (eigenvalue number ``n_eig``, or its square)."""
    torch = _torch()
    if mode not in _EIG_LOSSES:
        raise ValueError('loss mode must be one of %s, got %r' % (', '.join(_EIG_LOSSES), mode))
    if n_eig > 0 and len(eigval) < n_eig:
        raise ValueError('n_eig must be at most the number of eigenvalues.')
    if n_eig == 0:
        if mode in ('single', 'single2'):
            raise ValueError("n_eig must be given for 'single' and 'single2'.")
        n_eig = len(eigval)
    if mode == 'sum':
        out = torch.sum(eigval[:n_eig])
    elif mode == 'sum2':
        out = torch.sum(eigval[:n_eig] ** 2)
    elif mode == 'gap':
        out = eigval[0] - eigval[1]
    elif mode == 'its':
        out = torch.sum(1 / torch.log(eigval[:n_eig]))
    elif mode == 'single':
        out = eigval[n_eig - 1]
    else:
        out = eigval[n_eig - 1] ** 2
    return -out


def _rescaled_time(logweights, dt=1.0):
    """Accelerated (rescaled) time of a biased trajectory, as mlcolvar's
    ``tprime_evaluation``: dt' = dt * exp(l / logsumexp(l)) with
    l = logweights - max(logweights)."""
    lw = np.asarray(logweights, dtype=np.float64)
    lw = lw - lw.max()
    m = lw.max()
    lognorm = m + np.log(np.sum(np.exp(lw - m)))
    return np.cumsum(np.exp(lw / lognorm) * dt)


@njit(cache=True)
def _rescaled_pairs_kernel(t, lag_time, idx_end, fill, i_out, j_out, w_t, w_lag):
    # mlcolvar's find_timelagged_configurations: every j whose interval
    # [t_j, t_j+1) overlaps [t_i + lag, t_i+1 + lag) pairs with i; the lagged
    # weight is the overlap, and the interval length of i is shared equally
    n = t.shape[0]
    k = 0
    for i in range(idx_end):
        stop = lag_time + t[i + 1]
        n_j = 0
        for j in range(i, n):
            if t[j] < stop and t[j + 1] > t[i] + lag_time:
                if fill:
                    i_out[k] = i
                    j_out[k] = j
                    w_lag[k] = (min(t[i + 1] + lag_time, t[j + 1])
                                - max(t[i] + lag_time, t[j]))
                k += 1
                n_j += 1
            elif t[j] > stop:
                break
        if fill:
            for m in range(k - n_j, k):
                w_t[m] = (t[i + 1] - t[i]) / n_j
    return k


def _rescaled_time_pairs(t, lag_time):
    """Index pairs (i, j) and weights (w_t, w_lag) of configurations a time
    ``lag_time`` apart in the (rescaled) time ``t``, as mlcolvar's
    ``find_timelagged_configurations``."""
    t = np.ascontiguousarray(t, dtype=np.float64)
    pos = int(np.searchsorted(t, t[-1] - lag_time, side='left'))
    idx_end = 0 if pos == 0 else (-1 if pos == len(t) else pos - 1)
    idx_end = max(idx_end, 0)
    empty_i, empty_f = np.empty(0, np.int64), np.empty(0)
    n = _rescaled_pairs_kernel(t, float(lag_time), idx_end, False,
                               empty_i, empty_i, empty_f, empty_f)
    i_out, j_out = np.empty(n, np.int64), np.empty(n, np.int64)
    w_t, w_lag = np.empty(n), np.empty(n)
    _rescaled_pairs_kernel(t, float(lag_time), idx_end, True, i_out, j_out, w_t, w_lag)
    return i_out, j_out, w_t, w_lag


# --- shared machinery ---------------------------------------------------------

def _lagged_pairs(sequences, lag_time):
    x0, xt = [], []
    for X in sequences:
        X = np.asarray(array2d(X), dtype=np.float64)
        if len(X) > lag_time:
            x0.append(X[:-lag_time])
            xt.append(X[lag_time:])
    if not x0:
        raise ValueError('No sequence is longer than lag_time=%d.' % lag_time)
    return np.concatenate(x0), np.concatenate(xt)


def _mlp(sizes, activation, out_activation=None):
    torch = _torch()
    act = {'elu': torch.nn.ELU, 'relu': torch.nn.ReLU, 'tanh': torch.nn.Tanh,
           'sigmoid': torch.nn.Sigmoid, 'softplus': torch.nn.Softplus,
           'gelu': torch.nn.GELU, 'silu': torch.nn.SiLU}[activation]
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if out_activation is not None:
        layers.append(out_activation)
    # float32 whatever torch's default dtype is: training feeds float32 data
    return torch.nn.Sequential(*layers).float()


class _DeepEstimator(BaseEstimator, TransformerMixin):
    """Shared training loop for the neural-network estimators."""

    def _check_common(self):
        if int(self.lag_time) < 1:
            raise ValueError('lag_time must be at least 1, got %r' % self.lag_time)
        if int(self.n_components) < 1:
            raise ValueError('n_components must be at least 1, got %r' % self.n_components)
        if self.activation not in _ACTIVATIONS:
            raise ValueError('activation must be one of %s, got %r'
                             % (', '.join(_ACTIVATIONS), self.activation))

    def _standardize_fit(self, sequences):
        allX = np.concatenate([np.asarray(array2d(X), dtype=np.float64) for X in sequences])
        self.feature_means_ = allX.mean(axis=0)
        std = allX.std(axis=0)
        self.feature_scales_ = np.where(std > 0, std, 1.0)

    def _standardize(self, X):
        return (np.asarray(array2d(X), dtype=np.float64) - self.feature_means_) / self.feature_scales_

    def _train(self, modules, loss_fn, *arrays):
        """Mini-batch Adam over the rows of ``arrays`` (e.g. x(t), x(t + lag));
        ``loss_fn(*batch)`` is minimized. Returns the mean loss of each epoch."""
        torch = _torch()
        seed = _seed(self.random_state)
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        device = torch.device(self.device)
        params = [p for m in modules for p in m.parameters()]
        for m in modules:
            m.to(device)
        opt = torch.optim.Adam(params, lr=float(self.learning_rate))
        tensors = [torch.as_tensor(a, dtype=torch.float32, device=device) for a in arrays]
        n = tensors[0].shape[0]
        batch = min(int(self.batch_size), n)
        if batch < 2:
            raise ValueError('Need at least 2 training samples.')
        history = []
        for epoch in range(int(self.n_epochs)):
            perm = torch.randperm(n, generator=gen).to(device)
            losses = []
            for start in range(0, n - batch + 1, batch):  # drop the ragged last batch
                idx = perm[start:start + batch]
                opt.zero_grad()
                loss = loss_fn(*[t[idx] for t in tensors])
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            history.append(float(np.mean(losses)))
            if self.verbose:
                print('epoch %d/%d  loss %.6f' % (epoch + 1, self.n_epochs, history[-1]))
        return history

    def _net_output(self, Z):
        """Network output (float64 numpy) for already standardized rows."""
        torch = _torch()
        net = self._embedding_net()
        net.eval()
        with torch.no_grad():
            t = torch.as_tensor(Z, dtype=torch.float32, device=next(net.parameters()).device)
            return net(t).cpu().numpy().astype(np.float64)

    def _embed(self, X):
        """Network output for one raw sequence (numpy)."""
        return self._net_output(self._standardize(X))

    def _post_affine(self):
        """(shift, matrix) applied to the network output as (Y - shift) @ matrix,
        or None for no output transformation."""
        return None

    def _postprocess(self, Y):
        affine = self._post_affine()
        if affine is None:
            return Y
        shift, matrix = affine
        return (Y - shift) @ matrix

    def to_torchscript(self, path=None):
        """Export the fitted model as a TorchScript module.

        The module maps raw (unstandardized) features to the output of
        ``transform``, in float32 on the CPU: input standardization, the
        network and any output rotation are all included. It can be loaded
        with ``torch.jit.load`` or by PLUMED's ``PYTORCH_MODEL`` action to
        use the learned collective variables in enhanced-sampling
        simulations (the PLUMED input must compute the same features, in the
        same order, as were used for training).

        Parameters
        ----------
        path : str, optional
            If given, the module is also saved to this file.

        Returns
        -------
        module : torch.jit.ScriptModule
        """
        torch = _torch()
        self._check_fitted()
        affine = self._post_affine()
        f32 = dict(dtype=torch.float32)

        class _Exported(torch.nn.Module):
            def __init__(self, net, mean, scale, shift, matrix):
                super().__init__()
                self.net = net
                self.register_buffer('mean', torch.as_tensor(mean, **f32))
                self.register_buffer('scale', torch.as_tensor(scale, **f32))
                self.register_buffer('shift', torch.as_tensor(shift, **f32))
                self.register_buffer('matrix', torch.as_tensor(matrix, **f32))

            def forward(self, x):
                y = self.net((x - self.mean) / self.scale)
                return (y - self.shift) @ self.matrix

        net = copy.deepcopy(self._embedding_net()).cpu().float().eval()
        n_out = self._net_output(self.feature_means_[None, :]).shape[1]
        shift, matrix = affine if affine is not None else (np.zeros(n_out), np.eye(n_out))
        module = _Exported(net, self.feature_means_, self.feature_scales_, shift, matrix).eval()
        example = torch.zeros(1, len(self.feature_means_), **f32)
        with torch.no_grad():
            traced = torch.jit.trace(module, example)
        if path is not None:
            torch.jit.save(traced, path)
        return traced

    def _check_fitted(self):
        if not hasattr(self, 'loss_history_'):
            raise ValueError('This %s instance is not fitted yet. Call fit() first.'
                             % type(self).__name__)

    def partial_transform(self, sequence):
        """Embed one sequence.

        Parameters
        ----------
        sequence : array-like, shape (n_samples, n_features)

        Returns
        -------
        embedded : array, shape (n_samples, n_components)
        """
        self._check_fitted()
        return self._postprocess(self._embed(sequence))

    def transform(self, sequences):
        """Embed each sequence; returns a list of arrays (n_samples_i, n_components)."""
        check_iter_of_sequences(sequences, max_iter=3)
        return [self.partial_transform(X) for X in sequences]

    def fit_transform(self, sequences, y=None):
        return self.fit(sequences).transform(sequences)

    def _fit_vamp(self, sequences):
        lag = max(1, int(round(float(self.lag_time))))
        self.vamp_ = VAMP(lag_time=lag, score_method='VAMP2').fit(
            self.transform(sequences))

    def score(self, sequences, y=None):
        """VAMP-2 score of the learned embedding on (new) data.

        A linear :class:`~msmbuilder.decomposition.VAMP` model is fitted to the
        embedding of the training data during ``fit``; this returns its VAMP-2
        score on the embedding of ``sequences``. Higher is better; use
        held-out data for cross-validation.
        """
        self._check_fitted()
        return self.vamp_.score(self.transform(sequences))


_COMMON_DOC = """
    lag_time : int, default=1
        Lag time, in frames, between x(t) and x(t + lag_time).
    hidden_layers : tuple of int, default=(64, 64)
        Sizes of the hidden layers.
    activation : str, default='elu'
        Hidden-layer nonlinearity: one of 'elu', 'relu', 'tanh', 'sigmoid',
        'softplus', 'gelu', 'silu'.
    learning_rate : float, default=1e-3
        Adam learning rate.
    n_epochs : int, default=30
        Passes over the training pairs.
    batch_size : int, default=1024
        Time-lagged pairs per gradient step.
    random_state : int or None, default=None
        Seed for weight initialization and batch order. Set it for
        reproducible fits.
    device : str, default='cpu'
        PyTorch device, e.g. 'cpu', 'cuda' or 'mps'.
    verbose : bool, default=False
        Print the loss after each epoch."""


class VAMPNet(_DeepEstimator):
    __doc__ = """VAMPnet: a neural network trained to maximize a VAMP score.

    The same network (one "lobe") maps x(t) and x(t + lag_time) to
    ``n_components`` outputs, and is trained to maximize the VAMP score of the
    output pairs. With ``output='softmax'`` (as in [1]_) the outputs are
    membership probabilities of soft metastable states; with ``'linear'``
    they are unconstrained slow collective variables.

    Parameters
    ----------
    n_components : int, default=4
        Number of network outputs (states, for softmax).
    output : {'softmax', 'linear'}, default='softmax'
        Output layer.
    score_method : {'VAMP1', 'VAMP2', 'VAMPE'}, default='VAMP2'
        Training objective.
    epsilon : float, default=1e-6
        Threshold for small covariance eigenvalues in the loss.""" + _COMMON_DOC + """

    Attributes
    ----------
    loss_history_ : list of float
        Mean training loss (the negative VAMP score) of each epoch.
    vamp_ : VAMP
        Linear VAMP model fitted to the training-set outputs; used by
        ``score`` and providing ``singular_values_`` and ``timescales_``.
    singular_values_, timescales_ : array
        Shortcuts to ``vamp_.singular_values_`` and ``vamp_.timescales_``.
    feature_means_, feature_scales_ : array
        Input standardization learned from the training data.

    References
    ----------
    .. [1] Mardt, A., Pasquali, L., Wu, H. and Noe, F. Nat. Commun. 9, 5 (2018).
    """

    def __init__(self, n_components=4, lag_time=1, output='softmax', score_method='VAMP2',
                 epsilon=1e-6, hidden_layers=(64, 64), activation='elu', learning_rate=1e-3,
                 n_epochs=30, batch_size=1024, random_state=None, device='cpu', verbose=False):
        self.n_components = n_components
        self.lag_time = lag_time
        self.output = output
        self.score_method = score_method
        self.epsilon = epsilon
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _embedding_net(self):
        return self.lobe_

    def fit(self, sequences, y=None):
        """Train the VAMPnet on a list of feature sequences."""
        torch = _torch()
        self._check_common()
        if self.output not in ('softmax', 'linear'):
            raise ValueError("output must be 'softmax' or 'linear', got %r" % self.output)
        check_iter_of_sequences(sequences)
        self._standardize_fit(sequences)
        x0, xt = _lagged_pairs([self._standardize(X) for X in sequences], int(self.lag_time))
        sizes = [x0.shape[1], *map(int, self.hidden_layers), int(self.n_components)]
        out = torch.nn.Softmax(dim=1) if self.output == 'softmax' else None
        if _seed(self.random_state) is not None:
            torch.manual_seed(_seed(self.random_state))  # weight initialization
        self.lobe_ = _mlp(sizes, self.activation, out)

        def loss(a, b):
            return -vamp_score_torch(self.lobe_(a), self.lobe_(b), method=self.score_method,
                                     epsilon=self.epsilon)
        self.loss_history_ = self._train([self.lobe_], loss, x0, xt)
        self._fit_vamp(sequences)
        return self

    @property
    def singular_values_(self):
        self._check_fitted()
        return self.vamp_.singular_values_

    @property
    def timescales_(self):
        self._check_fitted()
        return self.vamp_.timescales_


class SRV(_DeepEstimator):
    __doc__ = """State-free reversible VAMPnet (SRV).

    A network maps x to ``n_components`` outputs and is trained to maximize
    the reversible VAMP-2 score (sum of squared eigenvalues of the
    symmetrized time-lagged eigenproblem). After training, the outputs are
    rotated into the eigenfunctions of that problem, ordered from slowest to
    fastest, so ``transform`` returns slow collective variables with implied
    timescales, much like a nonlinear tICA [2]_.

    Parameters
    ----------
    n_components : int, default=2
        Number of slow eigenfunctions to learn.
    epsilon : float, default=1e-6
        Threshold for small covariance eigenvalues in the loss.""" + _COMMON_DOC + """

    Attributes
    ----------
    eigenvalues_ : array, shape (n_components,)
        Eigenvalues of the learned eigenfunctions on the training data, in
        decreasing order.
    timescales_ : array, shape (n_components,)
        Implied timescales ``-lag_time / log(eigenvalues_)``.
    loss_history_ : list of float
        Mean training loss (negative reversible VAMP-2) of each epoch.
    vamp_ : VAMP
        Linear VAMP model on the training-set eigenfunctions, used by ``score``.
    feature_means_, feature_scales_ : array
        Input standardization learned from the training data.

    References
    ----------
    .. [2] Chen, W., Sidky, H. and Ferguson, A. L. J. Chem. Phys. 150, 214114 (2019).
    """

    def __init__(self, n_components=2, lag_time=1, epsilon=1e-6, hidden_layers=(64, 64),
                 activation='elu', learning_rate=1e-3, n_epochs=30, batch_size=1024,
                 random_state=None, device='cpu', verbose=False):
        self.n_components = n_components
        self.lag_time = lag_time
        self.epsilon = epsilon
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _embedding_net(self):
        return self.net_

    def fit(self, sequences, y=None):
        """Train the SRV on a list of feature sequences."""
        torch = _torch()
        self._check_common()
        check_iter_of_sequences(sequences)
        self._standardize_fit(sequences)
        x0, xt = _lagged_pairs([self._standardize(X) for X in sequences], int(self.lag_time))
        sizes = [x0.shape[1], *map(int, self.hidden_layers), int(self.n_components)]
        if _seed(self.random_state) is not None:
            torch.manual_seed(_seed(self.random_state))  # weight initialization
        self.net_ = _mlp(sizes, self.activation)

        def loss(a, b):
            eigval, _ = _reversible_eigen(self.net_(a), self.net_(b), self.epsilon)
            return -(1 + torch.sum(eigval ** 2)).float()
        self.loss_history_ = self._train([self.net_], loss, x0, xt)

        # rotate the outputs into ordered eigenfunctions on all training data
        y0 = np.concatenate([self._embed(X)[:-self.lag_time] for X in sequences
                             if len(X) > self.lag_time])
        yt = np.concatenate([self._embed(X)[self.lag_time:] for X in sequences
                             if len(X) > self.lag_time])
        eigval, eigvec = _reversible_eigen(torch.as_tensor(y0), torch.as_tensor(yt), self.epsilon)
        self.eigenvalues_ = eigval.numpy()
        self._rotation = eigvec.numpy()
        self._output_mean = 0.5 * (y0.mean(axis=0) + yt.mean(axis=0))
        self._fit_vamp(sequences)
        return self

    def _post_affine(self):
        return self._output_mean, self._rotation

    @property
    def timescales_(self):
        self._check_fitted()
        with np.errstate(divide='ignore', invalid='ignore'):
            return -self.lag_time / np.log(self.eigenvalues_)


class TimeLaggedAutoencoder(_DeepEstimator):
    __doc__ = """Time-lagged autoencoder (TAE).

    An encoder maps standardized x(t) to ``n_components`` latent variables and
    a decoder predicts standardized x(t + lag_time) from them; both are
    trained to minimize the mean squared prediction error [3]_. The latent
    variables (``transform``) capture the features' slow, predictable
    variation. With linear layers this recovers the tICA subspace.

    Parameters
    ----------
    n_components : int, default=2
        Size of the latent bottleneck.""" + _COMMON_DOC + """
        The decoder mirrors ``hidden_layers``.

    Attributes
    ----------
    loss_history_ : list of float
        Mean training loss (mean squared error) of each epoch.
    vamp_ : VAMP
        Linear VAMP model on the training-set latent variables, used by
        ``score``.
    feature_means_, feature_scales_ : array
        Input standardization learned from the training data.

    References
    ----------
    .. [3] Wehmeyer, C. and Noe, F. J. Chem. Phys. 148, 241703 (2018).
    """

    def __init__(self, n_components=2, lag_time=1, hidden_layers=(64, 64), activation='elu',
                 learning_rate=1e-3, n_epochs=30, batch_size=1024, random_state=None,
                 device='cpu', verbose=False):
        self.n_components = n_components
        self.lag_time = lag_time
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _embedding_net(self):
        return self.encoder_

    def fit(self, sequences, y=None):
        """Train the autoencoder on a list of feature sequences."""
        torch = _torch()
        self._check_common()
        check_iter_of_sequences(sequences)
        self._standardize_fit(sequences)
        x0, xt = _lagged_pairs([self._standardize(X) for X in sequences], int(self.lag_time))
        hidden = [int(h) for h in self.hidden_layers]
        if _seed(self.random_state) is not None:
            torch.manual_seed(_seed(self.random_state))  # weight initialization
        self.encoder_ = _mlp([x0.shape[1], *hidden, int(self.n_components)], self.activation)
        self.decoder_ = _mlp([int(self.n_components), *hidden[::-1], x0.shape[1]], self.activation)
        mse = torch.nn.MSELoss()

        def loss(a, b):
            return mse(self.decoder_(self.encoder_(a)), b)
        self.loss_history_ = self._train([self.encoder_, self.decoder_], loss, x0, xt)
        self._fit_vamp(sequences)
        return self

    def reconstruction_error(self, sequences):
        """Mean squared error of predicting standardized x(t + lag_time) from
        x(t) on ``sequences`` (lower is better)."""
        torch = _torch()
        self._check_fitted()
        x0, xt = _lagged_pairs([self._standardize(X) for X in sequences], int(self.lag_time))
        device = next(self.encoder_.parameters()).device
        with torch.no_grad():
            pred = self.decoder_(self.encoder_(torch.as_tensor(x0, dtype=torch.float32,
                                                               device=device)))
            return float(torch.mean((pred - torch.as_tensor(xt, dtype=torch.float32,
                                                             device=device)) ** 2))


def _as_sequence_list(values, sequences, name):
    if values is None:
        return None
    values = [np.asarray(v, dtype=np.float64).reshape(-1) for v in values]
    if len(values) != len(sequences) or any(len(v) != len(X) for v, X in zip(values, sequences)):
        raise ValueError('%s must have one value per frame of each sequence' % name)
    return values


class DeepTICA(_DeepEstimator):
    __doc__ = """Deep-TICA: a network trained to maximize TICA eigenvalues.

    A network maps standardized features to ``nn_outputs`` values; TICA of
    those outputs gives eigenvalues, and the network is trained to maximize
    (by default) their sum of squares [4]_. ``transform`` returns the
    ``n_components`` slowest TICA projections of the network output. The
    model is the one of mlcolvar's ``DeepTICA`` [6]_ and has the same
    definitions: mean-free outputs, eigenvectors of unit length (so the
    components are not scaled to unit variance) with a positive first entry.

    Biased simulations (e.g. metadynamics or OPES) can be used by passing
    ``logweights`` (typically ``beta * bias`` for each frame) to ``fit``.
    With ``reweighting='rescale_time'`` (as in [4]_) time-lagged pairs are
    found in the accelerated time ``t'`` (``dt' = dt * exp(...)``, see
    mlcolvar's ``tprime_evaluation``) and ``lag_time`` is measured in it;
    with ``'weights_t'`` frames ``lag_time`` apart are paired and weighted by
    ``exp(logweights)``.

    Parameters
    ----------
    n_components : int, default=2
        Number of slow components (collective variables) to return.
    lag_time : int or float, default=1
        Lag time in frames (or in rescaled time with
        ``reweighting='rescale_time'`` and ``logweights``).
    nn_outputs : int, optional
        Size of the network output layer that TICA is applied to; defaults
        to ``n_components``.
    loss_mode : str, default='sum2'
        Function of the TICA eigenvalues to maximize: 'sum2' (sum of
        squares), 'sum', 'gap', 'its', 'single' or 'single2' (the last two
        use the ``n_components``-th eigenvalue).
    reg_c0 : float, default=1e-6
        Added to the diagonal of C(0) before the Cholesky factorization.
    reweighting : {'rescale_time', 'weights_t'}, default='rescale_time'
        How ``logweights`` are used (ignored without them).\n""" + _COMMON_DOC[
        _COMMON_DOC.index('    hidden_layers'):].replace(
        "activation : str, default='elu'", "activation : str, default='tanh'") + """

    Attributes
    ----------
    eigenvalues_ : array, shape (n_components,)
        TICA eigenvalues of the network output on all training pairs.
    timescales_ : array, shape (n_components,)
        Implied timescales ``-lag_time / log|eigenvalues_|``.
    loss_history_ : list of float
        Mean training loss (the negative objective) of each epoch.
    vamp_ : VAMP
        Linear VAMP model on the (unweighted) training-set output, used by
        ``score``.
    feature_means_, feature_scales_ : array
        Input standardization learned from the training data.

    References
    ----------
    .. [4] Bonati, L., Piccini, G. and Parrinello, M. PNAS 118, e2113533118 (2021).
    .. [6] Bonati, L., Trizio, E., Rizzi, A. and Parrinello, M. J. Chem. Phys.
       159, 014801 (2023).
    """

    def __init__(self, n_components=2, lag_time=1, nn_outputs=None, loss_mode='sum2',
                 reg_c0=1e-6, reweighting='rescale_time', hidden_layers=(64, 64),
                 activation='tanh', learning_rate=1e-3, n_epochs=30, batch_size=1024,
                 random_state=None, device='cpu', verbose=False):
        self.n_components = n_components
        self.lag_time = lag_time
        self.nn_outputs = nn_outputs
        self.loss_mode = loss_mode
        self.reg_c0 = reg_c0
        self.reweighting = reweighting
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _embedding_net(self):
        return self.net_

    def _n_outputs(self):
        return int(self.n_components if self.nn_outputs is None else self.nn_outputs)

    def _pairs(self, Z, logweights):
        """Standardized time-lagged pairs and their weights."""
        if logweights is not None and self.reweighting == 'rescale_time':
            if float(self.lag_time) <= 0:
                raise ValueError('lag_time must be positive, got %r' % self.lag_time)
            x0, xt, w0, wt = [], [], [], []
            for X, lw in zip(Z, logweights):
                if len(X) < 2:
                    continue
                i, j, a, b = _rescaled_time_pairs(_rescaled_time(lw), float(self.lag_time))
                x0.append(X[i])
                xt.append(X[j])
                w0.append(a)
                wt.append(b)
            if not x0 or sum(len(a) for a in x0) == 0:
                raise ValueError('No time-lagged pairs at lag_time=%r in rescaled time.'
                                 % self.lag_time)
            return (np.concatenate(x0), np.concatenate(xt),
                    np.concatenate(w0), np.concatenate(wt))
        lag = int(self.lag_time)
        if lag != self.lag_time or lag < 1:
            raise ValueError('lag_time must be a positive integer number of frames, got %r'
                             % self.lag_time)
        x0, xt = _lagged_pairs(Z, lag)
        if logweights is None:
            return x0, xt, np.ones(len(x0)), np.ones(len(x0))
        # weights_t; a common shift of the log-weights cancels in the TICA
        shift = max(float(np.max(lw)) for lw in logweights)
        w0 = np.concatenate([np.exp(lw[:-lag] - shift) for lw in logweights if len(lw) > lag])
        wt = np.concatenate([np.exp(lw[lag:] - shift) for lw in logweights if len(lw) > lag])
        return x0, xt, w0, wt

    def fit(self, sequences, y=None, logweights=None):
        """Train Deep-TICA on a list of feature sequences.

        Parameters
        ----------
        sequences : list of array-like, each of shape (n_frames_i, n_features)
        y : None
            Ignored.
        logweights : list of array-like, optional
            Log-weights of each frame (e.g. ``beta * bias``) of each sequence,
            for biased simulations; see ``reweighting``.
        """
        torch = _torch()
        check_iter_of_sequences(sequences)
        if int(self.n_components) < 1:
            raise ValueError('n_components must be at least 1, got %r' % self.n_components)
        if self._n_outputs() < int(self.n_components):
            raise ValueError('nn_outputs (%r) must be at least n_components (%r)'
                             % (self.nn_outputs, self.n_components))
        if self.activation not in _ACTIVATIONS:
            raise ValueError('activation must be one of %s, got %r'
                             % (', '.join(_ACTIVATIONS), self.activation))
        if self.loss_mode not in _EIG_LOSSES:
            raise ValueError('loss_mode must be one of %s, got %r'
                             % (', '.join(_EIG_LOSSES), self.loss_mode))
        if self.reweighting not in ('rescale_time', 'weights_t'):
            raise ValueError("reweighting must be 'rescale_time' or 'weights_t', got %r"
                             % self.reweighting)
        logweights = _as_sequence_list(logweights, sequences, 'logweights')
        self._standardize_fit(sequences)
        Z = [self._standardize(X) for X in sequences]
        x0, xt, w0, wt = self._pairs(Z, logweights)

        n_cv = int(self.n_components)
        sizes = [x0.shape[1], *map(int, self.hidden_layers), self._n_outputs()]
        if _seed(self.random_state) is not None:
            torch.manual_seed(_seed(self.random_state))  # weight initialization
        self.net_ = _mlp(sizes, self.activation)
        n_eig = n_cv if self.loss_mode in ('single', 'single2') else 0

        def loss(a, b, wa, wb):
            eigval, _, _ = _tica_eigen(self.net_(a).double(), self.net_(b).double(),
                                       wa.double(), wb.double(), float(self.reg_c0), n_cv)
            return _reduce_eigenvalues(eigval, self.loss_mode, n_eig).float()
        self.loss_history_ = self._train([self.net_], loss, x0, xt, w0, wt)

        # TICA of the trained network output on all training pairs
        eigval, eigvec, mean = _tica_eigen(
            torch.as_tensor(self._net_output(x0)), torch.as_tensor(self._net_output(xt)),
            torch.as_tensor(w0), torch.as_tensor(wt), float(self.reg_c0), n_cv)
        self.eigenvalues_ = eigval.numpy()
        self._evecs = eigvec.numpy()
        self._tica_mean = mean.numpy()
        self._fit_vamp(sequences)
        return self

    def fit_transform(self, sequences, y=None, logweights=None):
        return self.fit(sequences, logweights=logweights).transform(sequences)

    def _post_affine(self):
        return self._tica_mean, self._evecs

    @property
    def timescales_(self):
        self._check_fitted()
        with np.errstate(divide='ignore', invalid='ignore'):
            return -float(self.lag_time) / np.log(np.abs(self.eigenvalues_))


class DeepLDA(_DeepEstimator):
    __doc__ = """Deep-LDA: a supervised network that separates labelled states.

    A network maps standardized features to ``nn_outputs`` values, and is
    trained to maximize the LDA eigenvalues (the ratio of between-state to
    within-state scatter) of those outputs for the labelled training frames
    [5]_. ``transform`` returns the ``n_classes - 1`` LDA projections of the
    network output: collective variables that discriminate the states, e.g.
    from short unbiased runs in each metastable state, for enhanced
    sampling. The model is the one of mlcolvar's ``DeepLDA`` [6]_, with its
    regularizations: ``sw_reg`` added to the within-state scatter, and a
    Lorentzian penalty ``-lorentzian_reg / (1 + (<|s|^2> - 1)^2)`` keeping
    the projections ``s`` of order one.

    Parameters
    ----------
    nn_outputs : int, optional
        Size of the network output layer that LDA is applied to; defaults to
        ``n_classes - 1``.
    sw_reg : float, default=0.05
        Regularization of the within-state scatter matrix.
    lorentzian_reg : float, optional
        Strength of the Lorentzian penalty; defaults to ``2 / sw_reg``. Set
        it to 0 to turn the penalty off.
    lda_mode : {'standard', 'harmonic'}, default='standard'
        'harmonic' uses the harmonic mean of the within-state scatters
        (HLDA).
    hidden_layers : tuple of int, default=(64, 64)
        Sizes of the hidden layers.
    activation : str, default='tanh'
        Hidden-layer nonlinearity: one of 'elu', 'relu', 'tanh', 'sigmoid',
        'softplus', 'gelu', 'silu'.
    learning_rate : float, default=1e-3
        Adam learning rate.
    n_epochs : int, default=30
        Passes over the training frames.
    batch_size : int, default=1024
        Frames per gradient step. Each batch should contain every state.
    random_state : int or None, default=None
        Seed for weight initialization and batch order.
    device : str, default='cpu'
        PyTorch device.
    verbose : bool, default=False
        Print the loss after each epoch.

    Attributes
    ----------
    classes_ : array
        The distinct labels, in sorted order.
    n_components_ : int
        Number of output components, ``n_classes - 1``.
    eigenvalues_ : array, shape (n_components_,)
        LDA eigenvalues of the network output on the training data.
    loss_history_ : list of float
        Mean training loss of each epoch.
    feature_means_, feature_scales_ : array
        Input standardization learned from the training data.

    References
    ----------
    .. [5] Bonati, L., Rizzi, V. and Parrinello, M. J. Phys. Chem. Lett. 11,
       2998-3004 (2020).
    .. [6] Bonati, L., Trizio, E., Rizzi, A. and Parrinello, M. J. Chem. Phys.
       159, 014801 (2023).
    """

    def __init__(self, nn_outputs=None, sw_reg=0.05, lorentzian_reg=None, lda_mode='standard',
                 hidden_layers=(64, 64), activation='tanh', learning_rate=1e-3, n_epochs=30,
                 batch_size=1024, random_state=None, device='cpu', verbose=False):
        self.nn_outputs = nn_outputs
        self.sw_reg = sw_reg
        self.lorentzian_reg = lorentzian_reg
        self.lda_mode = lda_mode
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _embedding_net(self):
        return self.net_

    def _labels(self, sequences, y):
        """Per-frame label indices for a list of sequences. ``y`` has one
        entry per sequence: a single label for all its frames, or one label
        per frame."""
        if y is None:
            raise ValueError('DeepLDA is supervised: pass labels y to fit and score.')
        if len(y) != len(sequences):
            raise ValueError('y must have one entry per sequence (got %d for %d sequences)'
                             % (len(y), len(sequences)))
        out = []
        for X, lab in zip(sequences, y):
            lab = np.asarray(lab)
            n = len(X)
            if lab.ndim == 0:
                lab = np.repeat(lab, n)
            elif len(lab) != n:
                raise ValueError('a label array must have one label per frame')
            out.append(lab)
        return np.concatenate(out)

    def _lorentzian(self):
        if self.lorentzian_reg is None:
            return 2.0 / float(self.sw_reg)
        return float(self.lorentzian_reg)

    def fit(self, sequences, y=None):
        """Train Deep-LDA on labelled feature sequences.

        Parameters
        ----------
        sequences : list of array-like, each of shape (n_frames_i, n_features)
        y : list
            One entry per sequence: the state label of all its frames, or an
            array with the label of each frame.
        """
        torch = _torch()
        check_iter_of_sequences(sequences)
        labels = self._labels(sequences, y)
        self.classes_, idx = np.unique(labels, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError('DeepLDA needs at least 2 classes, got %d' % len(self.classes_))
        counts = np.bincount(idx)
        if counts.min() < 2:
            raise ValueError('every class needs at least 2 frames')
        if self.activation not in _ACTIVATIONS:
            raise ValueError('activation must be one of %s, got %r'
                             % (', '.join(_ACTIVATIONS), self.activation))
        if self.lda_mode not in ('standard', 'harmonic'):
            raise ValueError("lda_mode must be 'standard' or 'harmonic', got %r" % self.lda_mode)
        self.n_components_ = len(self.classes_) - 1
        n_out = self.n_components_ if self.nn_outputs is None else int(self.nn_outputs)
        if n_out < self.n_components_:
            raise ValueError('nn_outputs (%r) must be at least n_classes - 1 (%d)'
                             % (self.nn_outputs, self.n_components_))

        self._standardize_fit(sequences)
        x = np.concatenate([self._standardize(X) for X in sequences])
        sizes = [x.shape[1], *map(int, self.hidden_layers), n_out]
        if _seed(self.random_state) is not None:
            torch.manual_seed(_seed(self.random_state))  # weight initialization
        self.net_ = _mlp(sizes, self.activation)
        k, sw_reg, alpha, mode = self.n_components_, float(self.sw_reg), self._lorentzian(), \
            self.lda_mode

        def loss(xb, lb):
            h = self.net_(xb).double()
            eigval, eigvec = _lda_eigen(h, lb, sw_reg, mode, k)
            out = _reduce_eigenvalues(eigval, 'sum')
            if alpha > 0:
                out = out + _lorentzian_penalty(h @ eigvec, alpha)
            return out.float()
        self.loss_history_ = self._train([self.net_], loss, x, idx.astype(np.float64))

        eigval, eigvec = _lda_eigen(torch.as_tensor(self._net_output(x)), torch.as_tensor(idx),
                                    sw_reg, mode, k)
        self.eigenvalues_ = eigval.numpy()
        self._evecs = eigvec.numpy()
        return self

    def fit_transform(self, sequences, y=None):
        return self.fit(sequences, y).transform(sequences)

    def _post_affine(self):
        return np.zeros(self._evecs.shape[0]), self._evecs

    def score(self, sequences, y=None):
        """Class separation of the learned collective variables on labelled
        data: the sum of the LDA eigenvalues (between- over within-state
        scatter) of the ``transform`` output. Higher is better."""
        torch = _torch()
        self._check_fitted()
        labels = self._labels(sequences, y)
        known = np.isin(labels, self.classes_)
        if not known.all():
            raise ValueError('y contains labels not seen during fit')
        idx = np.searchsorted(self.classes_, labels)
        s = np.concatenate(self.transform(sequences))
        eigval, _ = _lda_eigen(torch.as_tensor(s), torch.as_tensor(idx), float(self.sw_reg),
                               self.lda_mode, self.n_components_)
        return float(eigval.sum())


def _lorentzian_penalty(s, alpha):
    """mlcolvar's Deep-LDA regularization of the projections s."""
    reg = s.pow(2).sum() / s.shape[0]
    return -alpha / (1 + (reg - 1) ** 2)
