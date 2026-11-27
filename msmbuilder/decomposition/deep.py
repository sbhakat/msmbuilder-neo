"""Neural-network estimators of slow collective variables.

* :class:`VAMPNet` -- VAMPnets [1]_: a network trained to maximize a VAMP
  score of its outputs. With softmax outputs it learns soft metastable states.
* :class:`SRV` -- state-free reversible VAMPnets [2]_: a network trained to
  maximize the reversible VAMP-2 score; its outputs are ordered
  eigenfunctions with implied timescales.
* :class:`TimeLaggedAutoencoder` -- time-lagged autoencoders [3]_: an
  encoder-decoder trained to predict x(t + lag) from x(t) through a
  low-dimensional bottleneck.

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
"""
import numpy as np
from sklearn.base import TransformerMixin

from ..base import BaseEstimator
from ..utils import array2d, check_iter_of_sequences
from .vamp import VAMP

__all__ = ['VAMPNet', 'SRV', 'TimeLaggedAutoencoder', 'vamp_score_torch']

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
    return torch.nn.Sequential(*layers)


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

    def _train(self, modules, loss_fn, x0, xt):
        """Mini-batch Adam on (x0, xt) pairs; loss_fn(batch_x0, batch_xt) is
        minimized. Returns the mean loss of each epoch."""
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
        X0 = torch.as_tensor(x0, dtype=torch.float32, device=device)
        XT = torch.as_tensor(xt, dtype=torch.float32, device=device)
        n = X0.shape[0]
        batch = min(int(self.batch_size), n)
        if batch < 2:
            raise ValueError('Need at least 2 time-lagged pairs to train.')
        history = []
        for epoch in range(int(self.n_epochs)):
            perm = torch.randperm(n, generator=gen).to(device)
            losses = []
            for start in range(0, n - batch + 1, batch):  # drop the ragged last batch
                idx = perm[start:start + batch]
                opt.zero_grad()
                loss = loss_fn(X0[idx], XT[idx])
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            history.append(float(np.mean(losses)))
            if self.verbose:
                print('epoch %d/%d  loss %.6f' % (epoch + 1, self.n_epochs, history[-1]))
        return history

    def _embed(self, X):
        """Network output for one standardized-or-not sequence (numpy)."""
        torch = _torch()
        net = self._embedding_net()
        net.eval()
        with torch.no_grad():
            t = torch.as_tensor(self._standardize(X), dtype=torch.float32,
                                device=next(net.parameters()).device)
            return net(t).cpu().numpy().astype(np.float64)

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

    def _postprocess(self, Y):
        return Y

    def transform(self, sequences):
        """Embed each sequence; returns a list of arrays (n_samples_i, n_components)."""
        check_iter_of_sequences(sequences, max_iter=3)
        return [self.partial_transform(X) for X in sequences]

    def fit_transform(self, sequences, y=None):
        return self.fit(sequences).transform(sequences)

    def _fit_vamp(self, sequences):
        self.vamp_ = VAMP(lag_time=self.lag_time, score_method='VAMP2').fit(
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

    def _postprocess(self, Y):
        return (Y - self._output_mean) @ self._rotation

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
