# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.
"""Metropolis MCMC sampler for reversible transition matrices.

Numba port of the former metzner_mcmc.c / _metzner_mcmc_fast.pyx. With
``n_chains=1`` it yields exactly the same samples as before (and as
``metzner_mcmc_slow``). With several chains, each chain now keeps its own
running sum of K; the old code shared one sum across all chains, so parallel
chains overwrote each other's value.
"""
import math

import numpy as np
from numba import njit, prange
from sklearn.utils import check_random_state

_K_MINUS = 0.9
_K_PLUS = 1.1


@njit(cache=True, nogil=True)
def _acceptance_ratio(i, j, epsilon, Z, K, N, Q):
    term1 = Z[i, j] * (math.log(K[i, j] + epsilon) - math.log(K[i, j]))
    term3 = N[i] * (math.log(Q[i]) - math.log(Q[i] + epsilon))
    if i == j:
        return math.exp(term1 + term3)
    term2 = Z[j, i] * (math.log(K[j, i] + epsilon) - math.log(K[j, i]))
    term4 = N[j] * (math.log(Q[j]) - math.log(Q[j] + epsilon))
    return math.exp(term1 + term2 + term3 + term4)


@njit(cache=True, nogil=True)
def _mcmc_steps(Z, N, K, Q, random, sc, n_steps):
    """Advance one chain by n_steps. ``sc`` is a length-1 array holding the
    chain's running sum of K. ``random`` holds 4 * n_steps uniforms."""
    n = Z.shape[0]
    r_i = 0
    for _ in range(n_steps):
        i = int(random[r_i] * n)
        j = int(random[r_i + 1] * n)
        if i == j:
            a = max(-K[i, j], _K_MINUS - sc[0])
            b = _K_PLUS - sc[0]
        else:
            a = max(-K[i, j], 0.5 * (_K_MINUS - sc[0]))
            b = 0.5 * (_K_PLUS - sc[0])
        epsilon = a + random[r_i + 2] * (b - a)
        cutoff = _acceptance_ratio(i, j, epsilon, Z, K, N, Q)
        r = random[r_i + 3]
        r_i += 4
        if r < cutoff:
            K[i, j] += epsilon
            sc[0] += epsilon
            Q[i] += epsilon
            if i != j:
                K[j, i] += epsilon
                sc[0] += epsilon
                Q[j] += epsilon


@njit(cache=True, parallel=True)
def _advance_chains(Z, N, threadK, threadQ, threadSC, rand, n_thin):
    for tid in prange(threadK.shape[0]):
        _mcmc_steps(Z, N, threadK[tid], threadQ[tid], rand[tid], threadSC[tid], n_thin)


def metzner_mcmc_fast(Z, n_samples, n_thin=1, n_chains=1, random_state=None):
    r"""Metropolis Markov chain Monte Carlo sampler for reversible transition
    matrices

    Parameters
    ----------
    Z : np.array, shape=(n_states, n_states)
        The effective count matrix, the number of observed transitions
        between states plus the number of prior counts
    n_samples : int
        Number of steps to iterate each chain for
    n_thin : int
        Yield every ``n_thin``-th sample from the MCMC chain
    n_chains : int
        Number of parallel MCMC chains to run. The "inner iterations" of
        each MCMC chain advancing by ``n_thin`` iterations run in parallel.
    random_state : int or RandomState instance or None (default)
        Pseudo Random Number generator seed control. If None, use the
        numpy.random singleton.

    Notes
    -----
    The transition matrix posterior distribution is ::

        P(T | Z) \propto \Prod_{ij} T_{ij}^{Z_{ij}}

    and constrained to be reversible, such that there exists a \pi s.t. ::

        \pi_i T_{ij} = \pi_j T_{ji}

    Yields
    ------
    T : np.array, shape=(n_states, n_states)
        This generator yields samples from the transition matrix posterior

    References
    ----------
    .. [1] P. Metzner, F. Noe and C. Schutte, "Estimating the sampling error:
        Distribution of transition matrices and functions of transition
        matrices for given trajectory data." Phys. Rev. E 80 021106 (2009)

    See Also
    --------
    metzner_mcmc_slow
    """
    Z = np.array(Z, copy=True, dtype=np.float64)
    n_states = len(Z)
    if Z.ndim != 2 or Z.shape[1] != n_states:
        raise ValueError("Z must be square.")
    n_samples, n_thin, n_chains = int(n_samples), int(n_thin), int(n_chains)

    # Random numbers come from numpy rather than being generated inside the
    # kernel, so with synchronized seeds this sampler and metzner_mcmc_slow
    # yield exactly the same values (when n_chains=1), which helps testing.
    random = check_random_state(random_state)

    # K is the independent variable being sampled: a symmetric, non-negative
    # "virtual" count matrix. Q are its row sums, N the row sums of Z.
    K = 0.5 * (Z + Z.T) / np.sum(Z)
    Q = np.sum(K, axis=1, dtype=float)
    N = np.sum(Z, axis=1, dtype=float)

    threadQ = np.repeat(Q.reshape(1, n_states), n_chains, axis=0)
    threadK = np.repeat(K.reshape(1, n_states, n_states), n_chains, axis=0)
    threadSC = np.full((n_chains, 1), np.sum(K))

    for _ in range(n_samples // n_thin):
        rand = random.rand(n_chains, 4 * n_thin)
        _advance_chains(Z, N, threadK, threadQ, threadSC, rand, n_thin)
        for tid in range(n_chains):
            yield threadK[tid] / threadQ[tid][:, np.newaxis]
