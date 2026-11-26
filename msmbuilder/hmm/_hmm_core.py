"""Numba kernels for the hidden Markov models in msmbuilder.hmm.

These replace the C++ fitters (HMMFitter.h, GaussianHMMFitter.cpp,
VonMisesHMMFitter.cpp). The arithmetic follows the C++ code step by step
(same log-space recursions, same loop order inside a trajectory) so results
agree to rounding; only the order in which trajectories are summed can differ,
as it already did between OpenMP thread counts.

Trajectories are passed as one concatenated array ``X`` of shape
(n_frames_total, n_features) plus ``offsets`` of length n_trajectories + 1, so
each kernel can loop over trajectories in parallel with ``prange``.
"""
import math

import numpy as np
from numba import njit, prange

# The C++ Gaussian fitter stored log(2*pi) in a float; keep that rounding so
# log-likelihoods match the previous implementation exactly.
_LOG_2PI_F32 = float(np.float32(math.log(2 * math.pi)))
_LOG_2PI = math.log(2 * math.pi)


def concatenate(sequences):
    """Stack sequences into one float64 array and return it with offsets."""
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    X = np.ascontiguousarray(np.concatenate(sequences), dtype=np.float64)
    return X, offsets


def log_transmat(transmat):
    return np.log(np.maximum(np.asarray(transmat, dtype=np.float64), 1e-20))


# --- shared forward-backward ---------------------------------------------------

@njit(cache=True, nogil=True)
def _logsumexp(buf, n):
    m = buf[0]
    for i in range(1, n):
        if buf[i] > m:
            m = buf[i]
    s = 0.0
    for i in range(n):
        s += math.exp(buf[i] - m)
    return math.log(s) + m


@njit(cache=True, nogil=True)
def _forward(flp, log_startprob, log_tm):
    T, S = flp.shape
    fwd = np.empty((T, S))
    work = np.empty(S)
    for i in range(S):
        fwd[0, i] = log_startprob[i] + flp[0, i]
    for t in range(1, T):
        for j in range(S):
            for i in range(S):
                work[i] = fwd[t - 1, i] + log_tm[i, j]
            fwd[t, j] = _logsumexp(work, S) + flp[t, j]
    return fwd


@njit(cache=True, nogil=True)
def _backward(flp, log_tm):
    T, S = flp.shape
    bwd = np.empty((T, S))
    work = np.empty(S)
    for i in range(S):
        bwd[T - 1, i] = 0.0
    for t in range(T - 2, -1, -1):
        for i in range(S):
            for j in range(S):
                work[j] = flp[t + 1, j] + bwd[t + 1, j] + log_tm[i, j]
            bwd[t, i] = _logsumexp(work, S)
    return bwd


@njit(cache=True, nogil=True)
def _estep(flp, log_startprob, log_tm):
    """Posteriors, expected transition counts and log-probability of one
    trajectory, given its per-frame emission log-likelihoods."""
    T, S = flp.shape
    fwd = _forward(flp, log_startprob, log_tm)
    bwd = _backward(flp, log_tm)
    logprob = _logsumexp(fwd[T - 1], S)

    post = np.empty((T, S))
    gamma = np.empty(S)
    for t in range(T):
        for i in range(S):
            gamma[i] = fwd[t, i] + bwd[t, i]
        norm = _logsumexp(gamma, S)
        for i in range(S):
            post[t, i] = math.exp(gamma[i] - norm)

    trans = np.zeros((S, S))
    if T > 1:
        work = np.empty(T - 1)
        for i in range(S):
            for j in range(S):
                for t in range(T - 1):
                    work[t] = (fwd[t, i] + log_tm[i, j] + flp[t + 1, j]
                               + bwd[t + 1, j] - logprob)
                trans[i, j] = math.exp(_logsumexp(work, T - 1))
    return logprob, post, trans


@njit(cache=True, nogil=True)
def _viterbi(flp, log_startprob, log_tm):
    T, S = flp.shape
    lattice = np.empty((T, S))
    for i in range(S):
        lattice[0, i] = log_startprob[i] + flp[0, i]
    for t in range(1, T):
        for i in range(S):
            best = lattice[t - 1, 0] + log_tm[0, i]
            for j in range(1, S):
                v = lattice[t - 1, j] + log_tm[j, i]
                if v > best:
                    best = v
            lattice[t, i] = best + flp[t, i]

    states = np.empty(T, dtype=np.int32)
    max_pos = 0
    for i in range(1, S):
        if lattice[T - 1, i] > lattice[T - 1, max_pos]:
            max_pos = i
    states[T - 1] = max_pos
    logprob = lattice[T - 1, max_pos]
    for t in range(T - 2, -1, -1):
        nxt = states[t + 1]
        max_pos = 0
        for i in range(1, S):
            if lattice[t, i] + log_tm[i, nxt] > lattice[t, max_pos] + log_tm[max_pos, nxt]:
                max_pos = i
        states[t] = max_pos
    return logprob, states


# --- Gaussian emissions --------------------------------------------------------

def gaussian_coefficients(means, variances):
    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    a0 = means * means / variances + np.log(variances)
    a1 = -2.0 * means / variances
    a2 = 1.0 / variances
    return a0, a1, a2


@njit(cache=True, nogil=True)
def _gaussian_flp(X, a0, a1, a2):
    T, F = X.shape
    S = a0.shape[0]
    flp = np.empty((T, S))
    for t in range(T):
        for j in range(S):
            temp = 0.0
            for i in range(F):
                x = X[t, i]
                temp += a0[j, i] + x * (a1[j, i] + x * a2[j, i])
            flp[t, j] = -0.5 * (F * _LOG_2PI_F32 + temp)
    return flp


@njit(cache=True, parallel=True)
def gaussian_estep(X, offsets, log_startprob, log_tm, a0, a1, a2):
    n = offsets.shape[0] - 1
    S, F = a0.shape
    logprobs = np.empty(n)
    trans = np.zeros((n, S, S))
    post = np.zeros((n, S))
    obs = np.zeros((n, S, F))
    obs2 = np.zeros((n, S, F))
    for k in prange(n):
        Xk = X[offsets[k]:offsets[k + 1]]
        flp = _gaussian_flp(Xk, a0, a1, a2)
        lp, p, tc = _estep(flp, log_startprob, log_tm)
        logprobs[k] = lp
        trans[k] = tc
        T = Xk.shape[0]
        for i in range(S):
            s = 0.0
            for t in range(T):
                s += p[t, i]
            post[k, i] = s
            for j in range(F):
                t1 = 0.0
                t2 = 0.0
                for t in range(T):
                    x = Xk[t, j]
                    t1 += x * p[t, i]
                    t2 += x * x * p[t, i]
                obs[k, i, j] = t1
                obs2[k, i, j] = t2
    return (logprobs.sum(), trans.sum(axis=0), post.sum(axis=0),
            obs.sum(axis=0), obs2.sum(axis=0))


@njit(cache=True, parallel=True)
def gaussian_score(X, offsets, log_startprob, log_tm, a0, a1, a2):
    n = offsets.shape[0] - 1
    S = a0.shape[0]
    logprobs = np.empty(n)
    for k in prange(n):
        flp = _gaussian_flp(X[offsets[k]:offsets[k + 1]], a0, a1, a2)
        fwd = _forward(flp, log_startprob, log_tm)
        logprobs[k] = _logsumexp(fwd[fwd.shape[0] - 1], S)
    return logprobs.sum()


@njit(cache=True, nogil=True)
def gaussian_viterbi(X, log_startprob, log_tm, a0, a1, a2):
    return _viterbi(_gaussian_flp(X, a0, a1, a2), log_startprob, log_tm)


# --- von Mises emissions -------------------------------------------------------

@njit(cache=True, nogil=True)
def _vonmises_flp(X, means, kappas, log_i0_kappas):
    T, F = X.shape
    S = means.shape[0]
    flp = np.zeros((T, S))
    for i in range(S):
        for j in range(F):
            val = _LOG_2PI + log_i0_kappas[i, j]
            for t in range(T):
                flp[t, i] -= val
    kcos = np.empty((F, S))
    ksin = np.empty((F, S))
    for i in range(S):
        for j in range(F):
            kcos[j, i] = kappas[i, j] * math.cos(means[i, j])
            ksin[j, i] = kappas[i, j] * math.sin(means[i, j])
    for t in range(T):
        for j in range(F):
            c = math.cos(X[t, j])
            s = math.sin(X[t, j])
            for i in range(S):
                flp[t, i] += c * kcos[j, i] + s * ksin[j, i]
    return flp


@njit(cache=True, parallel=True)
def vonmises_estep(X, offsets, log_startprob, log_tm, means, kappas, log_i0_kappas):
    n = offsets.shape[0] - 1
    S, F = means.shape
    logprobs = np.empty(n)
    trans = np.zeros((n, S, S))
    post = np.zeros((n, S))
    cosobs = np.zeros((n, S, F))
    sinobs = np.zeros((n, S, F))
    for k in prange(n):
        Xk = X[offsets[k]:offsets[k + 1]]
        flp = _vonmises_flp(Xk, means, kappas, log_i0_kappas)
        lp, p, tc = _estep(flp, log_startprob, log_tm)
        logprobs[k] = lp
        trans[k] = tc
        T = Xk.shape[0]
        cosx = np.empty((F, T))
        sinx = np.empty((F, T))
        for j in range(F):
            for t in range(T):
                cosx[j, t] = math.cos(Xk[t, j])
                sinx[j, t] = math.sin(Xk[t, j])
        for i in range(S):
            s = 0.0
            for t in range(T):
                s += p[t, i]
            post[k, i] = s
            for j in range(F):
                t1 = 0.0
                t2 = 0.0
                for t in range(T):
                    t1 += cosx[j, t] * p[t, i]
                    t2 += sinx[j, t] * p[t, i]
                cosobs[k, i, j] = t1
                sinobs[k, i, j] = t2
    return (logprobs.sum(), trans.sum(axis=0), post.sum(axis=0),
            cosobs.sum(axis=0), sinobs.sum(axis=0))


@njit(cache=True, parallel=True)
def vonmises_score(X, offsets, log_startprob, log_tm, means, kappas, log_i0_kappas):
    n = offsets.shape[0] - 1
    S = means.shape[0]
    logprobs = np.empty(n)
    for k in prange(n):
        flp = _vonmises_flp(X[offsets[k]:offsets[k + 1]], means, kappas, log_i0_kappas)
        fwd = _forward(flp, log_startprob, log_tm)
        logprobs[k] = _logsumexp(fwd[fwd.shape[0] - 1], S)
    return logprobs.sum()


@njit(cache=True, nogil=True)
def vonmises_viterbi(X, log_startprob, log_tm, means, kappas, log_i0_kappas):
    return _viterbi(_vonmises_flp(X, means, kappas, log_i0_kappas), log_startprob, log_tm)
