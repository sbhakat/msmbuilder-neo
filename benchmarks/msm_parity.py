"""Check MSM kernel results and speed against a saved reference.

Covers msmbuilder.msm._ratematrix (rate-matrix likelihood, derivatives,
Hessian and uncertainties), _transmat_mle_prinz (reversible MLE) and
metzner_mcmc_fast (reversible-transition-matrix MCMC), directly and through
ContinuousTimeMSM, MarkovStateModel and BayesianMarkovStateModel.

    python benchmarks/msm_parity.py --save ref.npz
    python benchmarks/msm_parity.py --compare ref.npz

Each array must match the reference to within rtol (default 1e-8) times the
largest magnitude in that array.
"""
import argparse
import time

import numpy as np

from msmbuilder.msm import (BayesianMarkovStateModel, ContinuousTimeMSM,
                            MarkovStateModel, _ratematrix)
from msmbuilder.msm._markovstatemodel import _transmat_mle_prinz
from msmbuilder.msm._metzner_mcmc_fast import metzner_mcmc_fast


def random_problem(n, seed):
    rs = np.random.RandomState(seed)
    n_triu = n * (n - 1) // 2
    theta = np.concatenate([rs.uniform(0.05, 1.0, n_triu), rs.uniform(-1.0, 0.0, n)])
    counts = rs.randint(1, 40, size=(n, n)).astype(float)
    return theta, counts


def kernel_results():
    out = {}
    for n, seed in [(3, 0), (5, 1)]:
        theta, counts = random_problem(n, seed)
        size = len(theta)
        tag = f"n{n}"
        for which in ("K", "S"):
            M = np.zeros((n, n))
            _ratematrix.build_ratemat(theta, n, M, which=which)
            out[f"{tag}/build_{which}"] = M
        A = np.random.RandomState(seed + 10).randn(n, n)
        sums, slices = [], []
        for u in range(size):
            dKu = np.zeros((n, n))
            sums.append(_ratematrix.dK_dtheta_ij(theta, n, u, A, dKu))
            slices.append(dKu)
        out[f"{tag}/dK_dtheta_ij_sum"] = np.array(sums)
        out[f"{tag}/dK_dtheta_ij"] = np.array(slices)
        B = np.random.RandomState(seed + 20).randn(size, size)
        g1, g2 = [], []
        for i in range(n):
            for j in range(n):
                o, o2 = np.zeros(size), np.zeros(size)
                _ratematrix.dK_dtheta_u(theta, n, i, j, out=o, A=B, out2=o2)
                g1.append(o)
                g2.append(o2)
        out[f"{tag}/dK_dtheta_u"] = np.array(g1)
        out[f"{tag}/dK_dtheta_u_A"] = np.array(g2)
        for t in (1.0, 2.5):
            logl, grad = _ratematrix.loglikelihood(theta, counts, t)
            out[f"{tag}/loglik_t{t}"] = np.array(logl)
            out[f"{tag}/grad_t{t}"] = np.asarray(grad)
        H = _ratematrix.hessian(theta, counts)
        out[f"{tag}/hessian"] = H
        out[f"{tag}/hessian_inds"] = _ratematrix.hessian(
            theta, counts, inds=np.array([0, size - 1, 1], dtype=np.intp))
        covar = np.linalg.pinv(-H)
        covar = 0.5 * (covar + covar.T)
        out[f"{tag}/sigma_K"] = _ratematrix.sigma_K(covar, theta, n)
        out[f"{tag}/sigma_pi"] = _ratematrix.sigma_pi(covar, theta, n)
        out[f"{tag}/sigma_eigenvalues"] = _ratematrix.sigma_eigenvalues(covar, theta, n)
        out[f"{tag}/sigma_timescales"] = _ratematrix.sigma_timescales(covar, theta, n)
        pi = np.exp(theta[-n:])
        S = np.zeros((n, n))
        _ratematrix.build_ratemat(theta, n, S, which="S")
        w, U, V = map(np.asarray, _ratematrix.eig_K(S, n, pi, "S"))
        out[f"{tag}/eigS_w"], out[f"{tag}/eigS_U"], out[f"{tag}/eigS_V"] = w, U, V
        dKu = np.zeros((n, n))
        _ratematrix.dK_dtheta_ij(theta, n, 0, None, dKu)
        dw = np.zeros(n)
        _ratematrix.dw_du(dKu, np.ascontiguousarray(U), np.ascontiguousarray(V), n, np.zeros(n), dw)
        out[f"{tag}/dw_du"] = dw
        alpha = np.arange(1, n + 1, dtype=float)
        grad = np.zeros(n)
        out[f"{tag}/ldirichlet"] = np.array(
            _ratematrix.ldirichlet_softmax(theta[-n:].copy(), alpha, grad))
        out[f"{tag}/ldirichlet_grad"] = grad
        grad = np.zeros(size)
        out[f"{tag}/lexponential"] = np.array(
            _ratematrix.lexponential(theta, np.linspace(0.5, 2.0, size), grad))
        out[f"{tag}/lexponential_grad"] = grad
        T, pops = _transmat_mle_prinz(counts)
        out[f"{tag}/prinz_T"], out[f"{tag}/prinz_pi"] = T, pops
        for thin in (1, 3):
            samples = list(metzner_mcmc_fast(counts, 60, n_thin=thin, random_state=seed))
            out[f"{tag}/metzner_thin{thin}"] = np.array(samples)
    return out


def model_results():
    rs = np.random.RandomState(42)
    P = np.array([[0.90, 0.08, 0.02, 0.00],
                  [0.05, 0.85, 0.08, 0.02],
                  [0.01, 0.09, 0.85, 0.05],
                  [0.00, 0.03, 0.07, 0.90]])
    seqs = []
    for _ in range(5):
        s = [0]
        for _ in range(2000):
            s.append(rs.choice(4, p=P[s[-1]]))
        seqs.append(np.array(s))
    out = {}
    m = ContinuousTimeMSM(lag_time=2).fit(seqs)
    out["ctmsm/ratemat"] = m.ratemat_
    out["ctmsm/timescales"] = m.timescales_
    out["ctmsm/uncertainty_K"] = m.uncertainty_K()
    out["ctmsm/uncertainty_pi"] = m.uncertainty_pi()
    out["ctmsm/uncertainty_timescales"] = m.uncertainty_timescales()
    m = MarkovStateModel(lag_time=1, reversible_type="mle").fit(seqs)
    out["msm/transmat"] = m.transmat_
    out["msm/populations"] = m.populations_
    b = BayesianMarkovStateModel(n_samples=40, n_steps=5, n_chains=1, random_state=3).fit(seqs)
    out["bayesmsm/transmats"] = b.all_transmats_
    return out


def benchmark():
    timings = {}
    theta, counts = random_problem(30, 7)

    def timed(label, fn, repeat=3):
        fn()  # warm-up (JIT compilation)
        best = np.inf
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        timings[label] = best

    timed("loglikelihood + gradient (n=30)", lambda: _ratematrix.loglikelihood(theta, counts, 1.0))
    timed("hessian (n=30)", lambda: _ratematrix.hessian(theta, counts), repeat=1)
    big = np.random.RandomState(0).randint(1, 50, size=(300, 300)).astype(float)
    timed("transmat_mle_prinz (300 states)", lambda: _transmat_mle_prinz(big))
    timed("metzner_mcmc_fast (30 states, 200k steps)",
          lambda: list(metzner_mcmc_fast(counts, 200000, n_thin=200000, random_state=0)))
    # A structured chain (metastable states with nearest-neighbour hops). Data
    # with uniformly random transitions gives an almost flat likelihood, where
    # L-BFGS paths (and so timings) depend on last-digit rounding.
    n = 8
    P = np.full((n, n), 0.01)
    np.fill_diagonal(P, 0.9)
    for i in range(n - 1):
        P[i, i + 1] = P[i + 1, i] = 0.04
    P /= P.sum(axis=1, keepdims=True)
    rs = np.random.RandomState(0)
    seqs = []
    for _ in range(4):
        s = [0]
        for _ in range(20000):
            s.append(rs.choice(n, p=P[s[-1]]))
        seqs.append(np.array(s))
    timed("ContinuousTimeMSM.fit (8 states, 80k frames, lag 5)",
          lambda: ContinuousTimeMSM(lag_time=5).fit(seqs), repeat=3)
    return timings


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", metavar="FILE")
    g.add_argument("--compare", metavar="FILE")
    p.add_argument("--rtol", type=float, default=1e-8)
    args = p.parse_args()

    results = {**kernel_results(), **model_results()}
    timings = benchmark()
    for label, seconds in timings.items():
        results[f"time/{label}"] = np.asarray(seconds)

    if args.save:
        np.savez(args.save, **results)
        print(f"saved {len(results)} arrays to {args.save}")
        for label, seconds in timings.items():
            print(f"  {label}: {seconds:.4f} s")
        return

    ref = np.load(args.compare)
    failures = 0
    for key in sorted(k for k in ref.files if not k.startswith("time/")):
        a, b = np.asarray(results[key]), ref[key]
        scale = np.max(np.abs(b)) if b.size else 0.0
        ok = a.shape == b.shape and np.allclose(a, b, rtol=0, atol=args.rtol * scale + 1e-300)
        detail = "" if ok else (f"max diff / scale {np.max(np.abs(a - b)) / scale:.2e}"
                                if a.shape == b.shape else f"shape {a.shape} vs {b.shape}")
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {key} {detail}")
    print()
    print(f"{'benchmark':48s} {'old':>9s} {'new':>9s} {'new/old':>8s}")
    for label, seconds in timings.items():
        old = float(ref[f"time/{label}"])
        print(f"{label:48s} {old:9.4f} {seconds:9.4f} {seconds / old:8.2f}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
