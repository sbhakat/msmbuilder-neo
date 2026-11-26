"""Check HMM results and speed against a saved reference.

Run once with the old implementation to record a reference, then again with
the new one to compare:

    python benchmarks/hmm_parity.py --save ref.npz
    python benchmarks/hmm_parity.py --compare ref.npz

Each array must match the reference to within rtol (default 1e-8) times
the largest magnitude in that array. The old C++
fitter summed trajectories in OpenMP thread order, so exact bit-for-bit
equality is not expected.
"""
import argparse
import time

import numpy as np

from msmbuilder.hmm import GaussianHMM, VonMisesHMM


def markov_chain(rs, transmat, length):
    s = np.empty(length, dtype=int)
    s[0] = rs.randint(len(transmat))
    for t in range(1, length):
        s[t] = rs.choice(len(transmat), p=transmat[s[t - 1]])
    return s


TRANSMAT = np.array([[0.90, 0.07, 0.03], [0.05, 0.90, 0.05], [0.04, 0.06, 0.90]])


def vonmises_data(dtype, seed=0, n_traj=8, length=600):
    rs = np.random.RandomState(seed)
    centers = np.array([[-2.0, 1.0], [0.5, -1.5], [2.5, 2.0]])
    seqs = []
    for _ in range(n_traj):
        s = markov_chain(rs, TRANSMAT, length)
        x = centers[s] + 0.4 * rs.randn(length, 2)
        seqs.append(np.arctan2(np.sin(x), np.cos(x)).astype(dtype))
    return seqs


def gaussian_data(dtype, seed=0, n_traj=8, length=600):
    rs = np.random.RandomState(seed)
    means = np.array([[0.0, 0.0], [4.0, 1.0], [1.0, 5.0]])
    seqs = []
    for _ in range(n_traj):
        s = markov_chain(rs, TRANSMAT, length)
        seqs.append((means[s] + rs.randn(length, 2)).astype(dtype))
    return seqs




def cases():
    for dtype in (np.float64, np.float32):
        for init_algo in ("kmeans", "GMM"):
            for rev in ("mle", "transpose"):
                name = f"ghmm-{np.dtype(dtype).name}-{init_algo}-{rev}"
                yield name, (lambda d=dtype, i=init_algo, r=rev: (
                    GaussianHMM(n_states=3, n_init=2, n_iter=15, init_algo=i,
                                reversible_type=r, random_state=0),
                    gaussian_data(d)))
        for rev in ("mle", "transpose"):
            name = f"vmhmm-{np.dtype(dtype).name}-{rev}"
            yield name, (lambda d=dtype, r=rev: (
                VonMisesHMM(n_states=3, n_init=2, n_iter=15, reversible_type=r),
                vonmises_data(d)))


def run_case(make):
    model, seqs = make()
    np.random.seed(1234)  # VonMisesHMM._init uses the global generator
    model.fit(seqs)
    out = {
        "means": np.asarray(model.means_),
        "transmat": np.asarray(model.transmat_),
        "populations": np.asarray(model.populations_),
        "fit_logprob": np.asarray(model.fit_logprob_),
        "score": np.asarray(model.score(seqs)),
    }
    out["second"] = np.asarray(model.vars_ if isinstance(model, GaussianHMM) else model.kappas_)
    logprob, paths = model.predict(seqs)
    out["viterbi_logprob"] = np.asarray(logprob)
    out["viterbi_paths"] = np.concatenate(paths)
    return out


def benchmark():
    rs = np.random.RandomState(0)
    big = [rs.randn(5000, 10) + (i % 3) for i in range(20)]
    angles = [np.arctan2(np.sin(x), np.cos(x)) for x in big]
    timings = {}
    for label, model, data in [
        ("GaussianHMM fit (20 x 5000 x 10, 6 states, 20 iters)",
         GaussianHMM(n_states=6, n_init=1, n_iter=20, thresh=-1, random_state=0), big),
        ("VonMisesHMM fit (20 x 5000 x 10, 6 states, 20 iters)",
         VonMisesHMM(n_states=6, n_init=1, n_iter=20, thresh=-1), angles),
    ]:
        np.random.seed(0)
        model.fit(data[:2])  # warm-up (JIT compilation, caches)
        np.random.seed(0)
        t0 = time.perf_counter()
        model.fit(data)
        timings[label] = time.perf_counter() - t0
        t0 = time.perf_counter()
        model.score(data)
        timings[label.replace("fit", "score").split(",")[0] + ")"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        model.predict(data)
        timings[label.replace("fit", "predict").split(",")[0] + ")"] = time.perf_counter() - t0
    return timings


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", metavar="FILE")
    g.add_argument("--compare", metavar="FILE")
    p.add_argument("--rtol", type=float, default=1e-8)
    args = p.parse_args()

    results = {}
    for name, make in cases():
        for key, value in run_case(make).items():
            results[f"{name}/{key}"] = value
    timings = benchmark()
    for label, seconds in timings.items():
        results[f"time/{label}"] = np.asarray(seconds)

    if args.save:
        np.savez(args.save, **results)
        print(f"saved {len(results)} arrays to {args.save}")
        for label, seconds in timings.items():
            print(f"  {label}: {seconds:.3f} s")
        return

    ref = np.load(args.compare)
    failures = 0
    for key in sorted(k for k in ref.files if not k.startswith("time/")):
        a, b = results[key], ref[key]
        if key.endswith("viterbi_paths"):
            ok = np.array_equal(a, b)
            detail = f"{np.count_nonzero(a != b)} of {a.size} states differ" if not ok else ""
        else:
            # Compare relative to the array's overall scale, so entries that
            # are ~0 (e.g. an empty state's population) don't dominate.
            scale = np.max(np.abs(b)) if b.size else 0.0
            ok = a.shape == b.shape and np.allclose(a, b, rtol=0, atol=args.rtol * scale + 1e-300)
            detail = "" if ok else (f"max diff / scale {np.max(np.abs(a - b)) / scale:.2e}"
                                    if a.shape == b.shape else f"shape {a.shape} vs {b.shape}")
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {key} {detail}")
    print()
    print(f"{'benchmark':58s} {'old':>8s} {'new':>8s} {'new/old':>8s}")
    for label, seconds in timings.items():
        old = float(ref[f"time/{label}"])
        print(f"{label:58s} {old:8.3f} {seconds:8.3f} {seconds / old:8.2f}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
