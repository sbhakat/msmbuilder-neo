"""Check libdistance, k-medoids, sparse tICA and the Muller example against a
saved reference, and time them.

    python benchmarks/native_parity.py --save ref.npz
    python benchmarks/native_parity.py --compare ref.npz

Each array must match the reference to within rtol (default 1e-8) times the
largest magnitude in that array; integer arrays (labels, assignments) must
match exactly. RMSD tests use the alanine dipeptide and Fs peptide example
datasets (fetched into ~/msmbuilder_data by msmbuilder/tests/data_init.py).
"""
import argparse
import time

import numpy as np

from msmbuilder import libdistance
from msmbuilder.cluster import KMedoids, MiniBatchKMedoids
from msmbuilder.cluster._kmedoids import contigify_ids, kmedoids
from msmbuilder.decomposition import SparseTICA
from msmbuilder.decomposition._speigh import scdeflate, speigh
from msmbuilder.example_datasets import AlanineDipeptide, FsPeptide
from msmbuilder.example_datasets._muller import muller_potential, propagate

METRICS = ("euclidean", "sqeuclidean", "cityblock", "chebyshev", "canberra",
           "braycurtis", "hamming", "jaccard")

# Differences that are understood and accepted. They are reported as "known"
# instead of failing the comparison.
KNOWN_DIFFERENCES = {
    "rmsd/pdist_idx": (
        "the index list repeats frame 10, so one pair is a frame with itself: "
        "the kernel vendored in msmbuilder2022 returned exactly 0.0, mdtraj's "
        "returns ~1e-4 nm (float32 noise in QCP for identical structures); "
        "all other pairs match"),
}


def distance_results():
    out = {}
    rs = np.random.RandomState(0)
    for dtype in (np.float64, np.float32):
        X = rs.randn(40, 5).astype(dtype)
        Y = rs.randn(7, 5).astype(dtype)
        # some exact zeros and repeats, so hamming/jaccard/canberra branches run
        X[::4, 1] = 0
        X[1::5] = X[0]
        idx = rs.randint(0, 40, size=12).astype(np.intp)
        pairs = rs.randint(0, 40, size=(15, 2)).astype(np.intp)
        tag = np.dtype(dtype).name
        for m in METRICS:
            a, inertia = libdistance.assign_nearest(X, Y, m)
            out[f"{tag}/{m}/assign"] = a
            out[f"{tag}/{m}/inertia"] = np.array(inertia)
            a, inertia = libdistance.assign_nearest(X, Y, m, X_indices=idx)
            out[f"{tag}/{m}/assign_idx"] = a
            out[f"{tag}/{m}/inertia_idx"] = np.array(inertia)
            out[f"{tag}/{m}/cdist"] = libdistance.cdist(X, Y, m)
            out[f"{tag}/{m}/pdist"] = libdistance.pdist(X, m)
            out[f"{tag}/{m}/pdist_idx"] = libdistance.pdist(X, m, X_indices=idx)
            out[f"{tag}/{m}/dist"] = libdistance.dist(X, Y[0], m)
            out[f"{tag}/{m}/dist_idx"] = libdistance.dist(X, Y[0], m, X_indices=idx)
            out[f"{tag}/{m}/sumdist"] = np.array(libdistance.sumdist(X, m, pairs))
    traj = AlanineDipeptide().get_cached().trajectories[0]
    X, Y = traj[:60], traj[100:108]
    idx = np.array([3, 0, 59, 10, 10, 41], dtype=np.intp)
    pairs = np.array([[0, 1], [5, 59], [10, 10], [33, 2]], dtype=np.intp)
    a, inertia = libdistance.assign_nearest(X, Y, "rmsd")
    out["rmsd/assign"], out["rmsd/inertia"] = a, np.array(inertia)
    a, inertia = libdistance.assign_nearest(X, Y, "rmsd", X_indices=idx)
    out["rmsd/assign_idx"], out["rmsd/inertia_idx"] = a, np.array(inertia)
    out["rmsd/cdist"] = libdistance.cdist(X, Y, "rmsd")
    out["rmsd/pdist"] = libdistance.pdist(X, "rmsd")
    out["rmsd/pdist_idx"] = libdistance.pdist(X, "rmsd", X_indices=idx)
    out["rmsd/dist"] = libdistance.dist(X, Y[0], "rmsd")
    out["rmsd/dist_idx"] = libdistance.dist(X, Y[0], "rmsd", X_indices=idx)
    out["rmsd/sumdist"] = np.array(libdistance.sumdist(X, "rmsd", pairs))
    return out


def kmedoids_results():
    out = {}
    rs = np.random.RandomState(1)
    X = np.concatenate([rs.randn(30, 2), rs.randn(30, 2) + 6, rs.randn(30, 2) + [0, 8]])
    dmat = libdistance.pdist(X, "euclidean")
    for n_pass in (0, 1, 5):
        init = (np.arange(len(X)) % 3).astype(np.intp) if n_pass == 0 else None
        ids, err, found = kmedoids(3, dmat, n_pass, clusterid=init, random_state=2)
        out[f"kmedoids/pass{n_pass}/ids"] = ids
        out[f"kmedoids/pass{n_pass}/error"] = np.array(err)
        out[f"kmedoids/pass{n_pass}/ifound"] = np.array(found)
    ids = np.array([7, 7, 3, 9, 3, 7, 12], dtype=np.intp)
    new, mapping = contigify_ids(ids.copy())
    out["contigify/ids"] = new
    out["contigify/mapping"] = np.array(sorted(mapping.items()))
    m = KMedoids(n_clusters=3, n_passes=4, random_state=0).fit([X])
    out["KMedoids/labels"] = m.labels_[0]
    out["KMedoids/inertia"] = np.array(m.inertia_)
    m = MiniBatchKMedoids(n_clusters=3, batch_size=40, max_iter=5, random_state=0).fit([X])
    out["MiniBatchKMedoids/labels"] = m.labels_[0]
    out["MiniBatchKMedoids/inertia"] = np.array(m.inertia_)
    return out


def speigh_results():
    out = {}
    rs = np.random.RandomState(2)
    for n in (4, 9):
        M = rs.randn(n, n)
        A = M + M.T
        Bm = rs.randn(n, n)
        B = Bm @ Bm.T + n * np.eye(n)
        for rho in (0.0, 0.01, 0.1):
            u, v = speigh(A, B, rho=rho)
            out[f"speigh/n{n}/rho{rho}/u"] = np.array(u)
            out[f"speigh/n{n}/rho{rho}/v"] = v
        out[f"scdeflate/n{n}"] = scdeflate(A, rs.randn(n))
    X = [np.cumsum(rs.randn(800, 6), axis=0) for _ in range(3)]
    m = SparseTICA(n_components=2, lag_time=2, rho=0.01).fit(X)
    out["SparseTICA/components"] = m.components_
    out["SparseTICA/eigenvalues"] = m.eigenvalues_
    return out


def muller_results():
    out = {}
    out["muller/propagate"] = propagate(n_steps=2000, x0=(-0.5, 1.2), thin=8,
                                        random_state=3)
    out["muller/propagate_bounded"] = propagate(n_steps=2000, thin=1, random_state=4,
                                                min_x=-1.4, max_x=1.1, min_y=-0.1,
                                                max_y=1.9)
    gx, gy = np.meshgrid(np.linspace(-1.5, 1.2, 9), np.linspace(-0.2, 2, 7))
    out["muller/potential"] = muller_potential(gx, gy, beta=0.5)
    return out


def benchmark():
    timings = {}

    def timed(label, fn, repeat=3):
        fn()  # warm-up (JIT compilation)
        best = np.inf
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        timings[label] = best

    rs = np.random.RandomState(0)
    X64 = rs.randn(20000, 10)
    Y64 = rs.randn(200, 10)
    X32, Y32 = X64.astype(np.float32), Y64.astype(np.float32)
    timed("assign_nearest euclidean (20000 x 200, f64)",
          lambda: libdistance.assign_nearest(X64, Y64, "euclidean"))
    timed("assign_nearest euclidean (20000 x 200, f32)",
          lambda: libdistance.assign_nearest(X32, Y32, "euclidean"))
    timed("cdist canberra (20000 x 200, f64)", lambda: libdistance.cdist(X64, Y64, "canberra"))
    timed("pdist euclidean (3000 points, f64)", lambda: libdistance.pdist(X64[:3000], "euclidean"))
    fs = FsPeptide().get_cached().trajectories[0][:2000]
    centers = fs[::100]
    timed("assign_nearest rmsd (2000 x 20 frames, 264 atoms)",
          lambda: libdistance.assign_nearest(fs, centers, "rmsd"))
    timed("pdist rmsd (500 frames, 264 atoms)", lambda: libdistance.pdist(fs[:500], "rmsd"))
    timed("dist rmsd (2000 frames, 264 atoms)", lambda: libdistance.dist(fs, fs[0], "rmsd"))
    pts = np.concatenate([rs.randn(500, 2), rs.randn(500, 2) + 6])
    dmat = libdistance.pdist(pts, "euclidean")
    timed("kmedoids (1000 points, 4 clusters, 5 passes)",
          lambda: kmedoids(4, dmat, 5, random_state=0))
    M = rs.randn(60, 60)
    A = M + M.T
    Bm = rs.randn(60, 60)
    B = Bm @ Bm.T + 60 * np.eye(60)
    timed("speigh (60 x 60, rho=0.01)", lambda: speigh(A, B, rho=0.01))
    timed("muller propagate (1e6 steps)",
          lambda: propagate(n_steps=1_000_000, thin=100, random_state=0))
    return timings


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", metavar="FILE")
    g.add_argument("--compare", metavar="FILE")
    p.add_argument("--rtol", type=float, default=1e-8)
    args = p.parse_args()

    results = {**distance_results(), **kmedoids_results(), **speigh_results(),
               **muller_results()}
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
        if b.dtype.kind in "iu":
            ok = a.shape == b.shape and np.array_equal(a, b)
            detail = "" if ok else (f"{np.count_nonzero(a != b)} of {b.size} differ"
                                    if a.shape == b.shape else f"shape {a.shape} vs {b.shape}")
        else:
            scale = np.max(np.abs(b)) if b.size else 0.0
            ok = a.shape == b.shape and np.allclose(a, b, rtol=0, atol=args.rtol * scale + 1e-300,
                                                    equal_nan=True)
            detail = "" if ok else (f"max diff / scale {np.nanmax(np.abs(a - b)) / scale:.2e}"
                                    if a.shape == b.shape else f"shape {a.shape} vs {b.shape}")
        if not ok and key in KNOWN_DIFFERENCES:
            print(f"known {key} {detail}: {KNOWN_DIFFERENCES[key]}")
            continue
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {key} {detail}")
    print()
    print(f"{'benchmark':52s} {'old':>9s} {'new':>9s} {'new/old':>8s}")
    for label, seconds in timings.items():
        old = float(ref[f"time/{label}"])
        print(f"{label:52s} {old:9.4f} {seconds:9.4f} {seconds / old:8.2f}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
