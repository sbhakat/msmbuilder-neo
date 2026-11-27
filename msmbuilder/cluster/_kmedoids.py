# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.
#
# Ported from the C Clustering Library; its notice (also in THIRD_PARTY_LICENSES.txt):
#
# The C clustering library.
# Copyright (C) 2002 Michiel Jan Laurens de Hoon.
#
# This library was written at the Laboratory of DNA Information Analysis,
# Human Genome Center, Institute of Medical Science, University of Tokyo,
# 4-6-1 Shirokanedai, Minato-ku, Tokyo 108-8639, Japan.
# Contact: mdehoon 'AT' gsc.riken.jp
#
# Permission to use, copy, modify, and distribute this software and its
# documentation with or without modifications and for any purpose and
# without fee is hereby granted, provided that any copyright notices
# appear in all copies and that both those copyright notices and this
# permission notice appear in supporting documentation, and that the
# names of the contributors or copyright holders not be used in
# advertising or publicity pertaining to distribution of the software
# without specific prior permission.
#
# THE CONTRIBUTORS AND COPYRIGHT HOLDERS OF THIS SOFTWARE DISCLAIM ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL THE
# CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY SPECIAL, INDIRECT
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
# OR PERFORMANCE OF THIS SOFTWARE.
"""k-medoids clustering on a condensed distance matrix.

Numba port of the former kmedoids.cc (from Michiel de Hoon's C Clustering
Library) and its Cython wrapper. The random initial assignment still calls
``RandomState.binomial`` and ``RandomState.shuffle`` in the same order as the
C code did, so results for a given ``random_state`` are unchanged.
"""
import numpy as np
from numba import njit
from sklearn.utils import check_random_state

_DBL_MAX = np.finfo(np.float64).max


@njit(cache=True, nogil=True)
def _ix(i, j, n):
    """Index of (i, j), i != j, in a condensed distance matrix."""
    if i < j:
        return n * i - i * (i + 1) // 2 + j - 1 - i
    return n * j - j * (j + 1) // 2 + i - 1 - j


@njit(cache=True, nogil=True)
def _getclustermedoids(nclusters, distmatrix, clusterid, centroids, errors):
    n = clusterid.shape[0]
    for j in range(nclusters):
        errors[j] = _DBL_MAX
    for i in range(n):
        d = 0.0
        j = clusterid[i]
        for k in range(n):
            if i == k or clusterid[k] != j:
                continue
            d += distmatrix[_ix(i, k, n)]
            if d > errors[j]:
                break
        if d < errors[j]:
            errors[j] = d
            centroids[j] = i


@njit(cache=True, nogil=True)
def _one_pass(nclusters, distmatrix, tclusterid, saved, centroids, errors):
    """Iterate medoid updates and reassignment from the current tclusterid
    until the total stops decreasing or a previous solution reappears.
    Returns the final total distance."""
    n = tclusterid.shape[0]
    total = _DBL_MAX
    counter = 0
    period = 10
    while True:
        previous = total
        total = 0.0
        if counter % period == 0:  # save the current assignments
            for i in range(n):
                saved[i] = tclusterid[i]
            if period < (2**62):
                period *= 2
        counter += 1

        _getclustermedoids(nclusters, distmatrix, tclusterid, centroids, errors)

        for i in range(n):  # find the closest medoid
            distance = _DBL_MAX
            for icluster in range(nclusters):
                j = centroids[icluster]
                if i == j:
                    distance = 0.0
                    tclusterid[i] = icluster
                    break
                tdistance = distmatrix[_ix(i, j, n)]
                if tdistance < distance:
                    distance = tdistance
                    tclusterid[i] = icluster
            total += distance
        if total >= previous:
            break
        same = True
        for i in range(n):
            if saved[i] != tclusterid[i]:
                same = False
                break
        if same:
            break  # identical solution found
    return total


def _randomassign(nclusters, nelements, random):
    """Random initial assignment with every cluster non-empty: cluster sizes
    from a multinomial (as successive binomials), then shuffled. Matches the
    sequence of RandomState calls made by the old C code."""
    clusterid = np.empty(nelements, dtype=np.intp)
    n = nelements - nclusters
    k = 0
    for i in range(nclusters - 1):
        p = 1.0 / (nclusters - i)
        j = int(random.binomial(n, p))
        n -= j
        j += k + 1  # at least one element in cluster i
        clusterid[k:j] = i
        k = j
    clusterid[k:] = nclusters - 1  # the remaining elements go to the last cluster
    random.shuffle(clusterid)
    return clusterid


def kmedoids(n_clusters, distmatrix, n_pass, clusterid=None, random_state=None):
    """KMedoids clustering

    Arguments
    ---------
    n_clusters : int
        The number of clusters to be found.

    distmatrix : double array
        A condensed distance matrix of the pairwise distance between elements.
        This distance matrix should have the form of matrices produced by
        ``scipy.spatial.distance.pdist()``, or ``msmbuilder.libdistance.pdist``
        which is the lower triangular portion of a symmetric matrix in packed
        storage.

    n_pass : int
        The number of times clustering is performed. Clustering is performed
        n_pass times, each time starting from a different (random) initial
        assignment. The clustering solution with the lowest within-cluster
        sum of distances is chosen. If npass==0, then the clustering algorithm
        will be run once, where the initial assignment of elements to clusters
        is taken from the clusterid array.

    clusterid : int[nelements]
        On input, if npass==0, then clusterid contains the initial clustering
        assignment from which the clustering algorithm starts; all numbers in
        clusterid should be between zero and nelements-1 inclusive. If
        npass!=0, clusterid is ignored on input.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    Returns
    --------
    clusterid : int[nelements]
        On output, clusterid contains the clustering solution that was found:
        clusterid contains the number of the cluster to which each item was
        assigned. On output, the number of a cluster is defined as the item
        number of the centroid of the cluster.

    error : double
      The sum of distances to the cluster center of each item in the optimal
      k-medoids clustering solution that was found.

    ifound : int
        If kmedoids is successful: the number of times the optimal clustering
        solution was found. The value of ifound is at least 1; its maximum
        value is npass. If the user requested more clusters than elements
        available, ifound is set to 0.
    """
    distmatrix = np.ascontiguousarray(distmatrix, dtype=np.float64)
    n_clusters = int(n_clusters)
    n_pass = int(n_pass)
    n_elements = int(1 + np.sqrt(8 * len(distmatrix) + 1) / 2.0)
    if len(distmatrix) != (n_elements * (n_elements - 1) // 2):
        raise ValueError('len(distmatrix)=%s is not a valid size '
                         'of a condensed distance matrix, which should be of '
                         'size (N*(N-1)/2) for some positive integer, N' %
                         len(distmatrix))
    if n_clusters > n_elements:
        raise ValueError('Number of clusters requested (%d) greater than '
                         'number of elements (%d)' % (n_clusters, n_elements))
    if clusterid is not None and len(clusterid) != n_elements:
        raise ValueError('clusterid must be None or an array of length '
                         'n_elements')
    if n_pass < 0:
        raise ValueError('n_pass must be greater than or equal to zero.')

    if clusterid is None:
        clusterid = np.zeros(n_elements, dtype=np.intp)
    else:
        clusterid = np.array(clusterid, dtype=np.intp, copy=True)
    random = check_random_state(random_state)

    saved = np.empty(n_elements, dtype=np.intp)
    centroids = np.zeros(n_clusters, dtype=np.intp)
    errors = np.empty(n_clusters)
    error = _DBL_MAX
    ifound = -1
    ipass = 0
    while True:
        if n_pass != 0:
            start = _randomassign(n_clusters, n_elements, random)
            if n_pass <= 1:
                # The C code worked in clusterid itself when npass <= 1.
                clusterid[:] = start
                tclusterid = clusterid
            else:
                tclusterid = start
        else:
            tclusterid = clusterid
        total = _one_pass(n_clusters, distmatrix, tclusterid, saved, centroids, errors)

        new = centroids[tclusterid]
        if np.any(clusterid != new):
            if total < error:
                ifound = 1
                error = total
                clusterid[:] = new
        else:
            ifound += 1
        ipass += 1
        if ipass >= n_pass:
            break
    return clusterid, error, ifound


def contigify_ids(clusterids):
    """Renumber clusters to go from 0 to n_clusters - 1, in order of first
    appearance.

    This function modifies the array inplace, and returns it together with
    the mapping from old to new ids.
    """
    ids = np.asarray(clusterids)
    if ids.size == 0:
        return ids, {}
    uniq, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
    order = np.argsort(first, kind='stable')
    rank = np.empty(len(uniq), dtype=ids.dtype)
    rank[order] = np.arange(len(uniq), dtype=ids.dtype)
    mapping = {int(uniq[k]): int(rank[k]) for k in range(len(uniq))}
    ids[...] = rank[inverse.reshape(ids.shape)]
    return ids, mapping
