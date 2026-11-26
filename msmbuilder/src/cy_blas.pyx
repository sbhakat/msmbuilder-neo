# cython: boundscheck=False, cdivision=True, wraparound=False
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Direct BLAS DGEMM (matrix multiply) calls from cython.

See the discussion on scipy-list about this pattern.
http://comments.gmane.org/gmane.comp.python.scientific.devel/19041

Furthermore, since I like to (1) deal with c-major arrays in memory,
and (2) can't remember the DGEMM call signature, this module provides
copy-free wrappers, following from Christoph Lassner's blog post:

  http://www.christophlassner.de/using-blas-from-c-with-row-major-data.html
"""

cimport cython
from scipy.linalg.cython_blas cimport (
    dgemm as FORTRAN_DGEMM, dgemv as FORTRAN_DGEMV, ddot as FORTRAN_DDOT,
    dnrm2 as FORTRAN_DNRM2, daxpy as FORTRAN_DAXPY, dger as FORTRAN_DGER)


cdef char* _N = "N"
cdef char* _T = "T"


@cython.boundscheck(False)
cdef inline int cdgemm_NN(const double[:, ::1] a, const double[:, ::1] b, double[:, ::1] c, double alpha=1.0, double beta=0.0) noexcept nogil:
    """c = beta*c + alpha*dot(a, b)
    """
    cdef int m, k, n
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]
    if a.shape[1] != b.shape[0] or a.shape[0] != c.shape[0] or b.shape[1] != c.shape[1]:
        return -1

    FORTRAN_DGEMM(_N, _N, &n, &m, &k, &alpha, <double*>&b[0,0], &n, <double*>&a[0,0], &k, &beta, &c[0,0], &n)
    return 0


@cython.boundscheck(False)
cdef inline int cdgemm_NT(const double[:, ::1] a, const double[:, ::1] b, double[:, ::1] c, double alpha=1.0, double beta=0.0) noexcept nogil:
    """c = beta*c + alpha*dot(a, b.T)
    """
    cdef int m, k, n
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[0]
    if a.shape[1] != b.shape[1] or a.shape[0] != c.shape[0] or b.shape[0] != c.shape[1]:
        return -1

    FORTRAN_DGEMM(_T, _N, &n, &m, &k, &alpha, <double*>&b[0,0], &k, <double*>&a[0,0], &k, &beta, &c[0,0], &n)
    return 0


@cython.boundscheck(False)
cdef inline int cdgemm_TN(const double[:, ::1] a, const double[:, ::1] b, double[:, ::1] c, double alpha=1.0, double beta=0.0) noexcept nogil:
    """c = beta*c + alpha*dot(a.T, b)
    """
    cdef int m, k, n
    m = a.shape[1]
    k = a.shape[0]
    n = b.shape[1]
    if a.shape[0] != b.shape[0] or a.shape[1] != c.shape[0] or b.shape[1] != c.shape[1]:
        return -1

    FORTRAN_DGEMM(_N, _T, &n, &m, &k, &alpha, <double*>&b[0,0], &n, <double*>&a[0,0], &m, &beta, &c[0,0], &n)
    return 0


@cython.boundscheck(False)
cdef int cdgemv_N(const double[:, ::1] a, const double[:] x, double[:] y, double alpha=1.0, double beta=0.0) noexcept nogil:
    cdef int m, n, incx, incy
    m = a.shape[0]
    n = a.shape[1]
    incx = x.strides[0] // sizeof(double)
    incy = y.strides[0] // sizeof(double)
    if a.shape[1] != x.shape[0] or a.shape[0] != y.shape[0]:
        return -1

    FORTRAN_DGEMV(_T, &n, &m, &alpha, <double*>&a[0,0], &n, <double*>&x[0], &incx, &beta, <double*>&y[0], &incy)
    return 0


@cython.boundscheck(False)
cdef int cdgemv_T(const double[:, ::1] a, const double[:] x, double[:] y, double alpha=1.0, double beta=0.0) noexcept nogil:
    cdef int m, n, incx, incy
    m = a.shape[1]
    n = a.shape[0]
    incx = x.strides[0] // sizeof(double)
    incy = y.strides[0] // sizeof(double)
    if a.shape[0] != x.shape[0] or a.shape[1] != y.shape[0]:
        return -1

    FORTRAN_DGEMV(_N, &m, &n, &alpha, <double*>&a[0,0], &m, <double*>&x[0], &incx, &beta, <double*>&y[0], &incy)
    return 0


@cython.boundscheck(False)
cdef int cddot(const double[:] x, const double[:] y, double *result) noexcept nogil:
    cdef int n, incx, incy
    n = x.shape[0]
    incx = x.strides[0] // sizeof(double)
    incy = y.strides[0] // sizeof(double)
    if y.shape[0] != n:
        return -1

    result[0] = FORTRAN_DDOT(&n, <double*>&x[0], &incx, <double*>&y[0], &incy)
    return 0


@cython.boundscheck(False)
cdef int cdnrm2(double[:] x, double* result) noexcept nogil:
    cdef int n = x.shape[0]
    cdef int incx = x.strides[0] // sizeof(double)
    result[0] = FORTRAN_DNRM2(&n, <double*>&x[0], &incx)
    return 0


@cython.boundscheck(False)
cdef int cdaxpy(double alpha, const double[:] x, double[:] y) noexcept nogil:
    cdef int n = x.shape[0]
    cdef int incx = x.strides[0] // sizeof(double)
    cdef int incy = y.strides[0] // sizeof(double)
    FORTRAN_DAXPY(&n, &alpha, <double*>&x[0], &incx, <double*>&y[0], &incy)
    return 0


@cython.boundscheck(False)
cdef int cdger(double alpha, double[:] x, double[:] y, double[::1, :] A) noexcept nogil:
    cdef int n, m, incx, incy, lda
    m = x.shape[0]
    n = y.shape[0]
    incx = x.strides[0] // sizeof(double)
    incy = y.strides[0] // sizeof(double)
    # dger_t(int *n, int *m, d *alpha, d *x, int *incx, d *y, int *incy, d *a, int *lda)
    FORTRAN_DGER(&m, &n, &alpha, <double*>&x[0], &incx, &y[0], &incy, &A[0,0], &m)
    return 0
