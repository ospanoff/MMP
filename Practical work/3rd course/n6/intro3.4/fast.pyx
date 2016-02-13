from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt
import cython
import numpy as np
cimport numpy as np

# export used cpp functions from header
cdef extern from 'fast_impl.h':
    float CalcAvgDist(vector[float] px, vector[float] py)
cdef extern from 'fast_impl.h':
    float CalcAvgDistPtr(float* px, float* py, int n)


# really naive algo
def naive(px, py):
    n = len(px)
    sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sum += ((px[i] - px[j]) ** 2 + (py[i] - py[j]) ** 2) ** 0.5

    return 2 * sum / (n * (n - 1))


# naive algo with vector
def naive_with_vector(px, py):
    cdef vector[float] vx
    cdef vector[float] vy
    cdef int n = len(px)
    vx.resize(n)
    vy.resize(n)
    cdef int i, j
    for i in range(n):
        (vx[i], vy[i]) = (px[i], py[i])
    cdef float sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sum += sqrt((vx[i] - vx[j]) * (vx[i] - vx[j]) + (vy[i] - vy[j]) * (vy[i] - vy[j]));
    return 2 * sum / (n * (n - 1))

# malloc and free demonstration
def naive_with_malloc(px, py):
    cdef int n = len(px)
    cdef float* vx = <float*>malloc(n * sizeof(float))
    cdef float* vy = <float*>malloc(n * sizeof(float))
    cdef int i, j

    for i in range(n):
        (vx[i], vy[i]) = (px[i], py[i])

    cdef float sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sum += sqrt((vx[i] - vx[j]) * (vx[i] - vx[j]) + (vy[i] - vy[j]) * (vy[i] - vy[j]));

    free(vx)
    free(vy)
    return 2 * sum / (n * (n - 1))

# using cpp code with vectors demonstration
def naive_with_cpp(px, py):
    cdef vector[float] vx
    cdef vector[float] vy

    cdef int n = len(px)
    vx.resize(n)
    vy.resize(n)
    cdef int i, j

    for i in range(n):
        (vx[i], vy[i]) = (px[i], py[i])

    return CalcAvgDist(vx, vy)


# using fast numpy
# we should disable bounds checking(and also checking indexes for positiveness)
# to achive really good performance
@cython.boundscheck(False)
@cython.wraparound(False)
def naive_with_np(np.ndarray[float, ndim=1, mode='c'] px not None, np.ndarray[float, ndim=1, mode='c'] py not None):
    cdef float sum = 0.0
    cdef int n = px.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            sum += sqrt((px[i] - px[j]) * (px[i] - px[j]) + (py[i] - py[j]) * (py[i] - py[j]))
    return 2 * sum / (n * (n - 1))


# pass the real pointers from np.array to the real cpp code
@cython.boundscheck(False)
@cython.wraparound(False)
def naive_with_np_cpp(np.ndarray[float, ndim=1, mode='c'] px not None, np.ndarray[float, ndim=1, mode='c'] py not None):
    cdef int n = px.shape[0]
    return dCalcAvgDistPtr(&px[0], &py[0], n)
