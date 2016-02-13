import cython
import numpy as np
cimport numpy as np


from libcpp.vector cimport vector

cdef extern from 'k_medoids.h':
    void k_medoids_cpp(double *points, int points_number,
                       int clusters_number, vector[long] clusterization)


@cython.boundscheck(False)
@cython.wraparound(False)
def L(np.ndarray[double, ndim=2, mode='c'] dot1 not None,
      np.ndarray[double, ndim=2, mode='c'] dot2 not None):
    """
    dot1: first dot on the geographical coordinates in degrees
        - should be Nx2 np.array
        - N: number of dots
    dot2: second dot on the geographical coordinates in degrees
        - should be Mx2 np.array
        - M: number of dots
    R - radius of the sphere
    """
    cdef int m = dot1.shape[0]
    cdef int n = dot2.shape[0]
    dot1 = np.repeat(np.radians(dot1), n, axis=0)
    dot2 = np.tile(np.radians(dot2).ravel(), m).reshape(-1, 2)

    return (np.arccos(np.sin(dot1[:, 0]) * np.sin(dot2[:, 0]) +
                      np.cos(dot1[:, 0]) * np.cos(dot2[:, 0]) *
                      np.cos(dot1[:, 1] - dot2[:, 1]))).reshape(-1, n)


@cython.boundscheck(False)
@cython.wraparound(False)
def massCenter(np.ndarray[double, ndim=1, mode='c'] dot1 not None,
               np.ndarray[double, ndim=1, mode='c'] dot2 not None):
    """
    dot1: first dot on the geographical coordinates in degrees
        - should be 1x3 np.array
    dot2: second dot on the geographical coordinates in degrees
        - should be 1x3 np.array
    """
    cdef float m = dot1[0] / (dot1[0] + dot2[0])

    coord = np.vstack((np.radians(dot1[1:]), np.radians(dot2[1:])))

    x = np.cos(coord[:, 0]) * np.cos(coord[:, 1])
    y = np.cos(coord[:, 0]) * np.sin(coord[:, 1])
    z = np.sin(coord[:, 0])
    coord = np.vstack((x, y, z)).T

    cdef float alpha = m * np.arccos(np.dot(coord[0], coord[1]))
    v = np.array([np.linalg.det(coord[:, 1:]),
                  -np.linalg.det(coord[:, [0, 2]]),
                  np.linalg.det(coord[:, :2])])
    v /= np.linalg.norm(v)

    cdef float c = np.cos(alpha)
    cdef float s = np.sin(alpha)

    M = np.empty((3, 3))
    M[0] = [c + (1 - c) * (v[0] ** 2), (1 - c) * v[0]
            * v[1] - s * v[2], (1 - c) * v[0] * v[2] + s * v[1]]
    M[1] = [(1 - c) * v[1] * v[0] + s * v[2], c + (1 - c)
            * (v[1] ** 2), (1 - c) * v[2] * v[1] - s * v[0]]
    M[2] = [(1 - c) * v[0] * v[2] - s * v[1], (1 - c) * v[1]
            * v[2] + s * v[0], c + (1 - c) * (v[2] ** 2)]

    mid = np.dot(M, coord[0])
    cdef float th = np.arctan2(mid[2], np.sqrt(mid[0] ** 2 + mid[1] ** 2))
    cdef float phi = np.arctan(mid[1] / mid[0])

    if np.abs(np.cos(th) * np.cos(phi) - mid[0]) > 0.0001:
        phi += np.sign(-phi) * np.pi

    return np.array([dot1[0] + dot2[0],
                     np.degrees(th),
                     np.degrees(phi)])


@cython.boundscheck(False)
@cython.wraparound(False)
def k_means(np.ndarray[double, ndim=2, mode='c'] points not None,
            int points_number,
            int clusters_number,
            np.ndarray[long, ndim=1, mode='c'] clusterization not None):
    cdef float eps = 1e-5
    cdef int maxIterNum = 50
    nonIter = True

    if clusterization.size != points_number:
        print('Wrong clusterization size')
        return

    tmp = np.zeros((clusters_number, 2))
    mju = np.random.rand(clusters_number, 2)

    dotMax = np.max(points[:, 1:], axis=0)
    dotMin = np.min(points[:, 1:], axis=0)
    mju[:, 0] = mju[:, 0] * (dotMax[0] - dotMin[0]) + dotMin[0]
    mju[:, 1] = mju[:, 1] * (dotMax[1] - dotMin[1]) + dotMin[1]

    clusterization[:] = np.argmin(L(np.asarray(points[:, 1:], order='C'),
                                    np.asarray(mju, order='C')), axis=1)

    cdef int iterNum = 0
    cdef int cn = 0
    cdef int i = 0
    while iterNum < maxIterNum:
        for cn in set(clusterization):
            indxs = np.where(clusterization == cn)[0]
            if indxs.size == 1:
                mju[cn] = points[indxs[0], 1:]
                break
            coord = massCenter(points[indxs[0]], points[indxs[1]])
            for i in indxs[2:]:
                coord = massCenter(coord, points[i])
            mju[cn] = coord[1:]

        clusterization[:] = np.argmin(L(np.asarray(points[:, 1:], order='C'),
                                        np.asarray(mju, order='C')), axis=1)

        iterNum += 1
        if np.all(np.abs(tmp - mju) < eps) and nonIter:
            break

        tmp = mju
    #print('Iterations:', iterNum)


@cython.boundscheck(False)
@cython.wraparound(False)
def k_medoids(np.ndarray[double, ndim=2, mode='c'] points not None,
              int points_number,
              int clusters_number,
              np.ndarray[long, ndim=1, mode='c'] clusterization not None):
    cdef vector[long] clust
    clust.resize(points_number)

    k_medoids_cpp(&points[0, 0], points_number, clusters_number, clust)

    clusterization[:] = clust
