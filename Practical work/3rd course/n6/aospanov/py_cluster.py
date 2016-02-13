import geoCoord
import numpy as np


def k_means(points, points_number, clusters_number, clusterization):
    L = geoCoord.L
    massCenter = geoCoord.massCenter
    eps = 1e-5
    maxIterNum = 50
    nonIter = True

    if clusterization.size != points_number:
        print('Wrong clusterization size')
        return

    if 'int' not in str(clusterization.dtype):
        print('Wrong clusterization type. Should be int')
        return

    tmp = np.zeros((clusters_number, 2))
    mju = np.random.rand(clusters_number, 2)

    dotMax = np.max(points[:, 1:], axis=0)
    dotMin = np.min(points[:, 1:], axis=0)
    mju[:, 0] = mju[:, 0] * (dotMax[0] - dotMin[0]) + dotMin[0]
    mju[:, 1] = mju[:, 1] * (dotMax[1] - dotMin[1]) + dotMin[1]

    clusterization[:] = np.argmin(L(points[:, 1:], mju), axis=1)

    iterNum = 0
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

        clusterization[:] = np.argmin(L(points[:, 1:], mju), axis=1)

        iterNum += 1
        if np.all(np.abs(tmp - mju) < eps) and nonIter:
            break

        tmp = mju
    # print('Iterations:', iterNum)


def k_medoids(points, points_number, clusters_number, clusterization):
    L = geoCoord.L
    eps = 1e-5
    maxIterNum = 50
    nonIter = True

    if clusterization.size != points_number:
        print('Wrong clusterization size')
        return

    if 'int' not in str(clusterization.dtype):
        print('Wrong clusterization type. Should be int')
        return

    clusterization[:] = 0
    space = points_number // clusters_number

    for i in range(clusters_number):
        clusterization[i * space: (i + 1) * space] = i
    np.random.shuffle(clusterization)

    tmp = np.zeros((clusters_number, 2))
    mju = np.zeros((clusters_number, 2))
    iterNum = 0

    while iterNum < maxIterNum:
        for cn in set(clusterization):
            indxs = np.where(clusterization == cn)[0]
            if indxs.size == 1:
                mju[cn] = points[indxs[0], 1:]
                break

            s = np.sum(L(points[indxs, 1:], points[indxs[0], np.newaxis, 1:]) *
                       points[indxs, 0])
            mid = indxs[0]
            for i in indxs[1:]:
                t = np.sum(L(points[indxs, 1:], points[i, np.newaxis, 1:]) *
                           points[indxs, 0])
                if t < s:
                    mid = i
                    s = t
            mju[cn] = points[mid, 1:]

        clusterization[:] = np.argmin(L(points[:, 1:], mju), axis=1)

        iterNum += 1
        if np.all(np.abs(tmp - mju) < eps) and nonIter:
            break

        tmp = mju
    # print('Iterations:', iterNum)
