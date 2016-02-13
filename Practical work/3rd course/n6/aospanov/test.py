import numpy as np
import time
import pandas

import py_cluster
import cy_cluster

N = 3000

data = np.empty((N, 3))

data[:, 0] = np.random.randint(1000, 100000, N)
data[:, 1] = np.random.rand(N) * 140 - 70
data[:, 2] = np.random.rand(N) * 350 - 175


clusters_number = 10
clusterization = np.empty(N, dtype=np.int)

print('Clusters number:', clusters_number, '\n')

_time = time.clock()
py_cluster.k_means(data, N, clusters_number, clusterization)
print('Python k-Means:\n\tTime elapsed:', time.clock() - _time)
print('\tClusters:', len(set(clusterization)), '\n')

_time = time.clock()
py_cluster.k_medoids(data, N, clusters_number, clusterization)
print('Python k-Medoids:\n\tTime elapsed:', time.clock() - _time)
print('\tClusters:', len(set(clusterization)), '\n')

_time = time.clock()
cy_cluster.k_means(data, N, clusters_number, clusterization)
print('Cython k-Means:\n\tTime elapsed:', time.clock() - _time)
print('\tClusters:', len(set(clusterization)), '\n')

_time = time.clock()
cy_cluster.k_medoids(data, N, clusters_number, clusterization)
print('C++ k-Medoids:\n\tTime elapsed:', time.clock() - _time)
print('\tClusters:', len(set(clusterization)), '\n')
