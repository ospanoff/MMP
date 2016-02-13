#!/usr/bin/env python

import fast
import slow
import random
import time
import numpy as np


# method for running and testing some algo
def run_algo(method, name, px, py):
    template = 'Algorithm "%s":\n\tResult:       %.5g\n\tElapsed time: %.5g\n'
    start_time = time.time()
    result = method(px, py)
    print(template % (name, result, time.time() - start_time))


# generate random test data
n = 5000
px = []
py = []
for i in range(n):
    px.append(random.random())
    py.append(random.random())

# we can pass to cython np.array type but to be really fast
# we have to specify concrete type of internal data
npx = np.zeros(n, dtype=np.float32)
npy = np.zeros(n, dtype=np.float32)
for i in range(n):
    npx[i], npy[i] = px[i], py[i]


# run algos with vectors and malloc
run_algo(fast.naive_with_cpp, 'NaiveWithCpp', px, py)
run_algo(fast.naive_with_malloc, 'NaiveWithMalloc', px, py)
run_algo(fast.naive_with_vector, 'NaiveWithVector', px, py)

# numpy versions
run_algo(fast.naive_with_np, 'NaiveWithNumPy', npx, npy)
# pass the real numpy.array pointers to the cpp code through cython wrapper method
run_algo(fast.naive_with_np_cpp, 'NaiveWithNumPyCPP', npx, npy)

# really naive algos
run_algo(fast.naive, 'NaiveCythonized', px, py)
run_algo(slow.naive, 'NaiveClear', px, py)

