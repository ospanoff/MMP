import numpy as np


# single for
def first_method(x, a, b):
    v = np.empty([a.size, ])
    for i in range(a.size):
        v[i] = x[a[i], b[i]]
    return v


# list comprehension
def second_method(x, a, b):
    v = np.array([x[a[i], b[i]] for i in range(a.size)])
    return v


# indexing
def third_method(x, a, b):
    return x[a, b]