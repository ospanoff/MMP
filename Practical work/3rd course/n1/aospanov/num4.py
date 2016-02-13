import numpy as np


# with for-s and tmp array
def first_method(x):
    v = np.zeros(x.size, dtype=x.dtype)
    for i in range(x.size):
        if i < x.size - 1 and x[i] == 0:
            v[i + 1] = x[i + 1]
    return v.max()


# numpy funcs
def second_method(x):
    y = np.array(np.where(x == 0)) + 1
    if y.any():
        return x[y[y < x.size]].max()
    else:
        return 0


# with for-s and tmp var
def third_method(x):
    # k = x.dtype
    k = x[0]
    for i in range(x.size - 1):
        if x[i] == 0:
            k = x[i + 1]
            for j in range(i, x.size - 1):
                if x[j] == 0 and x[j + 1] > k:
                    k = x[j + 1]
            break
    if (k != x[0]):
        return k
    else:
        return 0
