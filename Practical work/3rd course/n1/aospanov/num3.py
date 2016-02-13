import numpy as np


# with for-s
def first_method(x, y):
    a = set(x)
    b = set(y)

    c = np.zeros(x.max().astype(np.int32) + 1, dtype=np.int32)
    if a == b and x.size == y.size:
        for i in range(x.size):
            c[x[i]] += 1
            c[y[i]] -= 1
    else:
        return False
    return np.all(c == 0)


# np funcs
def second_method(x, y):
    return np.all(np.sort(x) == np.sort(y))


# double for-s
def third_method(x, y):
    t1 = np.zeros(x.size, dtype=np.bool)
    t2 = np.zeros(x.size, dtype=np.bool)
    if x.size == y.size:
        for i in range(x.size):
            for j in range(y.size):
                if x[i] == y[j] and t1[i] == t2[j] is False:
                    t1[i] = t2[j] = True

    else:
        return False
    return np.all(t1)
