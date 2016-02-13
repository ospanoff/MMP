import numpy as np


# just running over
def first_method(x):
    a = np.array([x[0]])
    c = np.array([0])
    for i in x:
        if a[a.size - 1] != i:
            a = np.append(a, i)
            c = np.append(c, 0)
        c[a.size - 1] += 1
    return a, c


# numpy funcs
def second_method(x):
    d = np.diff(x)
    a = (np.array(np.where(d != 0)) + 1).flatten()
    if (a.size != 0):
        b = np.insert(np.diff(a), 0, a[0])
        return np.append(x[np.where(d != 0)], x[x.size - 1]), \
            np.append(b, d.size - a[a.size - 1] + 1)
    else:
        return np.array([x[0]]), np.array([x.size])


# numpy funcs + for-s
def third_method(x):
    d = np.diff(x)
    k = np.append(d, 0)
    l = np.array([x[0]], dtype=np.int32)
    for i in range(x.size):
        if k[i] != 0:
            l = np.append(l, x[i] + k[i])

    a = np.array([], dtype=np.int32)
    k = 0
    for i in range(d.size):
        if d[i] == 0:
            k += 1
        elif d[i] != 0:
            a = np.append(a, k + 1)
            k = 0
        if i == d.size - 1:
            a = np.append(a, k + 1)
    if d.size == 0:
        a = np.append(a, k + 1)
    return l, a