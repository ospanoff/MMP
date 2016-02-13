import numpy as np
import scipy.spatial.distance as ssd
import fractions as fr


# for-s with array operations
def first_method(x, y):
    d = np.zeros([x.shape[0], y.shape[0]])
    k = 0
    l = 0
    for i in x:
        for j in y:
            d[k, l] = np.sqrt((((i - j) ** 2).sum()))
            l += 1
        k += 1
        l = 0
    return d


def second_method(x, y):

    a, b = x, y

    for i in range(y.shape[0] - 1):
        a = np.hstack((a, x))

    for i in range(x.shape[0] - 1):
        b = np.vstack((b, y))

    a = a.reshape(x.shape[0] * y.shape[0], x.shape[1])

    return np.sqrt(((a - b) ** 2).sum(axis=1)).reshape(x.shape[0], y.shape[0])


# for-s by element
def third_method(x, y):
    d = np.zeros([x.shape[0], y.shape[0]])
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            a = x[i]
            b = y[j]
            sum = 0
            for k in range(a.size):
                sum += (a[k] - b[k]) ** 2
            d[i, j] = np.sqrt(sum)
    return d

# def fourth_method(x, y):
    # return ssd.cdist(x, y)
