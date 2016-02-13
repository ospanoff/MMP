import numpy as np
import scipy.stats as ss


# for-s with array operations
def first_method(x, m, C):
    return ss.multivariate_normal(m, C).logpdf(x)


def second_method(x, m, C):
    norm_factor = 1.0 / np.sqrt(((2 * np.pi) ** m.size) * np.linalg.det(C))
    nx = x - m
    a = nx.dot(np.linalg.inv(C)).dot(nx.T)
    return np.log(norm_factor * np.exp(-1 / 2 * a)).diagonal()


def third_method(x, m, C):
    norm_factor = 1
    for i in range(m.size):
        norm_factor *= (2 * np.pi)
    norm_factor *= np.linalg.det(C)
    norm_factor = 1 / np.sqrt(norm_factor)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] -= m[j]

    inv = np.linalg.inv(C)
    a = x.dot(inv)
    a = a.dot(x.T)
    res = norm_factor * np.power(2.7182818284, -1 / 2 * a)
    return np.log(res).diagonal()
