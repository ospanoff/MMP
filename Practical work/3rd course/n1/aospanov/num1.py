import numpy as np


def first_method(x):
    mult = 1
    for i in range(np.min(x.shape)):
        if x[i, i]:
            mult *= x[i, i]
    return mult


def second_method(x):
    y = x.diagonal()
    y = y[y != 0]
    return y.prod()


def third_method(x):
    mult = 1
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if i == j:
                if x[i, j]:
                    mult *= x[i, j]
    return mult


def fourth_method(x):
    x[x == 0] = 1
    return x.diagonal().prod()
