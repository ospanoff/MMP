import numpy as np


def compute_gradient(J, theta):
    # works only for theta.shape = (N, ) or (N, 1)
    if np.size(theta.shape) == 2 and theta.shape[1] != 1:
        print('Wrong shape!')
        return None

    t = 10e-4
    e = np.zeros(theta.shape[0]) #np.eye(theta.shape[0])
    grad = np.empty(theta.shape)
    for i in range(e.shape[0]):
        e[i] = 1
        grad[i] = (J(theta + t * e.reshape(theta.shape)) -
                   J(theta - t * e.reshape(theta.shape))) / (2 * t)
        e[i] = 0
    return grad


def check_gradient():
    def f(x):
        return np.sum(1 / (1 + np.exp(-x)))

    def f_gr(x):
        return np.exp(-x) / (1 + np.exp(-x)) ** 2

    testSize = (10, 100)
    x = np.random.random(testSize)
    for t in x:
        if np.allclose(compute_gradient(f, t), f_gr(t)) == False:
            print('Wrong gradient')
            return
    print('All alright!')
