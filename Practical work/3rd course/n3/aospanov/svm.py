import numpy as np
import cvxopt
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
import time
import sklearn
from sklearn import svm


def genData(numOfClasses, numOfElemInClass, D, coef_=10):
    mean = [i + np.random.random(D) * coef_ for i in range(numOfClasses)]
    cov = np.array(
        [np.diag(np.random.random(D)) for i in range(numOfClasses)])

    X = np.empty((numOfClasses * numOfElemInClass, D))
    for i in range(numOfClasses):
        X[i * numOfElemInClass : (i + 1) * numOfElemInClass, :] =\
            np.random.multivariate_normal(mean[i], cov[i], numOfElemInClass)

    # Adding 1
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    y = np.ones((numOfClasses * numOfElemInClass, 1), dtype=np.int32)
    if (numOfClasses == 2):
        y[:numOfElemInClass] = -1
    else:
        for i in range(numOfClasses):
            y[i * numOfElemInClass: (i + 1) * numOfElemInClass] = i
    return X, y


def sqrEuclidMetric(x, y):
    yn, xn = np.meshgrid((y ** 2).sum(axis=1), (x ** 2).sum(axis=1))
    return xn + yn - 2 * x.dot(y.T)


def compute_primal_objective(X, y, w, C):
    w = w[:X.shape[1]]
    ret = 0.5 * np.dot(w.T, w) + C * np.sum(
        [np.maximum(0, 1 - y[i] * np.dot(w.T, X[i])) for i in range(X.shape[0])])
    return ret[0, 0]


def compute_primal_objective_ksi(X, y, w, ksi, C):
    return 0.5 * np.dot(w.T, w) + C * np.sum(ksi)[0, 0]


def compute_dual_objective(X, y, A, C, gamma=0):
    if gamma < 0:
        print("Error: Gamma is negative!")
        return None

    if not gamma:
        nX = np.dot(X, X.T)
    else:
        nX = np.exp(-gamma * sqrEuclidMetric(X, X))
    Ay = A * y

    return (np.sum(A) - 0.5 * np.dot(np.dot(Ay.T, nX), Ay))[0][0]


def subgradient(x, y, w):
    if (y * np.dot(w.T, x) < 1):
        ret = -y * x
    else:
        ret = np.zeros(x.shape)
    return ret[:, np.newaxis]


def subdiff(X, y, w, C):
    return w + C * np.sum([subgradient(X[i], y[i], w) for i in range(X.shape[0])], axis=0)


def svm_subgradient_solver(X, y, C, tol=1e-2, max_iter=2000, alpha=2, beta=1,
                           stochastic=False, subSampleSize=0.1, verbose=False):
    _time = time.clock()

    w = np.ones((X.shape[1], 1))
    obj_curve = np.zeros(max_iter)
    t = 0
    obj_curve[t] = compute_primal_objective(X, y, w, C)
    status = 0
    nw = w - alpha * subdiff(X, y, w, C)
    if not stochastic:
        while np.abs(obj_curve[t] - compute_primal_objective(X, y, nw, C)) > tol or\
                np.sqrt(np.dot(nw.T, nw)) - np.sqrt(np.dot(w.T, w)) > tol:
            t += 1
            if (t >= max_iter):
                status = 1
                break
            w = nw
            obj_curve[t] = compute_primal_objective(X, y, w, C)
            nw = w - alpha / (t ** beta) * subdiff(X, y, w, C)
    else:
        while np.abs(obj_curve[t] - compute_primal_objective(X, y, nw, C)) > tol or\
                np.sqrt(np.dot(nw.T, nw)) - np.sqrt(np.dot(w.T, w)) > tol:
            t += 1
            if (t >= max_iter):
                status = 1
                break
            w = nw
            obj_curve[t] = compute_primal_objective(X, y, w, C)
            indx = np.random.choice(
                X.shape[0], subSampleSize * X.shape[0], replace=False)
            nw = w - alpha / (t ** beta) * subdiff(X[indx], y[indx], w, C)

    _time = time.clock() - _time

    return {
        'w': nw,
        'A': None,
        'status': status,
        'objective_curve': obj_curve[:t],
        'time': _time
    }


def svm_qp_primal_solver(X, y, C, tol=1e-6, max_iter=100, verbose=False):
    cvxopt.solvers.options['show_progress'] = verbose
    cvxopt.solvers.options['maxiters'] = max_iter
    cvxopt.solvers.options['reltol'] = tol
    _time = time.clock()

    # Adding 1
    # X = np.hstack((X, np.ones((X.shape[0], 1))))

    # Gen P
    P = np.eye(X.shape[0] + X.shape[1])
    P[X.shape[1]:, X.shape[1]:] = 0

    # Gen q
    q = np.zeros(X.shape[0] + X.shape[1])
    q[X.shape[1]:] = C

    # Gen G
    G = np.zeros((2 * X.shape[0], X.shape[1] + X.shape[0]))
    G[:X.shape[0], X.shape[1]:] = np.eye(X.shape[0]) * -1
    G[X.shape[0]:, X.shape[1]:] = np.eye(X.shape[0]) * -1
    G[:X.shape[0], :X.shape[1]] = -y * X

    # Gen h
    h = np.zeros(2 * X.shape[0])
    h[: X.shape[0]] = -1

    # convert to matrix
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)

    # solve
    solution = cvxopt.solvers.qp(P, q, G, h)

    status = 1
    if solution['status'] is 'optimal':
        status = 0

    _time = time.clock() - _time
    return {
        'w': solution['x'][:X.shape[1]],
        'A': None,
        'status': status,
        'objective_curve': None,
        'time': _time
    }


def svm_qp_dual_solver(X, y, C, tol=1e-6, max_iter=100, verbose=False, gamma=0):
    cvxopt.solvers.options['show_progress'] = verbose
    cvxopt.solvers.options['maxiters'] = max_iter
    cvxopt.solvers.options['reltol'] = tol
    _time = time.clock()

    if gamma < 0:
        print("Error: Gamma is negative!")
        return None

    # Gen P
    if not gamma:
        K = np.dot(X, X.T)
    else:
        K = np.exp(-gamma * sqrEuclidMetric(X, X))

    mgr = np.meshgrid(y, y)
    P = K * mgr[0] * mgr[1]

    # Gen q
    q = -1 * np.ones((X.shape[0], 1))

    # Gen G
    G = np.zeros((2 * X.shape[0], X.shape[0]))
    G[:X.shape[0], :] = np.eye(X.shape[0])
    G[X.shape[0]:, :] = -1 * np.eye(X.shape[0])

    # Gen h
    h = np.zeros((2 * X.shape[0], 1))
    h[:X.shape[0]] = C

    # Gen A
    A = y.T.astype(np.double)

    # Gen b
    b = np.zeros((1, 1))

    # convert to matrix
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(b)

    # solve
    solution = cvxopt.solvers.qp(P, q, G, h) #, A, b)

    status = 1
    if solution['status'] is 'optimal':
        status = 0

    w = compute_w(X, y, solution['x'])

    _time = time.clock() - _time

    return {
        'w': w,
        'A': np.array(solution['x']),
        'status': status,
        'objective_curve': None,
        'time': _time
    }


def svm_liblinear_solver(X, y, C, tol=1e-6, verbose=False):
    _time = time.clock()

    libl = sklearn.svm.LinearSVC(tol=tol, C=C, loss='l1')
    libl.fit(X, y.ravel())

    _time = time.clock() - _time
    return {
        'w': libl.raw_coef_.T,
        'A': None,
        'status': 0,
        'objective_curve': None,
        'time': _time
    }


def svm_libsvm_solver(X, y, C, tol=1e-6, max_iter=500, verbose=False, gamma=0):
    _time = time.clock()

    if not gamma:
        kern = 'linear'
    else:
        kern = 'rbf'

    svc = sklearn.svm.SVC(C=C, kernel=kern, gamma=gamma,
                          tol=tol, max_iter=max_iter)
    svc.fit(X, y.ravel())

    A = np.zeros((X.shape[0], 1))
    A[svc.support_] = np.abs(svc.dual_coef_[0][:, np.newaxis])

    w = compute_w(X, y, A)

    _time = time.clock() - _time
    return {
        'w': np.vstack((w, svc.intercept_)),
        'A': A,
        'status': 0,
        'objective_curve': None,
        'time': _time
    }


def compute_support_vectors(X, y, A):
    indx = np.where(A > 0.001)[0]
    return (X[indx], y[indx])


def compute_w(X, y, A):
    return np.sum(y * A * X, axis=0).reshape(X.shape[1], 1)


def RLE(x):
    # run length encoding
    d = np.diff(x)
    a = (np.array(np.where(d != 0)) + 1).flatten()
    if (a.size != 0):
        b = np.insert(np.diff(a), 0, a[0])
        return np.append(x[np.where(d != 0)], x[x.size - 1]), \
            np.append(b, d.size - a[a.size - 1] + 1)
    else:
        return np.array([x[0]]), np.array([x.size])


def predict_ovo(X, y, A, pred_x, gamma=0):
    # y = {-1, 1}
    w = compute_w(X, y, A)
    b = w[2]
    X = X[:, :pred_x.shape[1]]
    if not gamma:
        pred_x = pred_x.T
        supVec = compute_support_vectors(X, y, A)
        w = compute_w(X, y, A)
        tmp = supVec[1] - np.dot(supVec[0], w)
        b = np.median(tmp)
        rr = np.sum(y * A * np.dot(X, pred_x), axis=0) + b
        return np.sign(rr).astype(np.int32)
    else:
        ex = np.exp(-gamma * sqrEuclidMetric(X, pred_x))
        rr = np.sum(y * A * ex, axis=0) + b
        return np.sign(rr).astype(np.int32)


def predict_multi_ovo(X, y, A, pred_x, gamma=0):
    # y = {0, 1, ..., k}
    ys = np.array([i for i in set(y.ravel())])
    ret = np.empty((A.shape[1], pred_x.shape[0]))
    k = 0
    for i in range(ys.size):
        for j in range(i + 1, ys.size):
            id1 = np.where(y == ys[i])[0]
            indx = np.append(id1, np.where(y == ys[j])[0])
            ny = np.ones((indx.size, 1))
            ny[:id1.size] = -1
            pr = predict_ovo(X[indx], ny, A.T[k, :, np.newaxis], pred_x, gamma)
            ret[k] = i
            ret[k][np.where(pr == 1)[0]] = j
            k += 1
    for i in range(ret.shape[1]):
        tmp = RLE(np.sort(ret[:, i]))
        ret[0][i] = tmp[0][np.argmax(tmp[1])]
    return ret[0][:, np.newaxis].astype(np.int32)


def visualize(X, y, w=None, A=None, gamma=0):
    axX = X[:, 0]
    axY = X[:, 1]
    plt.rcParams['figure.figsize'] = (15, 10)
    if not (A is None):
        supVec = compute_support_vectors(X[:, :2], y, A)

        xs, ys = np.meshgrid(np.arange(min(axX) - 1, max(axX) + 1, 0.05),
                             np.arange(min(axY) - 1, max(axY) + 1, 0.05))
        labels = np.empty(xs.shape).astype(np.int32)
        for i in range(xs.shape[0]):
            labels[i] = predict_multi_ovo(X, y, A, np.array([[xs[i, j], ys[i, j]] for j in range(xs.shape[1])]), gamma).ravel()

        plt.pcolormesh(xs, ys, labels,
                       cmap=mplcol.ListedColormap(['#ff6666', '#ffb366', '#c1ab87', '#bfefff', '#d5f45d', '#8a807c']))
        plt.scatter(supVec[0][:, 0], supVec[0][:, 1], s=70,
                    c=[0 for i in supVec[1]], cmap=mplcol.ListedColormap(['#ffffff']))
    else:
        for nw in w:
            xs = np.array([min(axX), max(axX)])
            ys = (-nw[0] * xs - nw[2]) / nw[1]
            plt.plot(xs, ys)

    plt.scatter(axX, axY, c=y.astype(np.int32), cmap=mplcol.ListedColormap(
        ['#ff00ff', '#00ffff', '#7f4903', '#4876ff', '#ff0048', '#336600']))
