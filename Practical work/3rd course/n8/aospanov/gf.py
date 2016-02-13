import numpy as np


def gen_pow_matrix(primpoly):
    """
    primpoly – примитивный многочлен, десятичное число, двоичная запись
        которого соответствует коэффициентам полинома над F2,
        начиная со старшей степени.

    Функция возвращает матрицу соответствия между десятичным представлением и
    степенным представлением ненулевых элементов поля по стандартному
    примитивному элементу α, numpy.array- матрица размера 2q − 1×2, в которой
    в первой колонке в позиции i стоит степень j : αj = i, а во второй
    колонке в позиции i стоит значение αi, i=1,...,2q −1.
    """
    tmp = primpoly
    q = -1
    while tmp:
        tmp >>= 1
        q += 1

    res = np.empty(((1 << q) - 1, 2), dtype=np.int)
    alpha = 2
    for i in range(res.shape[0]):
        res[alpha - 1, 0] = i + 1
        res[i, 1] = alpha
        alpha <<= 1
        if alpha >= (1 << q):
            alpha ^= primpoly

    return res


def add(X, Y):
    """
    X, Y – две матрицы одинакового размера из элементов поля Fq2,
        numpy.array-матрицы, каждый элемент в матрицах представляет собой
        десятичное число, двоичная запись которого соответствует коэффициентам
        полинома над полем F2, первый разряд соответствует старшей степени полинома;

    Функция возвращает numpy.array-матрицу размера X, являющуюся поэлементным
    суммированием матриц X и Y.
    """
    return np.bitwise_xor(X, Y)


def sum(X, axis=0):
    """
    X – матрица из элементов поля Fq2, numpy.array-матрица, каждый элемент в
        матрице представляет собой десятичное число, двоичная запись которого
        соответствует коэффициентам полинома над полем F2, первый разряд
        соответствует старшей степени полинома;

    Функция возвращает результат суммирования матрицы X по размерности,
    определяемой параметром axis.
    """
    if axis == 0:
        res = np.copy(X[0, :])
        for i in range(1, X.shape[0]):
            res ^= X[i, :]

    elif axis == 1:
        res = np.copy(X[:, 0])
        for i in range(1, X.shape[1]):
            res ^= X[:, i]

    else:
        print('Wrong axis')

    return res


def prod(X, Y, pm):
    """
    X, Y – две матрицы одинакового размера из элементов поля Fq2,
        numpy.array-матрицы, каждый элемент в матрицах представляет собой
        десятичное число, двоичная запись которого соответствует коэффициентам
        полинома над полем F2, первый разряд соответствует старшей степени полинома;
    pm – матрица соответствия между десятичным и степенным представлением в поле Fq2;

    Функции возвращают numpy.array-матрицу размера X, являющуюся соответственно
    поэлементным произведением матриц X и Y.
    """
    powers = (pm[X - 1, 0] + pm[Y - 1, 0]) % pm.shape[0]
    res = pm[powers - 1, 1]
    res[X == 0] = 0
    res[Y == 0] = 0
    return res


def divide(X, Y, pm):
    """
    X, Y – две матрицы одинакового размера из элементов поля Fq2,
        numpy.array-матрицы, каждый элемент в матрицах представляет собой
        десятичное число, двоичная запись которого соответствует коэффициентам
        полинома над полем F2, первый разряд соответствует старшей степени полинома;
    pm – матрица соответствия между десятичным и степенным представлением в поле Fq2;

    Функции возвращают numpy.array-матрицу размера X, являющуюся соответственно
    поэлементным делением матриц X и Y.
    """
    powers = (pm[X - 1, 0] - pm[Y - 1, 0]) % pm.shape[0]
    res = pm[powers - 1, 1]
    res[X == 0] = 0
    return res


def linsolve(A, b, pm):
    """
    A – квадратная матрица из элементов поля Fq2;
    b – вектор из элементов поля Fq2;
    pm – матрица соответствия между десятичным и степенным представлением в
    поле Fq2;

    Функция возвращает решение СЛАУ в случае невырожденности A и numpy.nan иначе.
    """
    X = np.hstack((A, b[:, np.newaxis]))

    for j in range(X.shape[0] - 1):
        if X[j, j] == 0:
            nz = np.nonzero(X[j:, j])[0]
            if nz.size:
                tmp = X[nz[0: 1] + j]
                X[nz[0: 1] + j] = X[j]
                X[j] = tmp
            else:
                return np.nan

        mg = np.meshgrid(X[j], X[j + 1:, j, np.newaxis])
        X[j + 1:] ^= divide(prod(mg[0], mg[1], pm),
                            np.ones(mg[0].shape, dtype=np.int) * X[j, j], pm)

    if X[-1, -2] == 0:
        return np.nan

    ans = np.zeros(X.shape[0] + 1, dtype=np.int)
    ans[-1] = 1

    for i in range(-2, -X.shape[0] - 2, -1):
        ans[i] = divide(
            sum(prod(ans, X[i + 1], pm)[np.newaxis], axis=1), X[i + 1, i], pm)

    return ans[: -1]


def minpoly(x, pm):
    """
    x – вектор из элементов поля Fq2;
    pm – матрица соответствия между десятичным и степенным представлением в поле Fq2;

    Функция осуществляет поиск минимального полинома в F2[x] для набора корней,
    задаваемых x. Функция возвращает кортеж из переменных:
        - найденный минимальный полином, numpy.array-вектор с бинарными числами;
        - все корни минимального полинома (набор корней x, а также все смежные
        с ним), numpy.array-вектор из элементов поля Fq2.
    """
    s = set()
    for i in range(x.shape[0]):
        s.add(x[i])
        tmp = prod(np.array([x[i]]), np.array([x[i]]), pm)
        while (tmp[0] != x[i]) and (tmp[0] not in s):
            s.add(tmp[0])
            tmp = prod(tmp, tmp, pm)

    roots = np.array(list(s))
    m = polyprod(np.array([1, roots[0]]), np.array([1, roots[1]]), pm)
    for i in range(2, roots.shape[0]):
        m = polyprod(m, np.array([1, roots[i]]), pm)

    return m, roots


def polyval(p, x, pm):
    """
    p – полином из Fq2[x], numpy.array-вектор коэффициентов, начиная со старшей
        степени;
    x – вектор из элементов поля Fq2;
    pm – матрица соответствия между десятичным и степенным представлением в поле Fq2;

    Функция возвращает значения полинома p для набора элементов x.
    """
    P = np.empty((x.shape[0], p.shape[0]), dtype=np.int)
    P[:, :] = p

    X = np.empty((x.shape[0], p.shape[0]), dtype=np.int)
    X[:, -1] = 1
    X[:, -2] = x
    for i in range(-3, -X.shape[1] - 1, -1):
        X[:, i] = prod(X[:, i + 1], x, pm)

    return sum(prod(P, X, pm), axis=1)


def polyprod(p1, p2, pm):
    """
    p1, p2 – полиномы из Fq2[x], numpy.array-вектор коэффициентов, начиная со
        старшей степени;
    pm – матрица соответствия между десятичным и степенным представлением в поле Fq2;

    Функция возвращает результат произведения двух полиномов в виде
    numpy.array-вектора коэффициентов, начиная со старшей степени.
    """
    p = np.zeros((p1.shape[0] - 1) + p2.shape[0], dtype=np.int)
    if p1.shape[0] > p2.shape[0]:
        ps1 = p1
        ps2 = p2
    else:
        ps1 = p2
        ps2 = p1

    tmp = np.zeros(ps1.shape[0], dtype=np.int)
    tmp[:] = ps2[-1]
    tmp = prod(tmp, ps1, pm)
    p[-tmp.shape[0]:] = tmp
    for i in range(-2, -ps2.shape[0] - 1, -1):
        tmp[:] = ps2[i]
        tmp = prod(tmp, ps1, pm)
        p[-tmp.shape[0] + i + 1: i + 1] ^= tmp

    return p


def polydivmod(p1, p2, pm):
    """
    p1, p2 – полиномы из Fq2[x], numpy.array-вектор коэффициентов, начиная со
        старшей степени;
    pm – матрица соответствия между десятичным и степенным представлением в поле Fq2;

    Функция осуществляет деление с остатком многочлена p1 на многочлен p2.
    Функция возвращает кортеж из переменных:
        - частное, numpy-array-вектор коэффициентов, начиная со старшей степени;
        - остаток от деления, numpy-array-вектор коэффициентов, начиная со
        старшей степени.
    """
    if p1.shape[0] >= p2.shape[0]:
        q = np.empty(p1.shape[0] - p2.shape[0] + 1, dtype=np.int)
        p = p1
        for i in range(q.shape[0]):
            q[i] = divide(p[0: 1], p2[0: 1], pm)
            tmp = polyprod(p2, np.array([q[i]]), pm)
            p = add(p, np.append(tmp, np.zeros(p.shape[0] - tmp.shape[0],
                                               dtype=np.int)))[1:]

        while p.size > 1 and p[0] == 0:
            p = p[1:]

    else:
        q = np.array([0], dtype=np.int)
        p = p1

    return q, p


def euclid(p1, p2, pm, max_deg=0):
    """
    p1, p2 – полиномы из Fq2[x], numpy.array-вектор коэффициентов, начиная со
        старшей степени;
    pm – матрица соответствия между десятичным и степенным представлением в поле Fq2;
    max_deg – максимально допустимая степень остатка, число, если равно нулю,
        то алгоритм Евклида работает до конца;

    Функция реализует расширенный алгоритм Евклида для пары многочленов p1 и p2.
    Функция возвращает кортеж из переменных:
        - остаток, numpy-array-вектор коэффициентов, начиная со старшей степени;
        - коэффициент при p1, numpy-array-вектор коэффициентов, начиная со
            старшей степени;
        - коэффициент при p2, numpy-array-вектор коэффициентов, начиная со
            старшей степени.
    """
    p1 = np.copy(p1)
    p2 = np.copy(p2)

    if np.nonzero(p2)[0].size == 0:
        return p1, np.array([1]), np.array([0])

    x2, x1 = np.array([1]), np.array([0])
    y2, y1 = np.array([0]), np.array([1])
    while np.nonzero(p2)[0].size > 0:
        q, r = polydivmod(p1, p2, pm)
        x, y = polyprod(q, x1, pm), polyprod(q, y1, pm)
        x[-x2.size:] = add(x[-x2.size:], x2)
        y[-y2.size:] = add(y[-y2.size:], y2)
        p1, p2 = p2, r
        x2, x1 = x1, x
        y2, y1 = y1, y
        if max_deg and r.size - 1 <= max_deg:
            return p2, x1, y1
    return p1, x2, y2
