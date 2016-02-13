import numpy as np
import pandas
import gf


primpoly = np.array([7, 19, 37, 67, 131, 285, 529, 1033,
                     2053, 4179, 8219, 16427, 32771, 65581])


def coding(U, g):
    """
    U – набор исходных сообщений для кодирования, numpy.array-матрица, бинарная
        матрица размера <число_сообщений> × k;
    g – порождающий многочлен кода, numpy.array-вектор длины m + 1;

    Функция осуществляет систематическое кодирование циклического кода и
    возвращает numpy.array-матрицу с закодированными сообщениями размера
    <число_сообщений> × (k + m).
    """
    pm = np.array([[1, 1]], dtype=np.int)
    xm = np.zeros(g.shape[0], dtype=np.int)
    xm[0] = 1

    V = np.empty((U.shape[0], U.shape[1] + xm.shape[0] - 1), dtype=np.int)
    for i in range(U.shape[0]):
        V[i] = gf.polyprod(xm, U[i], pm)
        r = gf.polydivmod(V[i], g, pm)[1]
        V[i][-r.size:] = gf.add(V[i][-r.size:], r)

    return V


def dist(g, n):
    """
    g – порождающий многочлен кода, numpy.array-вектор;
    n – длина кода, число;

    Функция возвращает кодовое расстояние (число), найденное полным перебором.
    """
    k = n - g.shape[0] + 1
    U = np.zeros((2 ** k - 1, k), dtype=np.int)

    for i in range(U.shape[0]):
        num = np.array(list(bin(i + 1)[2:]), dtype=np.int)
        U[i][-num.size:] = num

    U = coding(U, g)
    dist = U.shape[1]
    for i in range(U.shape[0] - 1):
        for j in range(i + 1, U.shape[0]):
            tmp = np.sum(np.bitwise_or(U[i], U[j]))
            if tmp < dist:
                dist = tmp

    return dist


def genpoly(n, t):
    """
    n – длина кода, число;
    t – исправляемое число ошибок, число;

    Функция строит порождающий многочлен БЧХ-кода по заданным параметрам.
    Функция возвращает кортеж из переменных:
        - порождающий многочлен кода, numpy.array-вектор коэффициентов, начиная
            со старшей степени;
        - нули кода, numpy.array-вектор десятичных чисел, соответствующих
            элементам из Fq2;
        - матрица соответствия между десятичным и степенным представлением в
            поле Fq2. 
    """
    if n >= 2 ** 16:
        print('q > 16')
        return

    pm = gf.gen_pow_matrix(primpoly[primpoly >= n][0])

    x = np.ones(2 * t, dtype=np.int) * 2
    for i in range(1, x.shape[0]):
        x[i:] = gf.prod(x[i:], np.tile(x[0], x.shape[0] - i), pm)

    return gf.minpoly(x, pm)[0], x, pm


def decoding(W, R, pm, method='euclid'):
    """
    W – набор принятых сообщений, numpy.array-матрица размера <число_сообщений> × n
    R – нули кода, numpy.array-вектор
    pm – матрица соответствия между десятичным и степенным представлением в поле Fq2
    method – алгоритм декодирования, ’euclid’ или ’pgz’

    Функция осуществляет декодирование БЧХ кода и возвращает numpy.array-матрицу
    с декодированными сообщениями размера <число_сообщений> × n. В случае отказа
    от декодирования соответствующая строка матрицы состоит из numpy.nan
    """
    U = np.empty_like(W)
    for w in range(W.shape[0]):
        s = gf.polyval(W[w], R, pm)

        if np.sum(s) == 0:
            U[w] = W[w]
            continue

        U[w] = W[w]

        if method == 'pgz':
            nju = 0
            for i in range(R.size // 2, 0, -1):
                A = np.empty((i, i), dtype=np.int)
                for j in range(A.shape[0]):
                    A[j] = s[j: j + i]
                b = s[i: 2 * i]
                solve = gf.linsolve(A, b, pm)
                if solve is np.nan:
                    continue
                else:
                    nju = i
                    L = np.append(solve, 1)
                    break

            if nju == 0:
                U[w] = np.nan

            idx = np.where(gf.polyval(L, pm[:, 1], pm) == 0)[0]
            U[w, idx] ^= 1

            ss = gf.polyval(U[w], R, pm)

            if np.sum(ss) != 0:
                U[w] = np.nan

        elif method == 'euclid':
            S = np.append(s[:: -1], 1)
            z = np.zeros(R.size + 2, dtype=np.int)
            z[0] = 1
            r, A, L = gf.euclid(z, S, pm, max_deg=R.size // 2)

            idx = np.where(gf.polyval(L, pm[:, 1], pm) == 0)[0]
            U[w, idx] ^= 1

            if idx.size != L.size - 1:
                U[w] = np.nan

    return U
