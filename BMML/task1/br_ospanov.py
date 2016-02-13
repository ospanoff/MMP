import numpy as np
import time


def expectation(type, params):
    if type == 'unif':
        return (params['a'] + params['b']) / 2

    elif type == 'bin':
        return params['n'] * params['p']

    elif type == 'poiss':
        return params['l']

    elif type == 'custom':
        return np.sum(params['p'] * params['x'])


def variance(type, params):
    if type == 'unif':
        return ((params['b'] - params['a'] + 1) ** 2 - 1) / 12

    elif type == 'bin':
        return params['n'] * params['p'] * (1 - params['p'])

    elif type == 'poiss':
        return params['l']

    elif type == 'custom':
        return expectation('custom', {'x': params['x'] ** 2, 'p': params['p']}) -\
            expectation('custom', {'x': params['x'], 'p': params['p']}) ** 2


def generate(N, a, b, params, model):
    if model == 4:
        c = np.random.poisson(a * params['p1'] + b * params['p2'], N)
    elif model == 3:
        c = np.random.binomial(
            a, params['p1'], N) + np.random.binomial(b, params['p2'], N)

    return c + np.random.binomial(c, params['p3'])


def pa(params):
    a = np.arange(params['amin'], params['amax'] + 1)
    p = 1 / (params['amax'] - params['amin'] + 1) * np.ones(a.size)
    return p, a


def pb(params):
    b = np.arange(params['bmin'], params['bmax'] + 1)
    p = 1 / (params['bmax'] - params['bmin'] + 1) * np.ones(b.size)
    return p, b


def binomial(a, p):
    n = a
    carrier = np.arange(0, a + 1, dtype=np.int)
    pdf = np.copy(carrier.astype(np.float))
    pdf[1:] = (p / (1 - p)) * (n - pdf[1:] + 1) / pdf[1:]
    pdf[0] = (1 - p) ** n
    return np.cumprod(pdf), carrier


def poisson(a, l):
    carrier = np.arange(0, a + 1, dtype=np.int)
    pdf = np.copy(carrier.astype(np.float))
    pdf[1:] = l / pdf[1:]
    pdf[0] = np.exp(-l)
    return np.cumprod(pdf), carrier


def pc_ab(a, b, params, model):
    # model 4: Poiss(ap1 + bp2)
    # model 3: Bin(a, p1) + Bin(b, p2)
    carrier = np.arange(0, params['amax'] + params['bmax'] + 1, dtype=np.int)
    pdf = np.zeros_like(carrier, dtype=np.float)
    if model == 3:
        pdf[0: a + b + 1] = np.convolve(binomial(a, params['p1'])[0],
                                        binomial(b, params['p2'])[0])
    elif model == 4:
        pdf[0: a + b +
            1] = poisson(a + b, a * params['p1'] + b * params['p2'])[0]
    return pdf, carrier


def pd_c(c, params):
    carrier = np.arange(
        0, 2 * (params['amax'] + params['bmax']) + 1, dtype=np.int)
    pdf = np.zeros_like(carrier, dtype=np.float)
    pdf[c: 2 * c + 1] = binomial(c, params['p3'])[0]
    return pdf, carrier


def pc(params, model):
    carrier = np.arange(
        0, params['amax'] + params['bmax'] + 1, dtype=np.int)
    pdf = np.zeros_like(carrier, dtype=np.float)
    for a in range(params['amin'], params['amax'] + 1):
        for b in range(params['bmin'], params['bmax'] + 1):
            pdf += pc_ab(a, b, params, model)[0]
    return pdf * pa(params)[0][0] * pb(params)[0][0], carrier


def pd(params, model):
    carrier = np.arange(
        0, 2 * (params['amax'] + params['bmax']) + 1, dtype=np.int)
    pdf = np.zeros_like(carrier, dtype=np.float)
    p = pc(params, model)[0]
    for c in range(params['amax'] + params['bmax'] + 1):
        pdf += pd_c(c, params)[0] * p[c]
    return pdf, carrier


def pd_ab(a, b, params, model):
    # model 4: Poiss(ap1 + bp2)
    # model 3: Bin(a, p1) + Bin(b, p2)
    carrier = np.arange(
        0, 2 * (params['amax'] + params['bmax']) + 1, dtype=np.int)

    pc = pc_ab(a, b, params, model)
    pdf = np.zeros((pc[0].size, carrier.size), dtype=np.float)
    for c in pc[1]:
        pdf[c] = pd_c(c, params)[0] * pc[0][c]

    return pdf.sum(axis=0), carrier


def pb_d(d, params, model):
    """
    Input:
        d = np.array(size=N)
    Output:
        np.array([d(b|d1), d(b|d1, d2), ..., d(b|d1, ..., dN)])
    """
    carrier = np.arange(params['bmin'], params['bmax'] + 1)
    pdf = np.zeros((carrier.size, d.size), dtype=np.float)

    pb = np.zeros(d.size)
    for b in carrier:
        summ = np.zeros(d.size)
        for a in range(params['amin'], params['amax'] + 1):
            tmp = pd_ab(a, b, params, model)[0]
            summ += np.array([np.prod(tmp[d[: i]])
                              for i in range(1, d.size + 1)])

        pb += summ
        pdf[b - carrier[0]] = summ

    return pdf / pb, carrier


def pb_ad(a, d, params, model):
    """
    Input:
        a - scalar, d = np.array(size=N)
    Output:
        np.array([d(b|a, d1), d(b|a, d1, d2), ..., d(b|a, d1, ..., dN)])
    """
    carrier = np.arange(params['bmin'], params['bmax'] + 1)
    pdf = np.zeros((carrier.size, d.size), dtype=np.float)

    pb = np.zeros(d.size)
    for b in carrier:
        summ = np.zeros(d.size)
        tmp = pd_ab(a, b, params, model)[0]
        summ += np.array([np.prod(tmp[d[: i]]) for i in range(1, d.size + 1)])

        pb += summ
        pdf[b - carrier[0]] = summ

    return pdf / pb, carrier
