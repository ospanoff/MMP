import numpy as np
import gradient


def initialize(hidden_size, visible_size):
    mean = 0
    bWeight = 0
    neuronLayerSize = np.append(hidden_size, visible_size)
    connectionsNum = np.append(visible_size, hidden_size) * neuronLayerSize
    wb = np.ones(np.sum(neuronLayerSize) + np.sum(connectionsNum)) * bWeight

    k = 0
    for i in range(connectionsNum.size):
        sigma = np.sqrt(6 / (neuronLayerSize[i - 1] + neuronLayerSize[i] + 1))
        wb[k: k + connectionsNum[i]] =\
            np.random.normal(mean, sigma, connectionsNum[i])
        k += connectionsNum[i] + neuronLayerSize[i]
    return wb


def autoencoder_loss(data, theta, visible_size, hidden_size,
                     lambda_=0.0001, sparsity_param=0.01, beta=3, onlyJ=False):
    neuronLayerSize = np.append(hidden_size, visible_size)
    connectionsNum = np.append(visible_size, hidden_size) * neuronLayerSize
    onlyWIndx = np.zeros(theta.size, dtype=np.bool)

    # ForwardPropagation

    k = 0
    m = 0
    # activate function values
    act = np.empty((data.shape[0], np.sum(neuronLayerSize)))

    for i in range(neuronLayerSize.size):
        right = k + connectionsNum[i]
        z = np.dot(act[:, m - neuronLayerSize[i - 1]: m] if i != 0 else data,
                   theta[k: right].reshape(neuronLayerSize[i - 1], -1))
        onlyWIndx[k: right] = True
        k += connectionsNum[i] + neuronLayerSize[i]
        z += theta[right: k]
        act[:, m: m + neuronLayerSize[i]] = 1 / (1 + np.exp(-z))
        m += neuronLayerSize[i]

    # Sparsity

    ro_ = np.average(act, axis=0)

    firstArg = sparsity_param / ro_
    secondArg = (1 - sparsity_param) / (1 - ro_)

    KL = np.sum(sparsity_param * np.log(firstArg[: -visible_size]) +
                (1 - sparsity_param) * np.log(secondArg[: -visible_size]))

    # LossFunction

    J = np.average(0.5 *
                   (np.linalg.norm(act[:, -visible_size:] - data, axis=1) ** 2))
    J += 0.5 * lambda_ * np.sum(theta[onlyWIndx] ** 2) + beta * KL

    if onlyJ:
        return J

    # BackPropagation

    grad = np.ones(theta.size)
    deltaPrev = -(data - act[:, -visible_size:]) * \
        act[:, -visible_size:] * (1 - act[:, -visible_size:])

    wLeft = -connectionsNum[-1] - neuronLayerSize[-1]
    wRight = wLeft + connectionsNum[-1]
    dRight = -neuronLayerSize[-1]
    dLeft = -neuronLayerSize[-1] - neuronLayerSize[-2]

    for i in range(hidden_size.size):
        l = hidden_size.size - i - 1

        grad[wLeft: wRight] = (np.dot(deltaPrev.T, act[:, dLeft: dRight]).T /
                               data.shape[0]).ravel() + lambda_ * theta[wLeft: wRight]
        grad[wRight: wRight + neuronLayerSize[l + 1] if i != 0 else None] = \
            np.average(deltaPrev, axis=0)

        deltaPrev = (np.dot(deltaPrev, theta[wLeft: wRight].reshape(neuronLayerSize[l], -1).T) +
                     beta * (secondArg[dLeft: dRight] - firstArg[dLeft: dRight])) * \
            act[:, dLeft: dRight] * (1 - act[:, dLeft: dRight])

        wLeft -= connectionsNum[l] + neuronLayerSize[l]
        wRight = wLeft + connectionsNum[l]
        dRight = dLeft
        dLeft -= neuronLayerSize[l - 1]

    grad[wLeft: wRight] = (np.dot(deltaPrev.T, data).T / data.shape[0]).ravel() + \
        lambda_ * theta[wLeft: wRight]
    grad[wRight: wRight + neuronLayerSize[0]] = np.average(deltaPrev, axis=0)

    return J, grad


def autoencoder_transform(theta, visible_size, hidden_size, layer_number, data):
    if layer_number == 1:
        return data

    neuronLayerSize = np.append(hidden_size, visible_size)
    connectionsNum = np.append(visible_size, hidden_size) * neuronLayerSize

    # ForwardPropagation

    k = 0
    m = 0

    # activate function values
    act = np.empty((data.shape[0], np.sum(neuronLayerSize)))

    for i in range(neuronLayerSize.size):
        right = k + connectionsNum[i]
        z = np.dot(act[:, m - neuronLayerSize[i - 1]: m] if i != 0 else data,
                   theta[k: right].reshape(neuronLayerSize[i - 1], -1))
        k += connectionsNum[i] + neuronLayerSize[i]
        z += theta[right: k]
        act[:, m: m + neuronLayerSize[i]] = 1 / (1 + np.exp(-z))
        if i + 2 == layer_number:  # i = 0 - second layer
            return act[:, m: m + neuronLayerSize[i]]
        m += neuronLayerSize[i]

    return data