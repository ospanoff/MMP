import numpy as np
from PIL import Image


def display_layer(X, filename='layer.png'):
    N = X.shape[0]
    I = np.int(np.sqrt(N) + 0.9999)
    J = np.int(np.sqrt(N) + 0.5)
    d = np.int(np.sqrt(X.shape[1] / 3))
    frame = np.zeros((d * I, d * J, 3), dtype=np.uint8)

    for i in range(I):
        for j in range(J):
            if (i * J + j < N):
                frame[d * i : d * (i + 1), d * j : d * (j + 1)] =\
                    X[i * J + j].reshape(d, d, 3)
    Image.fromarray(frame, 'RGB').save(filename)
