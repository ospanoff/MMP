import numpy as np


def L(dot1=np.array([[0, 0]]), dot2=np.array([[0, 0]]), R=1):
    """
    dot1: first dot on the geographical coordinates in degrees
        - should be Nx2 np.array
        - N: number of dots
    dot2: second dot on the geographical coordinates in degrees
        - should be Mx2 np.array
        - M: number of dots
    R - radius of the sphere
    """
    m = dot1.shape[0]
    n = dot2.shape[0]
    dot1 = np.repeat(np.radians(dot1), n, axis=0)
    dot2 = np.tile(np.radians(dot2).ravel(), m).reshape(-1, 2)

    return (R * np.arccos(np.sin(dot1[:, 0]) * np.sin(dot2[:, 0]) +
                          np.cos(dot1[:, 0]) * np.cos(dot2[:, 0]) *
                          np.cos(dot1[:, 1] - dot2[:, 1]))).reshape(-1, n)


def massCenter(dot1, dot2):
    """
    dot1: first dot on the geographical coordinates in degrees
        - should be 1x3 np.array
    dot2: second dot on the geographical coordinates in degrees
        - should be 1x3 np.array
    """
    m = dot1[0] / (dot1[0] + dot2[0])

    coord = np.vstack((np.radians(dot1[1:]), np.radians(dot2[1:])))

    x = np.cos(coord[:, 0]) * np.cos(coord[:, 1])
    y = np.cos(coord[:, 0]) * np.sin(coord[:, 1])
    z = np.sin(coord[:, 0])
    coord = np.vstack((x, y, z)).T

    alpha = m * np.arccos(np.dot(coord[0], coord[1]))
    v = np.array([np.linalg.det(coord[:, 1:]),
                  -np.linalg.det(coord[:, [0, 2]]),
                  np.linalg.det(coord[:, :2])])
    v /= np.linalg.norm(v)

    c = np.cos(alpha)
    s = np.sin(alpha)

    M = np.empty((3, 3))
    M[0] = [c + (1 - c) * (v[0] ** 2), (1 - c) * v[0]
            * v[1] - s * v[2], (1 - c) * v[0] * v[2] + s * v[1]]
    M[1] = [(1 - c) * v[1] * v[0] + s * v[2], c + (1 - c)
            * (v[1] ** 2), (1 - c) * v[2] * v[1] - s * v[0]]
    M[2] = [(1 - c) * v[0] * v[2] - s * v[1], (1 - c) * v[1]
            * v[2] + s * v[0], c + (1 - c) * (v[2] ** 2)]

    mid = np.dot(M, coord[0])
    th = np.arctan2(mid[2], np.sqrt(mid[0] ** 2 + mid[1] ** 2))
    phi = np.arctan(mid[1] / mid[0])

    if np.abs(np.cos(th) * np.cos(phi) - mid[0]) > 0.0001:
        phi += np.sign(-phi) * np.pi

    return np.array([dot1[0] + dot2[0],
                     np.degrees(th),
                     np.degrees(phi)])
