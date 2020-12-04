import numpy as np
from numba import njit

@njit
def init_unary(maxD, imL, imR, c):

    height, width = imL.shape
    H = np.zeros((height, width, maxD + 1), dtype=np.float32)

    print("H initialising...")
    for j in range(0, height):
        for i in range(0, width):
            for d in range(0, maxD + 1):
                H[j, i, d] = np.exp(c - abs(imL[j, i] - imR[j, i - d]))
    return H


@njit
def init_binary(maxD, alpha, c):

    G = np.zeros((maxD + 1, maxD + 1), dtype=np.float32)

    for di in range(0, maxD + 1):
        for dj in range(0, maxD + 1):
            G[di, dj] = np.exp(c - alpha * abs(di - dj))

    return G


@njit
def init_left(H, G):

    height, width, maxD  = H.shape[0], H.shape[1], H.shape[2] - 1
    Li = np.zeros((height, width, maxD + 1, width))

    print(f"Li initialising...")
    for i in range(0, height):
        for j in range(0, width):
            for d in range(0, maxD):

                Li[i, j, d] = left(i, j ,d, Li, H, G, Li[i, j, d])

    return Li

@njit
def left(i, j, d, Li, H, G, res):

    height, width, maxD = H.shape[0], H.shape[1], H.shape[2] - 1

    if j == 0:
        return np.zeros(width)
    else:
        temp = np.zeros(width)

        for _d in range(0, maxD+1):

            temp[j-1] = _d
            res += H[i, j, _d]*G[d, _d]*temp + Li[i, j-1, _d]

        return res


@njit
def init_right(H, G):
    height, width, maxD = H.shape[0], H.shape[1], H.shape[2] - 1
    Ri = np.zeros((height, width, maxD + 1, width))

    print(f"Ri initialising...")
    for i in range(0, height):
        for j in range(0, width):
            for d in range(0, maxD):
                Ri[i, j, d] = right(i, j, d, Ri, H, G, Ri[i, j, d])

    return Ri

@njit
def right(i, j, d, Ri, H, G, res):
    height, width, maxD = H.shape[0], H.shape[1], H.shape[2] - 1

    if j == width:
        return np.zeros(width)
    else:
        temp = np.zeros(width)

        for _d in range(0, maxD + 1):
            temp[j + 1] = _d
            res += H[i, j, _d] * G[d, _d] * temp + Ri[i, j + 1, _d]

        return res


