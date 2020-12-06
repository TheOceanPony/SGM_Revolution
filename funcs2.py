import numpy as np
from numba import njit


@njit
def H_init(imL, imR, maxD, b, c):

    height, width = imL.shape[0], imL.shape[1]
    H = np.zeros((height, width, maxD+1), dtype=np.float64)

    for i in range(0, height):
        for j in range(0, width):
            for k in range(0, maxD+1):

                H[i, j, k] = np.exp(c - b*abs(imL[i, j] - imR[i, j - k]))

    return H


@njit
def G_init(maxD, a, c):

    G = np.zeros((maxD+1, maxD + 1), dtype=np.float64)

    for k1 in range(0, maxD+1):
        for k2 in range(0, maxD + 1):
            G[k1, k2] = np.exp(c - a*abs(k1 - k2))

    return G


@njit
def P_init(H, G):

    height, width, maxD = H.shape[0], H.shape[1], H.shape[2] - 1
    P = np.zeros((height, width, maxD+1, maxD+1))

    for i in range(0, height):
        for j in range(0, width):
            for k1 in range(0, maxD + 1):
                for k2 in range(0, maxD + 1):

                    P[i, j, k1, k2] = H[i, j, k2]*G[k1, k2] # k1 <-> k2 ?

                P[i, j,  k1] = P[i, j,  k1]/(P[i, j,  k1]).sum()

    return P


@njit
def magic(i, width, maxD, F, P, Res):

    F.fill(0)
    for iteration in range(1, width):
        F[:, -iteration] = np.arange(0, maxD + 1)
        F2 = F.copy()

        for t in range(1, iteration + 1):
            for k in range(0, maxD + 1):
                # F2[k, width - t] = (F[:, width - t]*p2(k, maxD, row, width - iteration, img_L, img_R, a, b, c) ).sum()
                F2[k, width - t] = ( F[:, width - t] * P[i, width - iteration, k] ).sum()

        F = F2.copy()

    Res.fill(0)
    for k1 in range(0, maxD + 1):
        Res += (1 / (maxD + 1)) * F[k1]

    return Res


@njit
def p2(k, maxD, i, j, imL, imR, a, b, c):
    w= np.ones(maxD+1)
    for _k in range(0, maxD+1):
        w[_k] = np.exp(c - a*abs( k - _k )) * np.exp(c - b*abs(imL[i, j] - imR[i, j - _k]))

    return w/np.sum(w)
