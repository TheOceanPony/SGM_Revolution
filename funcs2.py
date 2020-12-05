import numpy as np
from numba import njit


#@njit
def magic(img_L, img_R, row, MAX_DISP):

    width = img_L.shape[1]

    F = np.zeros((MAX_DISP + 1, width))
    for iteration in range(1, width):
        F[:, -iteration] = np.arange(0, MAX_DISP + 1)
        F2 = F.copy()

        for t in range(1, iteration + 1):
            for k in range(0, MAX_DISP + 1):
                F2[k, width - t] = np.average(F[:, width - t], weights=p2(k, MAX_DISP, row, width - iteration, img_L, img_R))

        F = F2.copy()

    res = np.zeros(width)
    for k1 in range(0, MAX_DISP + 1):
        res += (1 / (MAX_DISP + 1)) * F[k1]

    return res


@njit
def p2(k, maxD, i, j, imL, imR, alpha = 0.1, c=0):
    w= np.ones(maxD+1)
    for _k in range(0, maxD+1):
        w[_k] = np.exp(c - alpha*( k - _k )) * np.exp(c - abs(imL[i, j] - imR[i, j - _k]))

    return w/np.sum(w)








