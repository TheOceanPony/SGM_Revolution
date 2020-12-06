import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.color import rgb2gray

import funcs2 as f2


if __name__ == '__main__':

    # Params
    maxD = 5
    alpha = 2.5
    beta = 50
    C = 0

    # Reading imgs
    img_L = (rgb2gray( (imread("imgs/im0.png")) ))
    img_R = (rgb2gray( (imread("imgs/im1.png")) ))

    height, width = img_L.shape
    print(f"Img info - shape: {width, height}, max el: {np.max(img_L)}, dtype: {img_L.dtype}")

    G = f2.G_init(maxD, alpha, C)
    H = f2.H_init(img_L, img_R, maxD, beta, C)

    P = f2.P_init(H, G)

    F = np.zeros((maxD + 1, width), dtype=np.float64)
    Res = np.zeros(width, dtype=np.float64)

    Dm = np.zeros((height, width), dtype=np.float64)
    for row in tqdm(range(0, height)):
        Dm[row, :] = f2.magic(row, width, maxD, F, P, Res)

    Dm = Dm/Dm.max()
    print(np.max(Dm), np.min(Dm), Dm.shape)
    imsave("imgs/out/result_a2.png", Dm)
