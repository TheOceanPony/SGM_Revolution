import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.color import rgb2gray

import funcs2 as f2



if __name__ == '__main__':

    # Params
    maxD = 5
    alpha = 1
    beta = 100
    C = 0

    # Reading imgs
    img_L = (rgb2gray( (imread("imgs/im0.png")) ))
    img_R = (rgb2gray( (imread("imgs/im1.png")) ))

    height, width = img_L.shape
    print(f"Img info - shape: {width, height}, max el: {np.max(img_L)}, dtype: {img_L.dtype}")

    G = f2.G_init(maxD, alpha, C)
    H = f2.H_init(img_L, img_R, maxD, beta, C)

    P = f2.P_init(H, G)

    Dm = np.zeros((height, width), dtype=np.float64)
    for row in tqdm(range(0, height)):
        Dm[row, :] = f2.magic(row, width, maxD, P)

    # Dm = Dm * (np.max(Dm) / maxD)
    print(np.max(Dm), np.min(Dm), Dm.shape)
    imsave("imgs/out/result_a2.png", Dm)
