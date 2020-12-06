import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.transform import rescale, downscale_local_mean

import funcs2 as f2


def main(maxD, a, b):
    # Params
    maxD = 10
    alpha = a
    beta = b
    fname = f"imgs/cloth/maxD:{maxD},a:{alpha},b:{beta}.png"
    C = 0

    # Reading imgs
    img_L = (rgb2gray( (imread("imgs/im0_cloth.png")) ))
    img_R = (rgb2gray( (imread("imgs/im1_cloth.png")) ))

    img_L = downscale_local_mean(img_L, (4, 4))
    img_R = downscale_local_mean(img_R, (4, 4))


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
    imsave(fname, Dm)
    #imsave("imgs/out/L.png", img_L)
    #imsave("imgs/out/R.png", img_R)


main(5, 2, 50)
