import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.color import rgb2gray

import funcs2 as f2

if __name__ == '__main__':

    # Params
    MAX_DISP = 5
    alpha = 0.1
    C = 0

    # Reading imgs
    img_L = (rgb2gray( imread("imgs/im0.png")[100:200, 100:200]))
    img_R = (rgb2gray( imread("imgs/im1.png")[100:200, 100:200]))

    height, width = img_L.shape
    print(f"Img info - shape: {width, height}, max el: {np.max(img_L)}, dtype: {img_L.dtype}")

    Dm = np.zeros((height, width))
    for row in tqdm(range(0, height)):
        Dm[row, :] = f2.magic(img_L, img_R, row, MAX_DISP)

    Dm = Dm * (np.max(Dm) / MAX_DISP)
    print(np.max(Dm), Dm.shape)
    imsave("imgs/out/result.png", Dm)
    imsave("imgs/out/L.png", img_L)
    imsave("imgs/out/R.png", img_R)
