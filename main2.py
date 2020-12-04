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
    img_L = (rgb2gray( imread("imgs/im0.png")))
    img_R = (rgb2gray( imread("imgs/im1.png")))

    #img_L = img_L[300:900, 300:900]
    #img_R = img_R[300:900, 300:900]

    height, width = img_L.shape
    #print(f"Img info - shape: {width, height}, max el: {np.max(img_L)}, dtype: {img_L.dtype}")

    # Initialise
    H = f2.init_unary(MAX_DISP, img_L, img_R, C)
    G = f2.init_binary(MAX_DISP, alpha, C)

    #
    Li = f2.init_left(H, G)




