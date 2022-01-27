import numpy as np


def intergral(img):
    M, N = img.shape[0], img.shape[1]
    intergral_img = np.zeros((M, N), dtype=np.int32)
    for i in range(M):
        for j in range(N):
            intergral_img = img[i, j] + intergral_img[
                i - 1, j] + intergral_img[i, j - 1] - intergral_img[i - 1,
                                                                    j - 1]

    return intergral_img
