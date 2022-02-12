'''
Author: jianzhnie
Date: 2022-02-12 12:54:43
LastEditTime: 2022-02-12 13:44:34
LastEditors: jianzhnie
Description:

'''

import numpy as np


def meanFilter(imgArray, kernel_size=3):
    h, w = imgArray.shape
    pad = kernel_size // 2
    input = np.zeros((h + 2 * pad, w + 2 * pad))
    input[pad:pad + h, pad:pad + w] = imgArray.copy()

    conv_out = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            conv_out[y, x] = np.mean(input[y:y + kernel_size,
                                           x:x + kernel_size])

    return conv_out


def gaussianFilter(imgArray, kernel_size=3, sigma=1):
    h, w = imgArray.shape
    pad = kernel_size // 2
    input = np.zeros((h + 2 * pad, w + 2 * pad))
    input[pad:pad + h, pad:pad + w] = imgArray.copy()

    # filter
    kernel = np.zeros((kernel_size, kernel_size))
    for x in range(kernel_size):
        for y in range(kernel_size):

            kernel[y,
                   x] = np.exp(-((x - pad)**2 + (y - pad)**2)) / (2 * sigma**2)

    kernel /= (sigma * np.sqrt(2 * np.pi))
    kernel /= kernel.sum()

    # convolution
    conv_out = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            conv_out[y, x] = np.sum(
                kernel * input[y:y + kernel_size, x:x + kernel_size])

    return conv_out


def medianFilter(imgArray, kernel_size):
    h, w = imgArray.shape
    pad = kernel_size // 2
    input = np.zeros((h + 2 * pad, w + 2 * pad))
    input[pad:pad + h, pad:pad + w] = imgArray
    conv_out = np.zeros((h, w))

    for y in range(h):
        for x in range(w):
            conv_out[x, y] = np.median(input[y:y + kernel_size,
                                             x:x + kernel_size])
    return conv_out


if __name__ == '__main__':

    img = np.array(range(9))
    img = np.reshape(img, (3, 3))
    print(img)
    res = meanFilter(img)
    print(res)

    res = gaussianFilter(img)
    print(res)
