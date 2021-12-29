'''
Author: jianzhnie
Date: 2021-12-22 09:44:16
LastEditTime: 2021-12-22 09:46:32
LastEditors: jianzhnie
Description:

'''

import numpy as np


def softmax_(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x
