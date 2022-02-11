'''
Author: jianzhnie
Date: 2022-02-11 11:42:39
LastEditTime: 2022-02-11 12:00:05
LastEditors: jianzhnie
Description:

'''

from typing import List


def candy(ratings: List[int]) -> int:
    n = len(ratings)
    left = [0] * n
    right = [0] * n
    for i in range(n):
        if i > 0 and ratings[i] > ratings[i - 1]:
            left[i] = left[i - 1] + 1
        else:
            left[i] = 1
    for i in range(n - 1, -1, -1):
        if i < n - 1 and ratings[i] > ratings[i + 1]:
            right[i] = right[i + 1] + 1
        else:
            right[i] = 1
    ret = sum(max(left[i], right[i]) for i in range(n))
    return ret


if __name__ == '__main__':
    ratings = [1, 0, 2]
    res = candy(ratings)
    print(res)
