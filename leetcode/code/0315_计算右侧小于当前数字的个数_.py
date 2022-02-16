'''
Author: jianzhnie
Date: 2022-02-15 09:10:03
LastEditTime: 2022-02-15 09:18:07
LastEditors: jianzhnie
Description:

'''


def countSmaller(nums):
    n = len(nums)
    res = [0] * n
    for i in range(n - 1):
        cnt = 0
        for j in range(i + 1, n):
            if nums[j] < nums[i]:
                cnt += 1
        res[i] = cnt
    return res
