'''
Author: jianzhnie
Date: 2022-02-11 12:28:34
LastEditTime: 2022-02-11 17:17:23
LastEditors: jianzhnie
Description:

'''


def maxProduct(nums):
    n = len(nums)
    maxF = [1] * n
    minF = [1] * n

    for i in range(n):
        maxF[i] = max([
            maxF[i - 1] * nums[i],
            minF[i - 1] * nums[i],
            nums[i],
        ])
        minF[i] = min([
            minF[i - 1] * nums[i],
            maxF[i - 1] * nums[i],
            nums[i],
        ])
    res = max(maxF)
    return res


if __name__ == '__main__':

    nums = [-2, 4, -2, 5, 4, -3]
    res = maxProduct(nums)
    print(res)
