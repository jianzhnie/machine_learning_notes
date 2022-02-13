'''
Author: jianzhnie
Date: 2022-02-12 14:06:01
LastEditTime: 2022-02-12 14:15:26
LastEditors: jianzhnie
Description:

'''


def roate(nums, k=1):
    n = len(nums)
    res = [0] * n

    for i in range(n):
        j = (i + k) % n
        res[j] = nums[i]
    return res


def roate2(nums, k=1):
    n = len(nums)
    nums = [nums[(i + k) % n] for i in range(n)]
    return nums


if __name__ == '__main__':

    nums = [1, 1, 2, 3, 4]

    res = roate(nums, 3)
    print(res)

    res = roate2(nums, 3)
    print(res)
