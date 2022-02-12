'''
Author: jianzhnie
Date: 2022-02-12 17:32:55
LastEditTime: 2022-02-12 17:48:58
LastEditors: jianzhnie
Description:

'''


def minSubArray(nums, target):
    n = len(nums)
    flag = False
    res = float('inf')
    inner_sum = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            inner_sum = sum(nums[i:j])
            if inner_sum >= target:
                flag = True
                res = min(res, j - i)
    return res if flag else 0


if __name__ == '__main__':

    nums = [1, 1, 2, 3, 3, 4]
    res = minSubArray(nums, 6)
    print(res)
