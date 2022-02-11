'''
Author: jianzhnie
Date: 2022-02-11 17:39:10
LastEditTime: 2022-02-11 17:43:04
LastEditors: jianzhnie
Description:

'''


def maximumGap(nums):
    nums.sort()
    n = len(nums)
    maxgap = 0
    if n < 2:
        return 0
    else:
        for i in range(1, n):
            gap = nums[i] - nums[i - 1]
            maxgap = max(maxgap, gap)
    return
