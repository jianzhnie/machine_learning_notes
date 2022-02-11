'''
Author: jianzhnie
Date: 2022-02-10 17:46:18
LastEditTime: 2022-02-10 18:35:10
LastEditors: jianzhnie
Description:

'''


def maxArea(nums):
    n = len(nums)
    maxarea = 0
    for i in range(n):
        for j in range(i, n):
            minHeight = min(nums[i:j + 1])
            area = minHeight * (j - i + 1)
            maxarea = max(maxarea, area)
    return maxarea


def maxArea2(nums):
    n = len(nums)
    maxarea = 0
    for i in range(n - 1):
        minHeight = float('inf')
        for j in range(i, n):
            minHeight = min(minHeight, nums[j])
            area = minHeight * (j - i + 1)
            maxarea = max(maxarea, area)
    return maxarea


if __name__ == '__main__':
    nums = [1, 2, 3, 3, 4, 2, 4]
    res = maxArea(nums)
    print(res)
    res = maxArea2(nums)
    print(res)
