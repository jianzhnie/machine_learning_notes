'''
Author: jianzhnie
Date: 2022-02-10 10:00:34
LastEditTime: 2022-02-10 12:20:46
LastEditors: jianzhnie
Description:
'''


def trap(nums):
    n = len(nums)
    leftmax = [nums[0]] + [0] * (n - 1)
    for i in range(1, n):
        leftmax[i] = max(leftmax[i - 1], nums[i])
    rightmax = [0] * (n - 1) + [nums[n - 1]]
    for i in range(n - 2, -1, -1):
        rightmax[i] = max(rightmax[i + 1], nums[i])

    ans = sum(min(leftmax[i], rightmax[i]) - nums[i] for i in range(n))
    return ans


if __name__ == '__main__':
    nums1 = [4, 2, 1, 4]
    res = trap(nums1)
    print(res)
