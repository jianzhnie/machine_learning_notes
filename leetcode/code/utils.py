'''
Author: jianzhnie
Date: 2022-02-10 10:00:34
LastEditTime: 2022-02-10 18:22:30
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


def minPathSum(nums):
    if not nums or not nums[0]:
        return 0
    m = len(nums)
    n = len(nums[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = nums[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + nums[i][0]

    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + nums[0][j]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + nums[i][j]
    return dp[m - 1][n - 1]


if __name__ == '__main__':
    nums1 = [[4, 2, 1], [1, 3, 2], [3, 2, 1]]
    res = minPathSum(nums1)
    print(res)
