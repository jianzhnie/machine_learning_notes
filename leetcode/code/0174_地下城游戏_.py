'''
Author: jianzhnie
Date: 2022-02-11 18:01:00
LastEditTime: 2022-02-11 18:47:57
LastEditors: jianzhnie
Description:

'''


def calculateMinimumHP(nums):
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
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]) + nums[i][j]
    print(dp)
    res = dp[m - 1][n - 1]
    if res > 0:
        return None
    else:
        return -res


if __name__ == '__main__':
    nums = [[-2, -3, 3], [-5, -10, 1], [10, 30, -5]]
    res = calculateMinimumHP(nums)
    print(res)
