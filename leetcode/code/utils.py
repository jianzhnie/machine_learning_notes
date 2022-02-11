'''
Author: jianzhnie
Date: 2022-02-10 10:00:34
LastEditTime: 2022-02-11 11:31:47
LastEditors: jianzhnie
Description:
'''

from typing import List


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


class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        n = len(triangle)
        dp = [[0] * n for _ in range(n)]
        dp[0][0] = triangle[0][0]

        for i in range(1, n):
            dp[i][0] = dp[i - 1][0] + triangle[i][0]
            for j in range(1, i):
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle[i][j]
            dp[i][i] = dp[i - 1][i - 1] + triangle[i][i]

        return min(dp[n - 1])


def canCompleteCircuit(gas, cost):
    n = len(gas)
    i = 0
    while i < n:
        sumOfGas = 0
        sumOfCost = 0
        cnt = 0
        while cnt < n:
            j = (i + cnt) % n
            sumOfGas += gas[j]
            sumOfCost += cost[j]
            if sumOfCost > sumOfGas:
                break
            cnt += 1
        if cnt == n:
            return i
        else:
            i = i + cnt + 1
    return -1


if __name__ == '__main__':
    nums1 = [[4, 2, 1], [1, 3, 2], [3, 2, 1]]
    res = minPathSum(nums1)
    print(res)
    gas = [2, 3, 4]
    cost = [3, 4, 3]
    res = canCompleteCircuit(gas, cost)
    print(res)
