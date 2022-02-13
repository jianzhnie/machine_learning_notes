'''
Author: jianzhnie
Date: 2022-02-12 16:28:19
LastEditTime: 2022-02-12 17:50:41
LastEditors: jianzhnie
Description:

'''


def rob(nums):
    n = len(nums)
    dp = [0] * n
    dp[0], dp[1] = nums[0], max(nums[0], nums[1])
    for i in range(2, n):
        rob1 = dp[i - 2] + nums[i]
        rob2 = dp[i - 1]
        dp[i] = max(rob1, rob2)
    return dp[n - 1]


def rob2(nums):
    n = len(nums)
    memo = [-1] * n

    def dfs(i):
        if i >= n:
            return 0
        if memo[i] != -1:
            return memo[i]
        left = dfs(i + 1)
        right = dfs(i + 2)
        memo[i] = max(left, right + nums[i])

        return memo[i]

    return dfs(0)


if __name__ == '__main__':

    nums = [1, 1, 2, 3, 3, 4]

    res = rob(nums)
    print(res)
