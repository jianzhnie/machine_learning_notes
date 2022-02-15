'''
Author: jianzhnie
Date: 2022-02-10 11:29:16
LastEditTime: 2022-02-15 09:04:59
LastEditors: jianzhnie
Description:

'''

from typing import List


def maxProfit(nums):
    n = len(nums)
    i, j = 0, 0
    res = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if nums[j] > nums[i]:
                res = max(res, nums[j] - nums[i])
    return res


def maxProfit2(prices: List[int]) -> int:
    inf = int(1e9)
    minprice = inf
    maxprofit = 0
    for price in prices:
        maxprofit = max(price - minprice, maxprofit)
        minprice = min(price, minprice)
    return maxprofit


# 可以进行多次交易
def maxProfit3(prices):
    n = len(prices)
    dp0 = 0
    dp1 = -prices[0]
    for i in range(1, n):
        newdp0 = max(dp0, dp1 + prices[i])
        newdp1 = max(dp1, dp0 - prices[i])
        dp0 = newdp0
        dp1 = newdp1
    return dp0


# 可以进行多次交易
def maxProfit3_(prices):
    n = len(prices)
    dp = [[0] * 2 for _ in range(n)]
    dp[0][0] = 0
    dp[0][1] = -prices[0]
    for i in range(1, n):
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
        dp[i][1] = max(dp[i - 1][1], dp[i][0] - prices[i])
    return dp[n - 1][0]


# 可以进行多次交易
def maxProfit4(prices):
    ans = 0
    n = len(prices)
    for i in range(1, n):
        ans += max(0, prices[i] - prices[i - 1])
    return ans


# 限制交易次数不超过 2
def maxProfit5(prices: List[int]) -> int:
    n = len(prices)
    buy1 = buy2 = -prices[0]
    sell1 = sell2 = 0
    for i in range(1, n):
        buy1 = max(buy1, -prices[i])
        sell1 = max(sell1, buy1 + prices[i])
        buy2 = max(buy2, sell1 - prices[i])
        sell2 = max(sell2, buy2 + prices[i])
    return sell2


# 可以进行多次交易
# 包含冷冻期
def maxProfit6(prices: List[int]) -> int:
    if not prices:
        return 0

    n = len(prices)
    # f[i][0]: 手上持有股票的最大收益
    # f[i][1]: 手上不持有股票，并且处于冷冻期中的累计最大收益
    # f[i][2]: 手上不持有股票，并且不在冷冻期中的累计最大收益
    f = [[-prices[0], 0, 0]] + [[0] * 3 for _ in range(n - 1)]
    for i in range(1, n):
        f[i][0] = max(f[i - 1][0], f[i - 1][2] - prices[i])
        f[i][1] = f[i - 1][0] + prices[i]
        f[i][2] = max(f[i - 1][1], f[i - 1][2])

    return max(f[n - 1][1], f[n - 1][2])


def maxProfit7(prices):
    if not prices:
        return 0

    n = len(prices)
    f = [[-prices[0], 0, 0]] + [[0] * 3 for _ in range(n - 1)]

    for i in range(1, n):
        f[i][0] = max(f[i - 1][0], f[i - 1][2] - prices[i])
        f[i][1] = f[i - 1][0] + prices[i]
        f[i][2] = max(f[i - 1][1], f[i - 1][2])

    print(f)
    return max(f[n - 1][1], f[n - 1][2])


if __name__ == '__main__':
    nums = [1, 5, 10, 4, 5, 12, 3, 2, 7, 5]
    res = maxProfit3(nums)
    print(res)
    res = maxProfit3_(nums)
    print(res)
    res = maxProfit4(nums)
    print(res)
    res = maxProfit5(nums)
    print(res)
    res = maxProfit7(nums)
    print(res)
