'''
Author: jianzhnie
Date: 2022-02-10 11:29:16
LastEditTime: 2022-02-10 11:29:17
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
