'''
Author: jianzhnie
Date: 2022-02-15 09:32:24
LastEditTime: 2022-02-15 10:22:49
LastEditors: jianzhnie
Description:

'''
import functools
from typing import List


class Solution:
    # 函数自调用
    def coinChange(self, coins: List[int], amount: int) -> int:
        @functools.lru_cache(amount)
        def dp(rem) -> int:
            if rem < 0:
                return -1
            if rem == 0:
                return 0
            mini = int(1e9)
            for coin in self.coins:
                res = dp(rem - coin)
                if res >= 0 and res < mini:
                    mini = res + 1
            return mini if mini < int(1e9) else -1

        self.coins = coins
        if amount < 1:
            return 0
        return dp(amount)

    # 动态规划
    def coinChange2(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0

        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)
        return dp[amount] if dp[amount] != float('inf') else -1


if __name__ == '__main__':
    coins = [1, 2, 5]
    amount = 11
    solution = Solution()
    res = solution.coinChange(coins, 11)
    print(res)
