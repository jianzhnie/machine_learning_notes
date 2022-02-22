def change(amount, coins):
    dp = [0] * (amount + 1)
    for i in coins:
        for j in range(i, amount):
            dp[j] += dp[j - i]
    return dp[amount]
