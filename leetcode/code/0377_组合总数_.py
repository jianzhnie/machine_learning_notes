def combinationSum(nums, target):
    dp = [0] * (target + 1)
    dp[0] = 1

    for i in range(1, target + 1):
        for j in nums:
            if i >= j:
                dp[i] += dp[i - j]

    return dp[-1]
