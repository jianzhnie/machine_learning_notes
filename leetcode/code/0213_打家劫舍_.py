def rob(nums):
    n = len(nums)

    def robRange(nums, start, end):
        if end == start:
            return nums[start]
        dp = [0] * n
        dp[start] = nums[start]
        dp[start + 1] = max(nums[start], nums[start + 1])
        for i in range(start + 2, end):
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])

    if n == 0:
        return 0
    if n == 1:
        return nums[0]
    res1 = robRange(nums, 0, n - 1)
    res2 = robRange(nums, 1, n)

    return max(res1, res2)
