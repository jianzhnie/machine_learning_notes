def maxSubArray(nums):
    n = len(nums)
    res = 0
    subres = 1
    for i in range(1, n):
        if nums[i] - nums[i - 1] == 1:
            subres += 1
        else:
            subres = 1
        res = max(res, subres)
    return res


def maxlenSubArray(nums):
    n = len(nums)
    dp = [0] * 10
    for i in range(n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


nums = [1, 2, 3, 0, 9, 10, 11, 12]
res = maxSubArray(nums)
print(res)
