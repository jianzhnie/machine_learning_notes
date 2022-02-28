from typing import List


def canPartition(nums: List[int]) -> bool:
    taraget = sum(nums)
    if taraget % 2 == 1:
        return False
    taraget //= 2
    dp = [0] * 10001
    for i in range(len(nums)):
        for j in range(taraget, nums[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
    return taraget == dp[taraget]
