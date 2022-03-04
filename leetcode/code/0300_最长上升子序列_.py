from typing import List


def lengthOfLIS(nums: List[int]) -> int:
    if not nums:
        return 0
    dp = []
    for i in range(len(nums)):
        dp.append(1)
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


def lengthOfLIS2(nums: List[int]) -> int:
    d = []
    for n in nums:
        if not d or n > d[-1]:
            d.append(n)
        else:
            left, right = 0, len(d) - 1
            loc = right
            while left <= right:
                mid = (left + right) // 2
                if d[mid] >= n:
                    loc = mid
                    right = mid - 1
                else:
                    left = mid + 1
            d[loc] = n
    return len(d)
