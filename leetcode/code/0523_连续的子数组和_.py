class Solution(object):
    def checkSubarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        n = len(nums)
        preSum = [0] * (n + 1)
        for i in range(n - 1):
            preSum[i + 1] = preSum[i] + nums[i]

        if n < 2:
            return False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if (preSum[j] - preSum[i]) % k == 0:
                    return True
        return False

    def checkSubarraySum2(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        n = len(nums)
        if n < 2:
            return False
        for i in range(n - 1):
            for j in range(i + 1, n):
                subsum = sum(nums[i:(j + 1)])
                if (j - i + 1) >= 2 and subsum % k == 0:
                    return True
        return False
