from typing import List


class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        hashtable = {}
        for x in nums:
            if x not in hashtable:
                hashtable[x] = 1
            else:
                hashtable[x] += 1
            if hashtable[x] >= 2:
                return True
        return False

    # 复杂度： n *n
    def containsDuplicate2(self, nums, k):
        n = len(nums)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if nums[i] == nums[j] and j - i <= k:
                    return True
        return False

    # 复杂度： n* k
    def containsNearbyDuplicate3(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        n = len(nums)
        for i in range(n):
            j = 1
            while j <= k and j < n - i:
                if nums[i] == nums[i + j]:
                    return True
                j += 1
        return False

    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        hashtable = {}
        for i, x in enumerate(nums):
            if x not in hashtable:
                hashtable[x] = [i]
            else:
                hashtable[x].append(i)

            if len(hashtable[x]) >= 2 and (hashtable[x][-1] -
                                           hashtable[x][-2]) <= k:
                return True
        return False

    def containsNearbyDuplicate2(self, nums: List[int], k: int) -> bool:
        pos = {}
        for i, num in enumerate(nums):
            if num in pos and i - pos[num] <= k:
                return True
            pos[num] = i
        return False

    # 给你一个整数数组 nums 和两个整数 k 和 t 。
    # 请你判断是否存在 两个不同下标 i 和 j，使得 abs(nums[i] - nums[j]) <= t ，
    # 同时又满足 abs(i - j) <= k 。
    # 如果存在则返回 true，不存在返回 false。

    def containsNearbyAlmostDuplicate(self, nums: List[int], k: int,
                                      t: int) -> bool:

        n = len(nums)
        for i in range(n):
            j = 1
            while j <= k and j < n - i:
                if abs(nums[i] - nums[j]) <= t:
                    return True
        return False


if __name__ == '__main__':
    solution = Solution()
    intervals = [2, 0, 14, 1, 5, 3, 6, 7]
    res = solution.containsDuplicate(intervals)
    print(res)
    res = solution.containsNearbyDuplicate(intervals, 1)
    print(res)
    res = solution.containsNearbyAlmostDuplicate(intervals, 4, 3)
    print(res)
