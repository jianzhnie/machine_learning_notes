from typing import List


class Solution(object):
    def singleNonDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left = 0
        right = len(nums) - 1
        if len(nums) < 2:
            return nums[0]
        while left <= right:
            if nums[left] != nums[left + 1]:
                return nums[left]
            if nums[right] != nums[right - 1]:
                return nums[right]
            left += 2
            right -= 2

    def singleNonDuplicate2(self, nums: List[int]) -> int:
        low, high = 0, len(nums) - 1
        while low < high:
            mid = (low + high) // 2
            if nums[mid] == nums[mid + 1]:
                low = mid + 1
            else:
                high = mid
        return nums[low]


if __name__ == '__main__':
    nums = [1, 1, 3, 4, 4]
