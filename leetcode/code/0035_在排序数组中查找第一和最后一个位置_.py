'''
Author: jianzhnie
Date: 2022-02-10 11:36:51
LastEditTime: 2022-02-10 11:36:51
LastEditors: jianzhnie
Description:

'''


class Solution:
    def binarySearch(self, nums, target, lower=True):
        left, right = 0, len(nums)
        while (left < right):
            mid = (left + right) // 2
            if nums[mid] > target or (lower and nums[mid] >= target):
                right = mid - 1
                ans = mid
            else:
                left = mid + 1
        return ans

    def searchRange(self, nums, target):
        leftindex = self.binarySearch(nums, target, True)
        rightindex = self.binarySearch(nums, target, False)
        if leftindex <= rightindex and rightindex < len(nums) and nums[
                leftindex] == target and nums[rightindex] == target:
            return [leftindex, rightindex]
        return [-1, -1]


if __name__ == '__main__':
    nums1 = [1, 2, 4]
    nums2 = [1, 2, 4, 5]
