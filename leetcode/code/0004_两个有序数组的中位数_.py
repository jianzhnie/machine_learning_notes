from typing import List


class Solution:
    def findMedianSortedArrays(self, nums1: List[int],
                               nums2: List[int]) -> float:
        nums = []
        i, len1, j, len2 = 0, len(nums1), 0, len(nums2)
        while i < len1 and j < len2:
            if nums1[i] <= nums2[j]:
                nums.append(nums1[i])
                i += 1
            else:
                nums.append(nums2[j])
                j += 1

        while i < len1:
            nums.append(nums1[i])
            i += 1
        while j < len2:
            nums.append(nums2[j])
            j += 1

        total = len1 + len2
        if total % 2 == 0:
            mid = total // 2
            res = (nums[mid - 1] + nums[mid]) / 2
        else:
            mid = (total + 1) // 2
            res = nums[mid - 1]
        return nums, res


if __name__ == '__main__':
    solution = Solution()
    input1 = [1, 3, 4, 5, 7]
    input2 = [2, 6]
    ans = solution.findMedianSortedArrays(input1, input2)
    print(ans)
