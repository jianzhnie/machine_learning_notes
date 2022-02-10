'''
Author: jianzhnie
Date: 2022-02-10 09:39:57
LastEditTime: 2022-02-10 11:34:44
LastEditors: jianzhnie
Description:

'''
from typing import List


class Solution:
    def threeSum(self, nums: List[int], target) -> List[List[int]]:
        hashmap = {}
        for n in nums:
            if n not in hashmap:
                hashmap[n] = 1
            else:
                hashmap[n] += 1
        res = set()
        N = len(nums)
        for i in range(N - 1):
            for j in range(i + 1, N):
                val = target - (nums[i] + nums[j])
                if val in hashmap:
                    count = (nums[i] == val) + (nums[j] == val)

                    if hashmap[val] > count:
                        res.add(tuple(sorted([nums[i], nums[j], val])))

                else:
                    continue
        return list(res)


if __name__ == '__main__':
    nums1 = [1, 2, 4]
    nums2 = [1, 2, 3, 4, 0, 5]
    solution = Solution()
    res = solution.threeSum(nums2, 6)
    print(res)
