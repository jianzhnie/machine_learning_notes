from typing import List


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
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
                val = 0 - (nums[i] + nums[j])
                if val in hashmap:
                    count = (nums[i] == val) + (nums[j] == val)

                    if hashmap[val] > count:
                        res.add(tuple(sorted([nums[i], nums[j], val])))

                else:
                    continue
        return list(res)
