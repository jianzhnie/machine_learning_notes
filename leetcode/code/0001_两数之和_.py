'''
Author: jianzhnie
Date: 2022-02-10 09:39:57
LastEditTime: 2022-02-11 17:48:20
LastEditors: jianzhnie
Description:

'''
from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashtable = {}
        for idx, val in enumerate(nums):
            if target - val not in hashtable:
                hashtable[val] = idx
            else:
                return hashtable[target - val], idx
        return None

    def twosum2(self, nums, target):
        hashtable = {}
        for idx, val in enumerate(nums):
            if target - val not in hashtable:
                hashtable[val] = idx
            else:
                return [hashtable[target - val], idx]
        return None


if __name__ == '__main__':
    solution = Solution()
    input = [1, 2, 4, 5, 6, 7]
    ans = solution.twoSum(input, 12)
    print(ans)
