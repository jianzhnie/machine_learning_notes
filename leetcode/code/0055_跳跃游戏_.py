'''
Author: jianzhnie
Date: 2022-02-10 15:32:51
LastEditTime: 2022-02-10 15:41:54
LastEditors: jianzhnie
Description:

'''

from typing import List


class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n, rightmost = len(nums), 0
        for i in range(n):
            if i <= rightmost:
                rightmost = max(rightmost, i + nums[i])
                if rightmost >= n - 1:
                    return True
        return False

    def canJump2(nums):
        n, rightmax = len(nums), 0
        for i in range(n):
            rightmax = max(rightmax, i + nums[i])
            if rightmax >= n - 1:
                return True
        return False


if __name__ == '__main__':
    solution = Solution()
    nums = [1, 2, 2, 3, 4, 1, 2]
    ans = solution.canJump2(nums)
    print(ans)
