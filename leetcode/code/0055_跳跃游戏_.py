'''
Author: jianzhnie
Date: 2022-02-10 15:32:51
LastEditTime: 2022-02-10 15:45:58
LastEditors: jianzhnie
Description:

'''

from typing import List


class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n, rightmost = len(nums), 0
        for i in range(n):
            # 判断 rightmax 是否到达 i 的位置
            # 若到达,则可从i的位置继续跳
            if i <= rightmost:
                rightmost = max(rightmost, i + nums[i])
                if rightmost >= n - 1:
                    return True
        return False


if __name__ == '__main__':
    solution = Solution()
    nums = [3, 2, 1, 0, 4]
    ans = solution.canJump(nums)
    print(ans)
