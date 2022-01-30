from typing import List


class Solution:
    def maxArea(self, height: List[int]) -> int:
        n = len(height)
        res = 0
        i, j = 0, n - 1
        while i < j:
            area = min(height[i], height[j]) * (j - i)
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1

            res = max(res, area)

        return res
