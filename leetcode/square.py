from typing import List


class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        n = len(nums)
        i, j, k = 0, n - 1, n - 1
        ans = [-1] * n
        while i <= j:
            left = nums[i]**2
            right = nums[j]**2
            if left > right:
                ans[k] = left
                i += 1
            else:
                ans[k] = right
                j -= 1
            k -= 1
        return ans


if __name__ == '__main__':
    solution = Solution()
    nums = [-4, -1, 0, 3, 10]
    ans = solution.sortedSquares(nums)
    print(ans)
