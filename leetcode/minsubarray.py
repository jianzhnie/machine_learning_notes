from typing import List


class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        # 定义一个无限大的数
        res = float('inf')
        Sum = 0
        index = 0
        for i in range(len(nums)):
            Sum += nums[i]
            while Sum >= s:
                res = min(res, i - index + 1)
                Sum -= nums[index]
                index += 1

        return 0 if res == float('inf') else res


if __name__ == '__main__':
    solution = Solution()
    nums = [2, 4, 3, 1, 2, 3, 3]
    ans = solution.minSubArrayLen(7, nums)
    print(ans)
