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


if __name__ == '__main__':
    solution = Solution()
    input = [1, 2, 4, 5, 6, 7]
    ans = solution.twoSum(input, 12)
    print(ans)
