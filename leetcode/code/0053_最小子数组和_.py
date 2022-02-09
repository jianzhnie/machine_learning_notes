# class Solution {
# public:
#     int maxSubArray(vector<int>& nums) {
#         int pre = 0, maxAns = nums[0];
#         for (const auto &x: nums) {
#             pre = max(pre + x, x);
#             maxAns = max(maxAns, pre);
#         }
#         return maxAns;
#     }
# };
from typing import List


class Solution:
    def maxSubArray_recycle(self, nums: List[int]) -> int:
        n = len(nums)
        maxsum = float('-inf')
        if n < 2:
            maxsum = sum(nums)
        else:
            for i in range(n - 1):
                for j in range(i + 1, n):
                    inner_sum = sum(nums[i:j + 1])

                    if inner_sum > maxsum:
                        maxsum = inner_sum
        return maxsum

    def maxSubArray_(self, nums: List[int]):
        sum = 0
        maxsum = float('-inf')
        for x in nums:
            sum = sum + x
            maxsum = max(maxsum, sum)
            if sum < 0:
                sum = 0
        return maxsum
