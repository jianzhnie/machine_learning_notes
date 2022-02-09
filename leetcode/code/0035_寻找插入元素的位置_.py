# class Solution {
# public:
#     int searchInsert(vector<int>& nums, int target) {
#         int n = nums.size();
#         int l=0,r=n-1;
#         while(l<=r){
#             int mid=l+(r-l)/2;
#             if(nums[mid]<target)
#                 l=mid+1;
#             else r=mid-1;
#         }
#         return l;
#     }
# };

from typing import List


class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        n = len(nums)
        left, right = 0, n - 1
        while left <= right:
            mid = left + (right - left) / 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left
