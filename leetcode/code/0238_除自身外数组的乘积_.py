class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        n = len(nums)
        # L 和 R 分别表示左右两侧的乘积列表
        left, right, answer = [0] * n, [0] * n, [0] * n
        # L[i] 为索引 i 左侧所有元素的乘积
        # 对于索引为 '0' 的元素，因为左侧没有元素，所以 L[0] = 1

        left[0] = 1
        for i in range(1, n):
            left[i] = nums[i - i] * left[i - 1]
        # R[i] 为索引 i 右侧所有元素的乘积
        # 对于索引为 'length-1' 的元素，因为右侧没有元素，所以 R[length-1] = 1
        right[n - 1] = 1
        for i in range(n - 2, -1, -1):
            right[i] = nums[i + 1] * right[i + 1]
        # 对于索引 i，除 nums[i] 之外其余各元素的乘积就是左侧所有元素的乘积乘以右侧所有元素的乘积
        for i in range(n):
            answer[i] = left[i] * right[i]

        return answer


if __name__ == '__main__':
    solution = Solution()
    intervals = [2, 0, 1, 3, 5]
    res = solution.productExceptSelf(intervals)
    print(res)
