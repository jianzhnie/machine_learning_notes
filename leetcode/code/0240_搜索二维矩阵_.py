from typing import List


def binarySearch(nums, target):
    left, right = 0, len(nums)
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return True
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return False


class Solution(object):
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        for numlist in matrix:
            if binarySearch(numlist, target):
                return True
        return False

    def searchMatrix2(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        x, y = 0, n - 1
        while x < m and y >= 0:
            if matrix[x][y] == target:
                return True
            if matrix[x][y] > target:
                y -= 1
            else:
                x += 1
        return False


if __name__ == '__main__':
    solution = Solution()
    matrix = [[1, 4, 7, 11, 15], [2, 5, 8, 12, 19], [3, 6, 9, 16, 22],
              [10, 13, 14, 17, 24], [18, 21, 23, 26, 30]]
    res = solution.searchMatrix(matrix, 5)
    print(res)

    res = solution.binarysearchMatrix(matrix, 5)
    print(res)
