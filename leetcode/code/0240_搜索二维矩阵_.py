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
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        m = len(matrix)
        n = len(matrix[0])

        left = 0
        right = m
        up = 0
        down = n

        while left < right and up < down:
            midL = left + (right - left) // 2
            midH = up + (down - up) // 2

            if matrix[up][midL] == target:
                return True
            elif matrix[up][midL] < target:
                left = midL + 1
            else:
                right = midL - 1

            if matrix[left][midH] == target:
                return True
            elif matrix[left][midH] < target:
                up = midH + 1
            else:
                down = midH - 1

        return matrix[midH][midL]

    def searchMatrix2(self, matrix: List[List[int]], target: int) -> bool:
        for numlist in matrix:
            if binarySearch(numlist, target):
                return True
        return False

    def searchMatrix3(self, matrix: List[List[int]], target: int) -> bool:
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
