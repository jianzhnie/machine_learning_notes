'''
Author: jianzhnie
Date: 2022-02-15 12:11:29
LastEditTime: 2022-02-15 12:30:04
LastEditors: jianzhnie
Description:

'''

from typing import List


class Solution:

    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix:
            return 0

        def dfs(row: int, column: int) -> int:
            best = 1
            for dx, dy in Solution.DIRS:
                newRow, newColumn = row + dx, column + dy
                if 0 <= newRow < rows and 0 <= newColumn < columns and matrix[
                        newRow][newColumn] > matrix[row][column]:
                    best = max(best, dfs(newRow, newColumn) + 1)
            return best

        ans = 0
        rows, columns = len(matrix), len(matrix[0])
        for i in range(rows):
            for j in range(columns):
                ans = max(ans, dfs(i, j))
        return ans

    def longestIncreasingPath2(self, matrix):

        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        def dfs(row, col):
            best = 1
            for dx, dy in directions:
                newrow, newcol = row + dx, col + dy
                if 0 <= newrow < rows and 0 <= newcol < cols and matrix[
                        newrow][newcol] > matrix[row][col]:
                    best = max(best, dfs(newrow, newcol) + 1)

            return best

        if not matrix:
            return 0

        res = 0
        rows, cols = len(matrix), len(matrix[0])
        for row in range(rows):
            for col in range(cols):
                res = max(res, dfs(row, col))
        return res
