from typing import List

import numpy as np


class Solution_:
    def generateMatrix(self, n: int) -> List[List[int]]:
        # 初始化要填充的正方形
        matrix = [[0] * n for _ in range(n)]

        left, right, up, down = 0, n - 1, 0, n - 1
        number = 1  # 要填充的数字

        while left < right and up < down:

            # 从左到右填充上边
            for x in range(left, right):
                matrix[up][x] = number
                number += 1

            # 从上到下填充右边
            for y in range(up, down):
                matrix[y][right] = number
                number += 1

            # 从右到左填充下边
            for x in range(right, left, -1):
                matrix[down][x] = number
                number += 1

            # 从下到上填充左边
            for y in range(down, up, -1):
                matrix[y][left] = number
                number += 1

            # 缩小要填充的范围
            left += 1
            right -= 1
            up += 1
            down -= 1

        # 如果阶数为奇数，额外填充一次中心
        if n % 2:
            matrix[n // 2][n // 2] = number

        return matrix


class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m, n = len(matrix), len(matrix[0])
        left, right = 0, m - 1
        up, down = 0, n - 1
        res = []
        while left < right and up < down:

            for i in range(left, right):
                number = matrix[up][i]
                res.append(number)

            for j in range(up, down):
                number = matrix[j][right]
                res.append(number)

            for i in range(right, left, -1):
                number = matrix[down][i]
                res.append(number)

            for j in range(down, up, -1):
                number = matrix[j][left]
                res.append(number)

            left += 1
            right -= 1
            up += 1
            down -= 1

        if n % 2:
            number = matrix[n // 2][n // 2]
            res.append(number)
        return res


if __name__ == '__main__':
    solution = Solution_()
    ans = solution.generateMatrix(10)
    ans = np.array(ans)
    print(ans)

    solution = Solution()
    ans = solution.spiralOrder(ans)
    ans = np.array(ans)
    print(ans)
