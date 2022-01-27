from typing import List

import numpy as np


class Solution:
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


if __name__ == '__main__':
    solution = Solution()
    ans = solution.generateMatrix(6)
    ans = np.array(ans)
    print(ans)
