"""
input N * N
 transpose
"""

import numpy as np


def transpose(M):
    N = 3
    for i in range(N):
        for j in range(i + 1, N):
            M[i, j] += M[j, i]
            M[j, i] = M[i, j] - M[j, i]
            M[i, j] = M[i, j] - M[j, i]
    return M


class Solution:
    def rotate(self, matrix):
        n = len(matrix)
        # 水平翻转
        for i in range(n // 2):
            for j in range(n):
                matrix[i, j], matrix[n - i - 1, j] = matrix[n - i - 1,
                                                            j], matrix[i, j]
        print(matrix)
        # 主对角线翻转
        for i in range(n):
            for j in range(i):
                matrix[i, j], matrix[j, i] = matrix[j, i], matrix[i, j]

        return matrix


if __name__ == '__main__':
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(matrix)
    ans = Solution().rotate(matrix)
    print(ans)
