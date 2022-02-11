'''
Author: jianzhnie
Date: 2022-02-10 16:17:25
LastEditTime: 2022-02-10 16:36:33
LastEditors: jianzhnie
Description:

'''


def matrix2zero(matrix):
    m = len(matrix)
    n = len(matrix[0])
    row = [False] * m
    col = [False] * n

    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 0:
                row[i] = True
                col[j] = True

    for i in range(m):
        for j in range(n):
            if row[i] or col[j]:
                matrix[i][j] = 0

    return matrix
