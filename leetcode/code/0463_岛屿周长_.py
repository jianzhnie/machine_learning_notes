from typing import List


def islandPerimeter(grid):
    m = len(grid)
    n = len(grid[0])
    sum = 0
    cover = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                sum += 1
                if i - 1 >= 0 and grid[i - 1][j] == 1:
                    cover += 1
                if j - 1 >= 0 and grid[i][j - 1] == 1:
                    cover += 1

    return sum * 4 - cover * 2


def islandPerimeter2(grid: List[List[int]]) -> int:

    m = len(grid)
    n = len(grid[0])

    # 创建res二维素组记录答案
    res = [[0] * n for j in range(m)]

    for i in range(m):
        for j in range(len(grid[i])):
            # 如果当前位置为水域，不做修改或reset res[i][j] = 0
            if grid[i][j] == 0:
                res[i][j] = 0
            # 如果当前位置为陆地，往四个方向判断，update res[i][j]
            elif grid[i][j] == 1:
                if i == 0 or (i > 0 and grid[i - 1][j] == 0):
                    res[i][j] += 1
                if j == 0 or (j > 0 and grid[i][j - 1] == 0):
                    res[i][j] += 1
                if i == m - 1 or (i < m - 1 and grid[i + 1][j] == 0):
                    res[i][j] += 1
                if j == n - 1 or (j < n - 1 and grid[i][j + 1] == 0):
                    res[i][j] += 1

    # 最后求和res矩阵，这里其实不一定需要矩阵记录，可以设置一个variable res 记录边长，舍矩阵无非是更加形象而已
    ans = sum([sum(row) for row in res])

    return ans
