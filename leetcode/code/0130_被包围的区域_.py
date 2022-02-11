'''
Author: jianzhnie
Date: 2022-02-11 10:37:54
LastEditTime: 2022-02-11 10:41:45
LastEditors: jianzhnie
Description:

'''
from typing import List


def solve(board: List[List[str]]) -> None:
    if not board:
        return

    n, m = len(board), len(board[0])

    # 深度优先算法
    def dfs(x, y):
        if not 0 <= x < n or not 0 <= y < m or board[x][y] != 'O':
            return

        board[x][y] = 'A'
        dfs(x + 1, y)
        dfs(x - 1, y)
        dfs(x, y + 1)
        dfs(x, y - 1)

    for i in range(n):
        dfs(i, 0)
        dfs(i, m - 1)

    for i in range(m - 1):
        dfs(0, i)
        dfs(n - 1, i)

    for i in range(n):
        for j in range(m):
            if board[i][j] == 'A':
                board[i][j] = 'O'
            elif board[i][j] == 'O':
                board[i][j] = 'X'
