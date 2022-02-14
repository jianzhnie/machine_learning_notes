'''
Author: jianzhnie
Date: 2022-02-14 09:36:28
LastEditTime: 2022-02-14 10:51:44
LastEditors: jianzhnie
Description:

'''

from typing import List

import numpy as np


def gameOfLife(board: List[List[int]]) -> None:
    """Do not return anything, modify board in-place instead."""

    neighbors = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1),
                 (1, 1)]

    rows = len(board)
    cols = len(board[0])

    # 从原数组复制一份到 copy_board 中
    copy_board = [[board[row][col] for col in range(cols)]
                  for row in range(rows)]

    # 遍历面板每一个格子里的细胞
    for row in range(rows):
        for col in range(cols):

            # 对于每一个细胞统计其八个相邻位置里的活细胞数量
            live_neighbors = 0
            for neighbor in neighbors:

                r = (row + neighbor[0])
                c = (col + neighbor[1])

                # 查看相邻的细胞是否是活细胞
                if (r < rows and r >= 0) and (c < cols
                                              and c >= 0) and (copy_board[r][c]
                                                               == 1):
                    live_neighbors += 1

            # 规则 1 或规则 3
            if copy_board[row][col] == 1 and (live_neighbors < 2
                                              or live_neighbors > 3):
                board[row][col] = 0
            # 规则 4
            if copy_board[row][col] == 0 and live_neighbors == 3:
                board[row][col] = 1
    return board


if __name__ == '__main__':
    # 示例 1：
    # 输入：board = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
    # 输出：[[0,0,0],[1,0,1],[0,1,1],[0,1,0]]
    board = [[0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]]
    print(np.array(board))
    res = gameOfLife(board)
    print(np.array(res))
