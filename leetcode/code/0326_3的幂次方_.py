'''
Author: jianzhnie
Date: 2022-02-15 10:28:03
LastEditTime: 2022-02-15 11:21:22
LastEditors: jianzhnie
Description:

'''


def isPowerOf3(n):
    while n and n % 3 == 0:
        n = n // 3
    return n == 1


if __name__ == '__main__':
    res = isPowerOf3(12)
    print(res)
