'''
Author: jianzhnie
Date: 2022-02-12 17:17:34
LastEditTime: 2022-02-12 17:21:00
LastEditors: jianzhnie
Description:

'''


def isHappy(n):
    def get_next(n):
        total = 0
        while n > 0:
            n, digit = divmod(n, 10)
            total += digit**2
        return total

    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = get_next(n)

    return n == 1
