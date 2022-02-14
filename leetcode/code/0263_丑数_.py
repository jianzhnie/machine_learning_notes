'''
Author: jianzhnie
Date: 2022-02-14 08:39:59
LastEditTime: 2022-02-14 08:51:03
LastEditors: jianzhnie
Description:

'''


class Solution:
    def isUgly(self, n: int) -> bool:
        if n <= 0:
            return False

        factors = [2, 3, 5]
        for factor in factors:
            while n % factor == 0:
                n //= factor

        return n == 1


if __name__ == '__main__':
    n = 100
    solution = Solution()
    res = solution.isUgly(n)
    print(res)
