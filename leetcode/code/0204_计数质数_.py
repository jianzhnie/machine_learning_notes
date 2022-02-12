'''
Author: jianzhnie
Date: 2022-02-12 17:27:06
LastEditTime: 2022-02-12 17:30:57
LastEditors: jianzhnie
Description:

'''


def isPrime(x):
    i = 2
    while i <= x:
        if x % i == 0:
            return False
    return True


def countPrimes(n):
    res = 0
    for i in range(n):
        res += isPrime(i)
    return res
