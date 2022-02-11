'''
Author: jianzhnie
Date: 2022-02-11 10:54:43
LastEditTime: 2022-02-11 11:29:48
LastEditors: jianzhnie
Description:

'''


def canCompleteCircuit(gas, cost):
    n = len(gas)
    i = 0
    while i < n:
        sumOfGas = 0
        sumOfCost = 0
        cnt = 0
        while cnt < n:
            j = (i + cnt) % n
            sumOfGas += gas[j]
            sumOfCost += cost[j]
            if sumOfCost > sumOfGas:
                break
            cnt += 1
        if cnt == n:
            return i
        else:
            i = i + cnt + 1
    return -1
