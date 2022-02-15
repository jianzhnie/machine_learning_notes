'''
Author: jianzhnie
Date: 2022-02-15 18:28:39
LastEditTime: 2022-02-15 18:33:48
LastEditors: jianzhnie
Description:

'''


def topKFrequent(nums):
    hashtable = {}
    for num in nums:
        if num not in hashtable:
            hashtable[num] = 1
        else:
            hashtable[num] += 1
    return hashtable
