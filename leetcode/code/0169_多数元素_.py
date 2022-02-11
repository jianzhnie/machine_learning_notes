'''
Author: jianzhnie
Date: 2022-02-11 17:52:23
LastEditTime: 2022-02-11 17:58:36
LastEditors: jianzhnie
Description:

'''


def majorityElement(nums):
    n = len(nums)
    hashtable = {}
    for i in nums:
        if i not in hashtable:
            hashtable[i] = 1
        else:
            hashtable[i] += 1

    for key, value in hashtable.items():
        if value > (n // 2):
            return key
    return


if __name__ == '__main__':

    nums = [1, 1, 2]

    res = majorityElement(nums)
    print(res)
