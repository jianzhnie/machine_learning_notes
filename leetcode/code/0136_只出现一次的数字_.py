'''
Author: jianzhnie
Date: 2022-02-11 12:04:26
LastEditTime: 2022-02-11 12:13:34
LastEditors: jianzhnie
Description:

'''


def checkOnes(nums, target):
    hashtable = {}
    for number in nums:
        if number not in hashtable:
            hashtable[number] = 1
        else:
            hashtable[number] += 1

    for key, value in hashtable.items():
        if value == target:
            return key
    return None


if __name__ == '__main__':

    nums = [1, 2, 3, 4, 1, 1, 3, 4, 4, 3, 2, 1, 4, 5]
    res = checkOnes(nums, 4)
    print(res)
