'''
Author: jianzhnie
Date: 2022-02-11 17:34:45
LastEditTime: 2022-02-11 17:36:54
LastEditors: jianzhnie
Description:

'''


def findPeek(nums):
    n = len(nums)
    res = []
    if n < 2:
        return res
    else:
        for i in range(1, n - 1):
            if nums[i] > nums[i + 1] and nums[i] > nums[i - 1]:
                res.append(i)
    return res


if __name__ == '__main__':
    nums = [1, 3, 2, 1]
    res = findPeek(nums)
    print(res)
