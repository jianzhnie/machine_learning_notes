'''
Author: jianzhnie
Date: 2022-02-15 18:28:39
LastEditTime: 2022-02-16 11:45:08
LastEditors: jianzhnie
Description:

'''


def topKFrequent(nums, k=3):
    hashtable = {}
    for num in nums:
        if num not in hashtable:
            hashtable[num] = 1
        else:
            hashtable[num] += 1

    sorted_nums = sorted(hashtable.items(), key=lambda x: x[1], reverse=True)

    res = [key for (key, val) in sorted_nums[:k]]
    return res


if __name__ == '__main__':
    nums = [1, 3, 4, 2, 1, 3, 5, 6, 4, 4, 3, 3, 3, 32, 4, 4, 4, 2]
    res = topKFrequent(nums)
    print(res)
