'''
Author: jianzhnie
Date: 2022-02-10 11:28:35
LastEditTime: 2022-02-10 11:28:35
LastEditors: jianzhnie
Description:

'''


def mergeArray(nums1, nums2):
    m = len(nums1)
    n = len(nums2)
    i, j = 0, 0
    res = []
    while i < m and j < n:
        print(i, j)
        if nums1[i] <= nums2[j]:
            res.append(nums1[i])
            i += 1
        else:
            res.append(nums2[j])
            j += 1
    while i < m:
        res.append(nums1[i])
        i += 1
    while j < n:
        res.append(nums2[j])
        j += 1

    return res
