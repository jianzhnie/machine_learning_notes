from typing import List


def intersection(nums1: List[int], nums2: List[int]) -> List[int]:
    return list(set(nums1) & set(nums2))  # 两个数组先变成集合，求交集后还原为数组


def union(nums1: List[int], nums2: List[int]) -> List[int]:
    return list(set(nums1 + nums2))  # 两个数组先变成集合，求交集后还原为数组


if __name__ == '__main__':
    nums1 = [1, 3, 2, 3, 1, 1]
    nums2 = [2, 4, 3, 2]
    res = intersection(nums1, nums2)
    print(res)
    res = union(nums1, nums2)
    print(res)
