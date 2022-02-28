from typing import List


def nextGreaterElement(nums1: List[int], nums2: List[int]) -> List[int]:
    result = [-1] * len(nums1)
    stack = [0]
    for i in range(1, len(nums2)):
        # 情况一情况二
        if nums2[i] <= nums2[stack[-1]]:
            stack.append(i)
        # 情况三
        else:
            while len(stack) != 0 and nums2[i] > nums2[stack[-1]]:
                if nums2[stack[-1]] in nums1:
                    index = nums1.index(nums2[stack[-1]])
                    result[index] = nums2[i]
                stack.pop()
            stack.append(i)
    return result


def nextGrateElement2(nums1, nums2):
    n = len(nums1)
    result = [-1] * n
    for i in range(n):
        subnums = nums2[i + 1:]
        for num in subnums:
            if num > nums1[i]:
                result[i] = num
                break
    return result
