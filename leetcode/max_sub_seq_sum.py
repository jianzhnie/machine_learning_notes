from cmath import inf


def intSubArray(nums):

    res = float(-inf)
    count = 0
    for i in range(0, len(nums)):
        count = 0
        for j in range(i, len(nums)):
            count += nums[j]

            if count > res:
                res = count

    return res


def intSubArray1(nums):

    res = float(-inf)
    count = 0
    for i in range(0, len(nums)):
        count += nums[i]
        if count > res:
            res = count
        if count <= 0:
            count = 0
    return res


arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]

ans = intSubArray(arr)
print(ans)

ans = intSubArray1(arr)
print(ans)
