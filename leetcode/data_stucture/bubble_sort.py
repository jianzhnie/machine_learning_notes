def BubbleSort(nums):
    n = len(nums)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if nums[i] > nums[j]:
                nums[i], nums[j] = nums[j], nums[i]

    return nums


def InsertSort(nums):
    # 从第二个位置，即下标为1的元素开始向前插入
    for i in range(1, len(nums)):
        # 从第i个元素开始向前比较，如果小于前一个元素，交换位置
        for j in range(i, 0, -1):
            if nums[j] < nums[j - 1]:
                nums[j], nums[j - 1] = nums[j - 1], nums[j]

    return nums


def QuickSort(nums, start, end):
    if start >= end:
        return
    mid = nums[start]
    low = start
    high = end
    while low < high:
        while low < high and nums[high] >= mid:
            high -= 1
        nums[low] = nums[high]
        while low < high and nums[low] < mid:
            low += 1
        nums[high] = nums[low]

    nums[low] = mid

    QuickSort(nums, start, low - 1)
    QuickSort(nums, low + 1, end)


if __name__ == '__main__':
    li = [54, 26, 26, 93, 17, 77, 31, 44, 55, 20]
    res = BubbleSort(li)
    print(res)

    nums = [54, 54, 26, 93, 17, 77, 31, 44, 55, 20]
    res = InsertSort(nums)
    print(res)

    QuickSort(nums, 0, len(nums) - 1)
    print(nums)
