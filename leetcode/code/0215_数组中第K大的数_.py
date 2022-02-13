def findKthLargest(nums, k):
    n = len(nums)
    left = 0
    right = n - 1
    target = n - k
    while True:
        index = partition(nums, left, right)
        if index == target:
            return nums[target]
        elif index < target:
            left = index + 1
        else:
            right = index - 1


#  对数组 nums 的子区间 [left..right] 执行 partition 操作，返回 nums[left] 排序以后应该在的位置
#  在遍历过程中保持循环不变量的定义：
#  nums[left + 1..j] < nums[left]
#  nums(j..i) >= nums[left]
def partition(nums, left, right):
    pivot = nums[left]
    j = left
    # j 的初值为 left，先右移，再交换，小于 pivot 的元素都被交换到前面
    for i in range(left + 1, right):
        if nums[i] < pivot:
            j += 1
            nums[i], nums[j] = nums[j], nums[i]
    # 在之前遍历的过程中，满足 nums[left + 1..j] < pivot，并且 nums(j..i) >= pivot
    nums[left], nums[j] = nums[j], nums[left]
    # 交换以后 nums[left..j - 1] < pivot, nums[j] = pivot, nums[j + 1..right] >= pivot

    return j
