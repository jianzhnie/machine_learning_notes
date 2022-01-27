'''
Author: jianzhnie
Date: 2022-01-19 10:14:13
LastEditTime: 2022-01-27 17:25:24
LastEditors: jianzhnie
Description:

'''

from typing import List


class Algorithms():
    def twosum(self, numList, target):
        res_dict = {}
        for idx, val in enumerate(numList):
            if target - val not in res_dict:
                res_dict[val] = idx
            else:
                return [res_dict[target - val], idx]

    def palindrome(self, strs):
        res_len = 0
        res_str = strs[0]
        for i in range(len(strs) - 1):
            for j in range(i, len(strs)):
                substr = strs[i:j]
                sub_len = len(substr)
                if substr == substr[::-1]:
                    if sub_len > res_len:
                        res_len = sub_len
                        res_str = substr

        return res_str

    def threesum_(self, numList):

        res = []
        N = len(numList)

        numList.sort()

        for i in range(N):
            left = i + 1
            right = N - 1

            if numList[i] > 0:
                break
            if i >= 1 and numList == numList[i - 1]:
                continue
            while left < right:
                total = numList[left] + numList[i] + numList[right]

                if total > 0:
                    right -= 1
                elif total < 0:
                    left += 1
                else:
                    res.append([numList[left], numList[i], numList[right]])
                    while left != right and numList[left] == numList[left + 1]:
                        left += 1
                    while left != right and numList[right] == numList[right -
                                                                      1]:
                        right -= 1
                    left += 1
                    right -= 1

        return res

    def threesum(self, numList, target):
        hashmap = {}
        for n in numList:
            if n not in hashmap:
                hashmap[n] = 1
            else:
                hashmap[n] += 1
        res = set()
        N = len(numList)
        for i in range(N):
            for j in range(i + 1, N - 1):
                val = target - (numList[i] + numList[j])
                if val in hashmap:
                    count = (numList[i] == val) + (numList[j] == val)

                    if hashmap[val] > count:
                        res.add(tuple(sorted([numList[i], numList[j], val])))

                else:
                    continue
        return res

    def foursum(self, numList, target):
        hashmap = {}
        for n in numList:
            if n not in hashmap:
                hashmap[n] = 1
            else:
                hashmap[n] += 1
        res = set()
        N = len(numList)
        for i in range(N):
            for j in range(i + 1, N - 1):
                for k in range(j + 1, N - 2):
                    val = target - (numList[i] + numList[j] + numList[k])
                    if val in hashmap:
                        count = (numList[i] == val) + (numList[j] == val) + (
                            numList[k] == val)

                        if hashmap[val] > count:
                            res.add(
                                tuple(
                                    sorted([
                                        numList[i], numList[j], numList[k], val
                                    ])))

                    else:
                        continue

        return res

    def removeElement(self, numList, target):
        fast = slow = 0

        while fast < len(numList):

            if numList[fast] != target:
                numList[slow] = numList[fast]
                slow += 1

            # 当 fast 指针遇到要删除的元素时停止赋值
            # slow 指针停止移动, fast 指针继续前进
            fast += 1

        return slow

    def nextPermutation(self, nums: List[int]) -> None:
        """Do not return anything, modify nums in-place instead."""
        for i in range(len(nums) - 1, -1, -1):
            for j in range(len(nums) - 1, i, -1):
                if nums[j] > nums[i]:
                    nums[j], nums[i] = nums[i], nums[j]
                    nums[i + 1:len(nums)] = sorted(nums[i + 1:len(nums)])
                    return nums
        return nums.sort()

    def binarySearch(self, numList, target):
        left, right, = 0, len(numList) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if numList[mid] > target:
                right = mid - 1
            elif numList[mid] < target:
                left = mid + 1
            else:
                return mid
        return -1

    def searchRange(self, nums: List[int], target: int) -> List[int]:

        index = self.binarySearch(nums, target)
        if index == -1:
            return [-1, -1]  # nums 中不存在 target，直接返回 {-1, -1}
        # nums 中存在 targe，则左右滑动指针，来找到符合题意的区间
        left, right = index, index
        # 向左滑动，找左边界
        while left - 1 >= 0 and nums[left - 1] == target:
            left -= 1
        # 向右滑动，找右边界
        while right + 1 < len(nums) and nums[right + 1] == target:
            right += 1
        return [left, right]

    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1

        while left <= right:
            middle = (left + right) // 2

            if nums[middle] < target:
                left = middle + 1
            elif nums[middle] > target:
                right = middle - 1
            else:
                return middle
        return right + 1

    def trapTwoNiddle(self, height: List[int]) -> int:
        res = 0
        for i in range(len(height)):
            if i == 0 or i == len(height) - 1:
                continue
            lHight = height[i - 1]
            rHight = height[i + 1]
            for j in range(i - 1):
                if height[j] > lHight:
                    lHight = height[j]
            for k in range(i + 2, len(height)):
                if height[k] > rHight:
                    rHight = height[k]
            res1 = min(lHight, rHight) - height[i]
            if res1 > 0:
                res += res1
        return res

    def trapDP(self, height: List[int]) -> int:
        leftheight, rightheight = [0] * len(height), [0] * len(height)

        leftheight[0] = height[0]
        for i in range(1, len(height)):
            leftheight[i] = max(leftheight[i - 1], height[i])
        rightheight[-1] = height[-1]
        for i in range(len(height) - 2, -1, -1):
            rightheight[i] = max(rightheight[i + 1], height[i])

        result = 0
        for i in range(0, len(height)):
            summ = min(leftheight[i], rightheight[i]) - height[i]
            result += summ
        return result

    def jump(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return 0
        ans = 0
        curDistance = 0
        nextDistance = 0
        for i in range(len(nums)):
            nextDistance = max(i + nums[i], nextDistance)
            if i == curDistance:
                if curDistance != len(nums) - 1:
                    ans += 1
                    curDistance = nextDistance
                    if nextDistance >= len(nums) - 1:
                        break
        return ans


class SumCombined:
    def __init__(self):
        self.path = []
        self.paths = []

    def combinationSum(self, candidates: List[int],
                       target: int) -> List[List[int]]:
        """因为本题没有组合数量限制，所以只要元素总和大于target就算结束."""
        self.path.clear()
        self.paths.clear()

        # 为了剪枝需要提前进行排序
        candidates.sort()
        self.backtracking(candidates, target, 0, 0)
        return self.paths

    def backtracking(self, candidates: List[int], target: int, sum: int,
                     start_index: int) -> None:
        # Base Case
        # 因为是shallow copy，所以不能直接传入self.path
        if sum == target:
            self.paths.append(self.path[:])

            return
        # 单层递归逻辑
        # 如果本层 sum + condidates[i] > target，就提前结束遍历，剪枝
        for i in range(start_index, len(candidates)):
            if sum + candidates[i] > target:
                return
            sum += candidates[i]
            self.path.append(candidates[i])
            self.backtracking(candidates, target, sum, i)  # 因为无限制重复选取，所以不是i-1
            sum -= candidates[i]  # 回溯
            self.path.pop()  # 回溯


if __name__ == '__main__':

    algo = Algorithms()
    numList = [2, -1, 1, 3, -2, 4, 5, -3, 0, 2]

    res = algo.threesum(numList, 0)
    print(res)

    res = algo.foursum(numList, target=0)
    print(res)

    res = algo.nextPermutation([2, 3, 4, 1])
    print(res)
