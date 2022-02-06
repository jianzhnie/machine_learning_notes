from typing import List


class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        hashmap = {}
        for n in nums:
            if n not in hashmap:
                hashmap[n] = 1
            else:
                hashmap[n] += 1
        res = set()
        N = len(nums)
        for i in range(N - 2):
            for j in range(i + 1, N - 1):
                for k in range(j + 1, N):
                    val = target - (nums[i] + nums[j] + nums[k])
                    print(i, j, k)
                    if val in hashmap:
                        count = (nums[i] == val) + (nums[j] == val) + (nums[k]
                                                                       == val)

                        if hashmap[val] > count:
                            res.add(
                                tuple(sorted([nums[i], nums[j], nums[k],
                                              val])))

                        else:
                            continue
        return list(res)


if __name__ == '__main__':
    solution = Solution()
    input = [1, 0, -1, 0, -2, 2]
    ans = solution.fourSum(input, 0)
    print(ans)
