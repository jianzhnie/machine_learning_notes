class Solution:
    def climbStairs(self, n: int) -> int:
        res = [0 for i in range(45)]
        res[1] = 1
        res[2] = 2
        if n >= 3:
            for i in range(3, n + 1):
                res[i] = res[i - 1] + res[i - 2]
        return res[n]


if __name__ == '__main__':
    res = Solution().climbStairs(2)
    print(res)
