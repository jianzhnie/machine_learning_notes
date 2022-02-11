class Solution:
    def climbStairs(self, n: int) -> int:
        if n < 3:
            res = [0, 1, 2]
        if n >= 3:
            res = [0 for i in range(n + 1)]
            res[:3] = [0, 1, 2]
            for i in range(3, n + 1):
                res[i] = res[i - 1] + res[i - 2]
        return res[n]

    def climbStairs2(self, n: int) -> int:
        p, q, r = 0, 0, 1
        for i in range(1, n + 1):
            p = q
            q = r
            r = p + q
        return r


if __name__ == '__main__':
    solution = Solution()
    res = solution.climbStairs(1000)
    print(res)
