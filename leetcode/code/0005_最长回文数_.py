class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s) < 2:
            return s
        n = len(s)
        total = 0
        for i in range(n - 1):
            for j in range(i, n):
                substr = s[i:j + 1]
                if substr == substr[::-1]:
                    if j - i + 1 > total:
                        total = j - i + 1
                        res = substr
        return res


if __name__ == '__main__':
    solution = Solution()
    input = 'abcbadecscdsdfghlkjhygtsdfkjhdfjhdfghsdfshas'
    input = 'bbbbbb'
    ans = solution.longestPalindrome(input)
    print(ans)
