class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows < 2:
            return s
        res = ['' for _ in range(numRows)]
        period = numRows * 2 - 2
        for i in range(len(s)):
            mod = i % period
            if mod < numRows:
                res[mod] += s[i]
            else:
                res[period - mod] += s[i]
        return ''.join(res)


if __name__ == '__main__':
    solution = Solution()
    input = 'PAYPALISHIRING'
    ans = solution.convert(input, 3)
    print(ans)
