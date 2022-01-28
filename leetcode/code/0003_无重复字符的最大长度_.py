class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        n = len(s)
        ans = 0
        longstr = ''
        for left_k in range(n):
            substring = ''
            right_k = left_k
            while right_k < n and s[right_k] not in substring:
                # 不断地移动右指针
                substring += s[right_k]
                right_k += 1
            # 第 left_k 到 right_k 个字符是一个极长的无重复字符子串
            ans = max(ans, len(substring))

            if len(substring) > len(longstr):
                longstr = substring
        return ans, longstr


if __name__ == '__main__':
    solution = Solution()
    input = 'abcbcaecbde'
    ans, longstr = solution.lengthOfLongestSubstring(input)
    print(ans, longstr)
