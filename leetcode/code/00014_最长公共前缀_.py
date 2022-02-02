from typing import List


class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        strslen = [len(s) for s in strs]
        min_l = min(strslen)
        commonprefix = ''
        for i in range(min_l):
            substrlist = [s[:i + 1] for s in strs]
            if (len(set(substrlist)) <= 1):
                commonprefix = substrlist[0]

        return commonprefix


if __name__ == '__main__':
    solution = Solution()
    input = ['cir']
    ans = solution.longestCommonPrefix(input)
    print(ans)
