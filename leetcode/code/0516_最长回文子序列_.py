def longestPalindromeSubseq(s):
    res = 0
    for i in range(len(s)):
        substr = s[:i]
        if substr == substr[::-1]:
            res = max(res, i + 1)
    return res
