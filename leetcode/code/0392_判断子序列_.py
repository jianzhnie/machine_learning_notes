def subString(source, target):
    m = len(source)
    n = len(target)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if source[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = dp[i][j - 1]

    if dp[m][n] == m:
        return True
    return False


if __name__ == '__main__':
    s = 'abc'
    t = 'ahbgdc'
    res = subString(s, t)
    print(res)
