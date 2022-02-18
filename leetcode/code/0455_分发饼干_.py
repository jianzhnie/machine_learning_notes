from typing import List


# 思路1：优先考虑胃饼干
def findContentChildren(g: List[int], s: List[int]) -> int:
    g.sort()  # 胃口
    s.sort()  # 饼干
    res = 0
    for i in range(len(s)):
        if res < len(g) and s[i] >= g[res]:  # 小饼干先喂饱小胃口
            res += 1
    return res
