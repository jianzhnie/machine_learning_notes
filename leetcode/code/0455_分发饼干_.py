from typing import List


# 思路1：优先考虑饼干
def findContentChildren(g: List[int], s: List[int]) -> int:
    g.sort()  # 胃口
    s.sort()  # 饼干
    j = 0
    for i in range(len(s)):
        if j < len(g) and s[i] >= g[j]:  # 小饼干先喂饱小胃口
            j += 1
    return j


# 思路2：优先考虑胃口
def findContentChildren2(g: List[int], s: List[int]) -> int:
    g.sort()  # 胃口
    s.sort()  # 饼干
    start, count = len(s) - 1, 0
    for i in range(len(g) - 1, -1, -1):  # 先喂饱大胃口
        if start >= 0 and g[i] <= s[start]:
            start -= 1
            count += 1
    return count
