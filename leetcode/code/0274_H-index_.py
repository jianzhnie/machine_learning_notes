'''
Author: jianzhnie
Date: 2022-02-14 08:59:46
LastEditTime: 2022-02-14 09:33:21
LastEditors: jianzhnie
Description:

'''
from typing import List


class Solution:
    """h 指数的定义：h 代表“高引用次数”（high citations）， 一名科研人员的 h 指数是指他（她）的 （n 篇论文中）总共.

    有 h 篇论文分别被引用了至少 h 次。且其余的 n - h 篇论文
    每篇被引用次数 不超过 h 次。
    """
    def hIndex(self, citations: List[int]) -> int:
        sorted_citation = sorted(citations, reverse=True)
        h = 0
        i = 0
        n = len(citations)
        while i < n and sorted_citation[i] > h:
            h += 1
            i += 1
        return h

    def hIndex2(self, citations: List[int]) -> int:
        n = len(citations)
        left = 0
        right = n - 1
        while left <= right:
            mid = left + (right - left) // 2
            if citations[mid] >= n - mid:
                right = mid - 1
            else:
                left = mid + 1
        return n - left


if __name__ == '__main__':
    citation = [0, 1, 3, 5, 6]
    solution = Solution()
    res = solution.hIndex2(citation)
    print(res)
