'''
Author: jianzhnie
Date: 2022-02-10 16:08:44
LastEditTime: 2022-02-10 16:11:27
LastEditors: jianzhnie
Description:

'''

from typing import List


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            # 如果列表为空，或者当前区间与上一区间不重合，直接添加
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                # 否则的话，我们就可以与上一区间进行合并
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged


if __name__ == '__main__':
    solution = Solution()
    intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
    res = solution.merge(intervals)
    print(res)
