def eraseOverlapIntervals(intervals):
    if len(intervals) == 0:
        return 0
    intervals.sort(key=lambda x: x[1])
    count = 1  # 记录非交叉区间的个数
    end = intervals[0][1]  # 记录区间分割点
    for i in range(1, len(intervals)):
        if end <= intervals[i][0]:
            count += 1
            end = intervals[i][1]
    return len(intervals) - count


def eraseOverlapIntervals2(intervals):

    intervals.sort(key=lambda x: x[1])
    count = 0
    end = intervals[0][1]
    for i in range(1, len(intervals)):
        if end > intervals[i][0]:
            count += 1
            end = intervals[i][1]
    return count


if __name__ == '__main__':
    intervals = [[1, 2], [2, 3], [3, 4], [1, 3]]
    res = eraseOverlapIntervals(intervals)
    print(res)
    res = eraseOverlapIntervals2(intervals)
    print(res)
