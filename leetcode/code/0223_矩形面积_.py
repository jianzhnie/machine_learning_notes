class Solution(object):
    def computeArea(self, ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
        """
        :type ax1: int
        :type ay1: int
        :type ax2: int
        :type ay2: int
        :type bx1: int
        :type by1: int
        :type bx2: int
        :type by2: int
        :rtype: int
        """
        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)

        sa = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
        sb = max(bx2 - bx1, 0) * max(by2 - by1, 0)
        s_iou = max(x2 - x1, 0) * max(y2 - y1, 0)

        res = sa + sb - s_iou
        return res


if __name__ == '__main__':
    matrix = [['1', '0', '1', '0', '0'], ['1', '0', '1', '1', '1'],
              ['1', '1', '1', '1', '1'], ['1', '0', '0', '1', '0']]

    solution = Solution()
    res = solution.computeArea(ax1=-3,
                               ay1=0,
                               ax2=3,
                               ay2=4,
                               bx1=0,
                               by1=-1,
                               bx2=9,
                               by2=2)
    print(res)
