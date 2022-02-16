'''
Author: jianzhnie
Date: 2022-02-15 18:22:10
LastEditTime: 2022-02-15 18:23:12
LastEditors: jianzhnie
Description:

'''

# 从 X-Y 平面上的点 (0,0) 开始，
# 先向北移动 distance[0] 米，
# 然后向西移动 distance[1] 米，
# 向南移动 distance[2] 米，
# 向东移动 distance[3] 米，持续移动。
# 也就是说，每次移动后你的方位会发生逆时针变化。
# 判断你所经过的路径是否相交。如果相交，返回 true ；否则，返回 false


def IfCross(distance):
    dx1, dy1, dx2, dy2 = distance
    endx = dx1 - dx2
    endy = dy1 - dy2
    if endx <= 0 and endy >= 0:
        return True
    else:
        return False


if __name__ == '__main__':
    nums = [4, 2, 1, 3]
    res = IfCross(nums)
    print(res)
