'''
Author: jianzhnie
Date: 2021-12-22 09:46:16
LastEditTime: 2021-12-22 09:46:16
LastEditors: jianzhnie
Description:

'''


def iou(bbox1, bbox2):
    """
    bbox1: [x1, y1, x2, y2]
    bbox2: [x1, y1, x2, y2]
    """
    xx1 = max(bbox1[0], bbox2[0])
    yy1 = max(bbox1[1], bbox2[1])
    xx2 = min(bbox1[2], bbox2[2])
    yy2 = min(bbox1[3], bbox2[3])

    interArea = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    iou = interArea / float(bbox1_area + bbox2_area - interArea)

    return iou
