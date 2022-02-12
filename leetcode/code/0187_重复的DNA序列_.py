'''
Author: jianzhnie
Date: 2022-02-12 13:56:37
LastEditTime: 2022-02-12 17:50:05
LastEditors: jianzhnie
Description:

'''

from collections import defaultdict


def findRepeatedDNAseq(sequence, length):

    res = []
    cnt = defaultdict(int)
    n = len(sequence)
    for i in range(n - length + 1):
        subseq = sequence[i:i + length]
        cnt[subseq] += 1
        if cnt[subseq] == 2:
            res.append(subseq)
    return res
