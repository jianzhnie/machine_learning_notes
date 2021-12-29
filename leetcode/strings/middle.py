'''
Author: jianzhnie
Date: 2021-12-22 09:47:56
LastEditTime: 2021-12-22 11:23:19
LastEditors: jianzhnie
Description:

'''

from itertools import groupby


def FindRepeatWords(sentence):

    str_lst = sentence.lower().split(' ')
    res_list = [
        name for name, group in groupby(str_lst) if len(list(group)) > 1
    ]
    res_len = len(res_list)
    return res_len


def FindBestMatchUser(user_dict):
    return min(user_dict, key=lambda x: (abs(x - user_dict[x]), x))


def FindNumsAppearOnce(array):
    re = []
    for x in array:
        if x not in re:
            re.append(x)
        else:
            re.remove(x)
    return sorted(re)


def FindKLeastNumber(array, k):
    if len(array) < k:
        return []
    else:
        array.sort()
        return array[:k]


if __name__ == '__main__':
    sentence = 'dog dog cat cat dog dog dog'
    res = FindRepeatWords(sentence)
    print(res)

    users = {1: 10000, 10: 11, 100: 100}
    res = FindBestMatchUser(users)
    print(res)

    array = [1, 4, 5, 6, 6, 3, 1]
    res = FindNumsAppearOnce(array)
    print(res)
