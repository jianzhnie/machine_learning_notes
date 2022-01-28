import pandas as pd


def readtxt(file):
    hashtable = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            hashtable.append(line.split('\t'))
    return hashtable


def concate_df(filenames):
    frames = []
    for filename in filenames:
        data = pd.read_csv(filename, sep='\t', head=True)

        frames.append(data)
    results = pd.concate(frames)

    return results
