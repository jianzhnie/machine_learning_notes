from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    "John likes to watch movies, Mary likes movies too",
    "John also likes to watch football games",
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names())
print(X.toarray())


 
from sklearn.feature_extraction.text import TfidfTransformer
#类调用
transformer = TfidfTransformer()

#将词频矩阵X统计成TF-IDF值
tfidf = transformer.fit_transform(X)
#查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
print(tfidf.toarray())


from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())

print(X)
print(X.toarray())


import numpy as np

def euclidean_dist_normalize(list_a):
    result = sum([i**2 for i in list_a]) ** 0.5
    return [i/result for i in list_a]
    
def ele_multi(list1,list2):
    return [i*j for i,j in zip(list1,list2)]

corpus = np.array([[3, 0, 1],
                   [2, 0, 0],
                   [3, 0, 0],
                   [4, 0, 0],
                   [3, 2, 0],
                   [3, 0, 2]])

inverse_doc_freq = np.log(corpus.shape[0]/sum((corpus > 0)*1) +1)

tf_idf = np.array([euclidean_dist_normalize(ele_multi(doc, inverse_doc_freq)) for doc in corpus])
print(tf_idf)


