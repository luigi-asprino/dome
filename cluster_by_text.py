import pandas as pd
from gensim import corpora
from gensim.matutils import corpus2csc
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from preprocessing.Tokenizer import SplitTokenizer
from utils.Utils import load_list_from_file, get_stopwords
import pickle
import numpy as np


class CorpusFiltered:

    def __init__(self, input_corpus, quantile):
        self.input_corpus = input_corpus
        self.quantile = quantile

    def filter_doc(self, doc):
        if len(doc) == 0:
            return doc

        threshold = np.quantile([e[1] for e in doc], self.quantile)
        return [e for e in doc if e[1] > threshold and e[0] > 0]

    def __iter__(self):
        count = 0
        for doc in self.input_corpus:
            if count % 10000 == 0:
                print(f"{count} document processed")
            count += 1
            yield self.filter_doc(doc)

    def __getitem__(self, item):
        return self.filter_doc(self.input_corpus[item])


token_number = 8
laundromat_corpus = "/Users/lgu/Dropbox/Backups/Corpus_lod"
tfidf_corpus_file = laundromat_corpus + "/tfidf_corpus"
dictionary_file = laundromat_corpus + "/dictionary"
#corpus_tfidf = corpora.MmCorpus(tfidf_corpus_file)
#dictionary = corpora.Dictionary.load(dictionary_file)
doc_ids = load_list_from_file(laundromat_corpus + "/doc_ids", token_number, extractid=True)
#print(doc_ids)
#print(len(doc_ids))
#exit()
#id2doc = {k: v for v, k in enumerate(doc_ids)}
#stop = get_stopwords("stopwords.txt")
#
#
# corpus_filtered = CorpusFiltered(corpus_tfidf, 0.9)
# print(corpus_tfidf[42])
# D = corpus_filtered[42]
# for w in D:
#     print(f"{dictionary[w[0]]} {w[1]}")


#words = set()
#for d in corpus_filtered:
#    for w in d:
#        words.add(w[0])

#print(len(words))

#exit()

#print(corpus_tfidf[0])

#X = corpus2csc(corpus_filtered, printprogress=True)
#pickle.dump(X, open(laundromat_corpus + "/corpus_csc.p", "wb"))
#exit()
X = pickle.load(open(laundromat_corpus + "/corpus_csc.p", "rb"))
print(X.shape)
X_t = X.transpose()
print(X_t.shape)
print(doc_ids[0])
k_means = "/Users/lgu/Desktop/K-means"
sse = {}
for k in [128, 256, 512]:
    print(f"Computing K-means with K={k}")
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0).fit(X_t)
    #kmeans = MiniBatchKMeans(n_clusters=64, random_state=0).fit(X_t)

    pickle.dump(kmeans, open(k_means + f"/k_means_{k}.p", "wb"))
    labels = kmeans.labels_
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    print(f"Number of clusters {k} Inertia: {kmeans.inertia_}")
    f = open(k_means + f"/clusters_{k}.txt", "w")
    for idx, l in enumerate(kmeans.labels_):
        f.write(f"{doc_ids[idx]}\t{l}\n")
    f.close()
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

#print(len(X))

#kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
#pickle.dump(kmeans, open(laundromat_corpus + "/kmeans.p", "wb"))
#pickle.load( open(laundromat_corpus + "/kmeans.p", "rb"))
#exit()

#print(f"Computed {len(kmeans.labels_)}")
#f = open(f"/Users/lgu/Desktop/NOTime/EKR/Corpus_lod_4/clusters_text.txt", "w")
#for idx, l in enumerate(kmeans.labels_):
#    #print(f"{doc_ids[idx]}\t{l}\n")
#    #print(f"{doc_ids[idx]}\t{l}\n")
#    #if idx < len(doc_ids):
#    #    f.write(f"{doc_ids[idx]}\t{l}\n")
#    #else:
#    #    print(idx)
#    pass
#f.close()




# sse = {}
# for k in [2, 4, 8, 16, 32, 64, 80, 96, 112, 128, 256, 512]:
#     kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
#     labels = kmeans.labels_
#     #sil_coeff = silhouette_score(X, labels, metric='euclidean')
#     sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
#     print(f"Number of clusters {k} Inertia: {kmeans.inertia_}")
#     f = open(f"/Users/lgu/Desktop/NOTime/EKR/Corpus_lod_4/clusters_{k}.txt", "w")
#     for idx, l in enumerate(kmeans.labels_):
#         f.write(f"{sources[0][idx]}\t{l}\n")
#     f.close()
# plt.figure()
# plt.plot(list(sse.keys()), list(sse.values()))
# plt.xlabel("Number of cluster")
# plt.ylabel("SSE")
# plt.show()