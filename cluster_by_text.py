import pandas as pd
from gensim import corpora
from gensim.matutils import corpus2csc
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from preprocessing.Tokenizer import SplitTokenizer
from utils.Utils import load_list_from_file, get_stopwords
import pickle
import numpy as np

token_number = 8
laundromat_corpus = "/Users/lgu/Desktop/NOTime/EKR/Corpus_lod_4"

tfidf_corpus_file = laundromat_corpus + "/tfidf_corpus"
dictionary_file = laundromat_corpus + "/dictionary"
corpus_tfidf = corpora.MmCorpus(tfidf_corpus_file)
dictionary = corpora.Dictionary.load(dictionary_file)
doc_ids = load_list_from_file(laundromat_corpus + "/doc_ids", token_number, extractid=True)
id2doc = {k: v for v, k in enumerate(doc_ids)}
stop = get_stopwords("stopwords.txt")

#print(corpus_tfidf[0])

X = corpus2csc(corpus_tfidf, printprogress=True)
#pickle.dump(X, open(laundromat_corpus + "corpus_csc.p", "wb"))
X = pickle.load(open(laundromat_corpus + "corpus_csc.p", "rb"))

print(len(X))

kmeans = KMeans(n_clusters=10, random_state=0).fit(X)

print("Computed")
f = open(f"/Users/lgu/Desktop/NOTime/EKR/Corpus_lod_4/clusters_text.txt", "w")
for idx, l in enumerate(kmeans.labels_):
    #print(f"{doc_ids[idx]}\t{l}\n")
    #print(f"{doc_ids[idx]}\t{l}\n")
    f.write(f"{doc_ids[idx]}\t{l}\n")
f.close()




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