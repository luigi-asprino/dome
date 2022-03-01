import pandas as pd
from gensim import corpora
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing.Tokenizer import SplitTokenizer
from utils.Utils import load_list_from_file, get_stopwords


token_number = 8
input = "/Users/lgu/Desktop/NOTime/EKR/Corpus_lod_4/sources.txt"
laundromat_corpus = "/Users/lgu/Desktop/NOTime/EKR/Corpus_lod_4"
# laundromat_corpus = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/Corpus_lov"
corpus_resampled_folder = "/Users/lgu/Desktop/NOTime/EKR/experiments/lov_benchmark_tf_da/mlsmote_iterative"
classifier_folder = corpus_resampled_folder + "/MLPClassifier"
id_to_domain_file = "/Users/lgu/workspace/ekr/dome/resources/20211126_input_unified/id2domain.tsv"
use_tfidf = True
use_domain_annotator = True

tfidf_corpus_file = laundromat_corpus + "/tfidf_corpus"
dictionary_file = laundromat_corpus + "/dictionary"
corpus_tfidf = corpora.MmCorpus(tfidf_corpus_file)
print(f"tf-idf corpus loaded length: {len(corpus_tfidf)}")
print(f"loading dictionary from {dictionary_file}")
dictionary = corpora.Dictionary.load(dictionary_file)
id_to_dictionary_token = {v: k for k, v in dictionary.token2id.items()}
print("dictionary loaded")
doc_ids = load_list_from_file(laundromat_corpus + "/doc_ids", token_number, extractid=True)
id2doc = {k: v for v, k in enumerate(doc_ids)}
stop = get_stopwords("stopwords.txt")

#print(corpus_tfidf[0])

#X = corpus2csc(corpus_tfidf)
#exit()

sources = pd.read_csv(input, sep='\t', header=None, usecols=[0, 1])

X = sources[0][1:]

stopwords = ["lodlaundromat", "org", "http", "https", "purl", "net", "ttl", "n3", "rdf", "html", "nq", "com", "owl",
             "nt"]

cv = CountVectorizer(lowercase=True, stop_words=stopwords, tokenizer=SplitTokenizer(), binary=True)
X = cv.fit_transform(X)
#print(cv.get_feature_names())

#kmeans = KMeans(n_clusters=1000, random_state=0).fit(X)




sse = {}
for k in [2, 4, 8, 16, 32, 64, 80, 96, 112, 128, 256, 512]:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = kmeans.labels_
    #sil_coeff = silhouette_score(X, labels, metric='euclidean')
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    print(f"Number of clusters {k} Inertia: {kmeans.inertia_}")
    f = open(f"/Users/lgu/Desktop/NOTime/EKR/Corpus_lod_4/clusters_{k}.txt", "w")
    for idx, l in enumerate(kmeans.labels_):
        f.write(f"{sources[0][idx]}\t{l}\n")
    f.close()
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()