import warnings
from utils.Utils import load_list_from_file

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
from utils.Utils import get_stopwords
import logging
import numpy as np

np.random.seed(0)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.getLogger("gensim").setLevel(logging.ERROR)
logging.getLogger("polyglot").setLevel(logging.ERROR)
trace = logging.getLogger("trace")
trace.setLevel(logging.INFO)
timings = logging.getLogger("timings")
timings.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
from gensim import corpora
from scipy.sparse import csr_matrix
import pickle


def doc_to_csr_matrix(doc, dic_id_to_feature_id, features, binary=True):
    col = np.array([dic_id_to_feature_id[w[0]] for w in doc if w[0] in dic_id_to_feature_id])
    row = np.zeros((len(col),), dtype=int)
    # print(len(col))
    if binary:
        data = np.ones((len(col),), dtype=int)
    else:
        data = np.array([w[1] for w in doc if w[0] in dic_id_to_feature_id])
    return csr_matrix((data, (row, col)), shape=(1, len(features)))

token_number = 8
laundromat_corpus = "/Users/lgu/Desktop/NOTime/EKR/Corpus_lod_4"
# laundromat_corpus = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/Corpus_lov"
corpus_resampled_folder = "/Users/lgu/Desktop/NOTime/EKR/experiments/lov_benchmark_no_hierarchy_stratified/mlsmote_iterative"
classifier_folder = corpus_resampled_folder + "/MLPClassifier"

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

cv = pickle.load(open(corpus_resampled_folder + "/cv.p", "rb"))
mlb = pickle.load(open(corpus_resampled_folder + "/mlb.p", "rb"))
clf = pickle.load(open(classifier_folder + "/clf.p", "rb"))

features = cv.get_feature_names()
dic_id_to_feature_id = {dictionary.token2id[feature]: feature_id for feature_id, feature in enumerate(features) if
                        feature in dictionary.token2id}

for doc in corpus_tfidf[50000:50100]:
    print(" ".join([f"{id_to_dictionary_token[w[0]]} {w[1]}" for w in sorted(doc, key=lambda item: item[1], reverse=True)[:10] if w[0] > 0 ]))
    print(" ".join(
        [f"{id_to_dictionary_token[w[0]]} {w[1]}" for w in sorted(doc, key=lambda item: item[1], reverse=True)[:10] if
         w[0] > 0 and w[0] in dic_id_to_feature_id]))
    d = doc_to_csr_matrix(doc, dic_id_to_feature_id, features)
    p = clf.predict(d)
    print(mlb.inverse_transform(p))
