import warnings

from sklearn.feature_extraction.text import TfidfTransformer

from domainannotators.DocumentDomainAnnotators import SimpleDocumentAnnotator, DocumentAnnotatorAggregationStrategy
from domainannotators.WordAnnotators import RocksDBDomainDisambiguator, AggregationStrategy
from utils.Utils import load_list_from_file, load_map_from_file
from utils.ml_utils import DomainTransformer
import simplemma
langdata = simplemma.load_data('en', 'it', 'de', 'es', 'fr', 'nl', 'ru')

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
from utils.Utils import get_stopwords
import logging
import numpy as np
from sklearn.pipeline import Pipeline

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


def doc_to_csr_matrix(doc, dic_id_to_feature_id, features, id_to_token, binary=True, da=None):
    print(len(dic_id_to_feature_id))
    for w in doc:
        if w[0] in dic_id_to_feature_id:
            continue
        if w[0] not in id_to_token:
            continue
        lemma = simplemma.lemmatize(id_to_token[w[0]], langdata)
        if lemma in features:
            dic_id_to_feature_id[w[0]] = features.index(lemma)
    print(len(dic_id_to_feature_id))

    col = np.array([dic_id_to_feature_id[w[0]] for w in doc if w[0] in dic_id_to_feature_id])
    row = np.zeros((len(col),), dtype=int)
    # print(len(col))
    if binary:
        data = np.ones((len(col),), dtype=int)
    else:
        data = np.array([w[1] for w in doc if w[0] in dic_id_to_feature_id])

    if da is None:
        return csr_matrix((data, (row, col)), shape=(1, len(features)))
    else:
        return da.transform(csr_matrix((data, (row, col)), shape=(1, len(features))))

token_number = 8
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

# Load resources
id_to_domain = load_map_from_file(id_to_domain_file)
domain_to_id = {k: v for v, k in id_to_domain.items()}

lemma_to_domain_dbs = ["/Users/lgu/workspace/ekr/dome/resources/20211126_input_unified/lemma_to_domain_wn",
                        "wn",
                        "/Users/lgu/workspace/ekr/dome/resources/20211126_input_unified/lemma_to_domain_bn",
                        "bn"]
lemma_dbs = []
i = 0
while i < len(lemma_to_domain_dbs):
    print(f"Loading DB from path {lemma_to_domain_dbs[i]} {lemma_to_domain_dbs[i + 1]}")
    db = RocksDBDomainDisambiguator(lemma_to_domain_dbs[i], lemma_to_domain_dbs[i + 1], id_to_domain,
                                        hierarchy={}, strategy=AggregationStrategy.MAX)
    lemma_dbs.append(db)
    i = i + 2

da = SimpleDocumentAnnotator(None, id_to_domain, lemma_dbs, [],
    strategy=DocumentAnnotatorAggregationStrategy.SUM_WORD_MAX)

X = pickle.load(open(corpus_resampled_folder + "/X.p", "rb"))
y = pickle.load(open(corpus_resampled_folder + "/y.p", "rb"))
cv = pickle.load(open(corpus_resampled_folder + "/cv.p", "rb"))
tfidf = TfidfTransformer()
domaintransformer = DomainTransformer(da, cv)

transformers_pipeline = [("vect", cv)]

domaintransformer.fit([])

#if use_tfidf:
#    transformers_pipeline.append(("tfidf", tfidf))

#if use_domain_annotator:
#    transformers_pipeline.append(("da", domaintransformer))

#pipeline = Pipeline(transformers_pipeline)

#pipeline.fit_transform(X)

mlb = pickle.load(open(corpus_resampled_folder + "/mlb.p", "rb"))
clf = pickle.load(open(classifier_folder + "/clf.p", "rb"))

features = cv.get_feature_names()
feature_to_id = {feature : feature_id for feature_id, feature in enumerate(features)}
dic_id_to_feature_id = {dictionary.token2id[feature]: feature_id for feature_id, feature in enumerate(features) if
                        feature in dictionary.token2id}

# print(len(dic_id_to_feature_id))
# for token_id, token in id_to_dictionary_token.items():
#     if token_id in dic_id_to_feature_id:
#         continue
#     lemma = simplemma.lemmatize(token, langdata)
#     if lemma in feature_to_id:
#         dic_id_to_feature_id[token_id] = feature_to_id[lemma]
# print(len(dic_id_to_feature_id))

for doc in corpus_tfidf[50000:50100]:
    d = doc_to_csr_matrix(doc, dic_id_to_feature_id, features, id_to_dictionary_token, binary=False,
                          da=domaintransformer)
    print(" ".join([f"{id_to_dictionary_token[w[0]]} {w[1]}" for w in sorted(doc, key=lambda item: item[1], reverse=True)[:10] if w[0] > 0 ]))
    print(" ".join(
        [f"{id_to_dictionary_token[w[0]]} {w[1]}" for w in sorted(doc, key=lambda item: item[1], reverse=True)[:10] if
         w[0] > 0 and w[0] in dic_id_to_feature_id]))
    p = clf.predict(d)
    print(p)
    print(mlb.inverse_transform(p))
