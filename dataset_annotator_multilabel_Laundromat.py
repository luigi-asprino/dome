import os
from utils.Utils import load_map_from_file, load_list_from_file, load_vectors_from_file
import bz2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
from sklearn.ensemble import RandomForestClassifier
from utils.Utils import get_stopwords
import logging
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
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
from sklearn.multioutput import MultiOutputClassifier
from gensim import corpora
from preprocessing.Tokenizer import LemmaTokenizer
from scipy.sparse import csr_matrix

def doc_to_csr_matrix(doc, dic_id_to_feature_id, features, binary=True):
    col = np.array([dic_id_to_feature_id[w[0]] for w in doc if w[0] in dic_id_to_feature_id])
    row = np.zeros((len(col),), dtype=int)
    #print(len(col))
    if binary:
        data = np.ones((len(col),), dtype=int)
    else:
        data = np.array([w[1] for w in doc if w[0] in dic_id_to_feature_id])
    return csr_matrix((data, (row, col)), shape=(1, len(features)))

id_to_domain = "/Users/lgu/workspace/ekr/dome/resources/20211126_input_unified/id2domain.tsv"
input_folder_corpus = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/Corpus_lov"
token_number = 8
uri_to_doc_id_file = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/output/index.tsv"
gold_standard = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/LOV_KD_annotations.tsv"
hierarchy_file = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/KD_hierarchy.tsv"
laundromat_corpus = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/Corpus_lov"


id_to_domain = load_map_from_file(id_to_domain)
domain_to_id = {k: v for v, k in id_to_domain.items()}

print(id_to_domain)

doc_ids = load_list_from_file(input_folder_corpus + "/doc_ids", token_number, extractid=True)
id2doc = {k: v for v, k in enumerate(doc_ids)}

print(id2doc)

uri_to_doc_id = load_map_from_file(uri_to_doc_id_file)
doc_id_to_uri = {k: v for v, k in uri_to_doc_id.items()}

print(doc_id_to_uri)

uri_to_gold_classes = load_vectors_from_file(gold_standard,  usecols=[0,1,2,3], nullstring="-")
print(uri_to_gold_classes)

tfidf_corpus_file = laundromat_corpus + "/tfidf_corpus"
print(f"loading tf-idf corpus from {tfidf_corpus_file}")
dictionary_file = input_folder_corpus + "/dictionary"
corpus_tfidf = corpora.MmCorpus(tfidf_corpus_file)
print(f"tf-idf corpus loaded length: {len(corpus_tfidf)}")
print(f"loading dictionary from {dictionary_file}")
dictionary = corpora.Dictionary.load(dictionary_file)
id_to_dictionary_token = {v: k for k, v in dictionary.token2id.items()}
print("dictionary loaded")

doc_ids = load_list_from_file(input_folder_corpus + "/doc_ids", token_number, extractid=True)
id2doc = {k: v for v, k in enumerate(doc_ids)}

stop = get_stopwords("stopwords.txt")

hierarchy = {}
for (k, v) in load_map_from_file(hierarchy_file).items():
    hierarchy[int(k)] = [int(kd.strip()) for kd in v.split(",")]

print(hierarchy)

data = []

for root, dirs, files in os.walk("/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/output"):
    for filename in files:
        if (filename == "virtualdocument.txt.bz2"):
            key = os.path.basename(root)

            if doc_id_to_uri[key] in uri_to_gold_classes :
                txt = " ".join([str(line.decode("utf-8")).strip("\n") for line in bz2.open(os.path.join(root, filename),"r")])

                direct_klasses = uri_to_gold_classes[doc_id_to_uri[key]]
                undirect_classes = [k for k in direct_klasses]

                for klass in undirect_classes:
                    if domain_to_id[klass] in hierarchy:
                        for super_klass in hierarchy[domain_to_id[klass]]:
                            if id_to_domain[super_klass] not in undirect_classes:
                                undirect_classes.append(id_to_domain[super_klass])

                data.append([key, doc_id_to_uri[key],  undirect_classes, txt])

df = pd.DataFrame(data, columns=['Doc Key', 'Doc URI', 'Class Label', 'Text'])

df = df[['Class Label', 'Text']]

X, y = resample(df['Text'], df['Class Label'], random_state=0)

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.1)

#tfidf = TfidfTransformer()
cv = CountVectorizer(lowercase=True, stop_words=stop, tokenizer=LemmaTokenizer(), binary=True)
#pipeline = Pipeline(
#    [
#        ("vect", cv),
#        ("tfidf", tfidf),
#    ]
#)

X = cv.fit_transform(X)

multi_target = MultiOutputClassifier(RandomForestClassifier(class_weight="balanced"), n_jobs=-1)
multi_target.fit(X, y)

print(X.shape)
print(len(corpus_tfidf))
print(corpus_tfidf[0][0][0])
print(corpus_tfidf[0][0][1])
print(id_to_dictionary_token[corpus_tfidf[0][0][0]])
features = cv.get_feature_names()
print(features.index(id_to_dictionary_token[corpus_tfidf[0][0][0]]))
print(len(features))

dic_id_to_feature_id = {dictionary.token2id[feature]: feature_id for feature_id, feature in enumerate(features) if feature in dictionary.token2id}
print(dic_id_to_feature_id[corpus_tfidf[0][0][0]])

for doc in corpus_tfidf:
    d = doc_to_csr_matrix(doc, dic_id_to_feature_id, features)
    p = multi_target.predict(d)
    print(mlb.inverse_transform(p))
#d = X[0]
#print(d)
#print(multi_target.score(d))
#print(p)

