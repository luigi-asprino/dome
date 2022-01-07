import os
from sklearn.base import BaseEstimator, TransformerMixin
from utils.Utils import load_map_from_file, load_list_from_file
import bz2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
from sklearn.utils import resample
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from stop_words import get_stop_words, AVAILABLE_LANGUAGES
import logging
import pycountry
from domainannotators.WordAnnotators import RocksDBDomainDisambiguator, AggregationStrategy
from domainannotators.DocumentDomainAnnotators import SimpleDocumentAnnotator, DocumentAnnotatorAggregationStrategy
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import scipy.sparse as sp
import pandas as pd

import numpy as np
np.random.seed(0)

from nltk import word_tokenize
from gensim.models import word2vec

from sklearn.model_selection import cross_val_score



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.getLogger("gensim").setLevel(logging.ERROR)
logging.getLogger("polyglot").setLevel(logging.ERROR)
trace = logging.getLogger("trace")
trace.setLevel(logging.INFO)
timings = logging.getLogger("timings")
timings.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

import pickle

def get_stopwords(stopwords_file):

    if os.path.exists(stopwords_file + ".p"):
        return pickle.load(open(stopwords_file + ".p", "rb"))


    # Create a stopword list
    stop = set()
    with open(stopwords_file, "r") as fp:
        line = fp.readline()
        stop.add(line[:-1])
        while line:
            line = fp.readline()
            stop.add(line[:-1])

    for c in pycountry.countries:
        stop.add(c.alpha_2.lower())
        stop.add(c.alpha_3.lower())

    # # Importing stopwords for available languages https://github.com/Alir3z4/python-stop-words
    for l in AVAILABLE_LANGUAGES:
        for sw in get_stop_words(l):
            stop.add(sw)


    words_to_exclude = ["property", "label", "comment", "class", "restriction", "ontology", "nil", "individual",
                        "value", "domain", "range", "first", "rest", "resource", "datatype", "integer", "equivalent",
                        "title", "thing", "creator", "disjoint", "predicate", "dublin", "taxonomy", "axiom", "foaf",
                        "dc", "uri", "void", "dataset", "subject", "term", "agent",
                        "boolean", "xml", "httpd", "https", "sub"]

    for w in words_to_exclude:
        stop.add(w)

    logger.info(f"Number of Stopwords {len(stop)}")

    pickle.dump([w for w in stop], open(stopwords_file + ".p", "wb"))
    return stop

class W2vVectorizer(object):

    def __init__(self, w2v):
        # Takes in a dictionary of words and vectors as input
        self.w2v = w2v
        if len(w2v) == 0:
            self.dimensions = 0
        else:
            self.dimensions = len(w2v[next(iter(glove))])

    # Note: Even though it doesn't do anything, it's required that this object implement a fit method or else
    # it can't be used in a scikit-learn pipeline
    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.w2v[w] for w in words if w in self.w2v]
                   or [np.zeros(self.dimensions)], axis=0) for words in X])

class DomainTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, da, cv):
        self.da = da
        self.cv = cv

    def fit(self, X, y=None):
        self.words = self.cv.get_feature_names()
        print(f"Number of words {len(self.words)}")
        return self

    def transform(self, X, y=None):
        print("Transform called")
        print(type(X))
        print(X.shape)

        d = np.zeros((X.shape[0], len(da.id_to_domain)))
        print(f"Number of documents {X.shape[0]}")
        for row in range(0,X.shape[0]):
            #doc_words_all = {}
            #nz = X[row, :].nonzero()
            #for col in nz[1]:
            #    word = self.words[col]
            #    score = X[row,col]
            #    doc_words_all[word] = score
            #doc_words_all = {self.words[row_col[1]]: X_[row_col[0]][row_col[1]] for row_col in X_[row,:].nonzero()}

            doc_words_all = {self.words[col]: X[row, col] for col in X[row, :].nonzero()[1]}
            d[row] = da.get_domain_vector(doc_words_all)[0]

        return sp.hstack([X,d])

id_to_domain = "/Users/lgu/workspace/ekr/dome/resources/20211126_input_unified/id2domain.tsv"
input_folder_corpus = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/Corpus_lov"
token_number = 8
uri_to_doc_id_file = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/output/index.tsv"
gold_standard = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/LOV_KD_annotations.tsv"

id_to_domain = load_map_from_file(id_to_domain)
domain_to_id = {k: v for v, k in id_to_domain.items()}

print(id_to_domain)

doc_ids = load_list_from_file(input_folder_corpus + "/doc_ids", token_number, extractid=True)
id2doc = {k: v for v, k in enumerate(doc_ids)}

print(id2doc)

uri_to_doc_id = load_map_from_file(uri_to_doc_id_file)
doc_id_to_uri = {k: v for v, k in uri_to_doc_id.items()}

print(doc_id_to_uri)

uri_to_gold_class = load_map_from_file(gold_standard)
print(uri_to_gold_class)

stop = get_stopwords("stopwords.txt")

data = []

for root, dirs, files in os.walk("/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/output"):
    for filename in files:
        if (filename == "virtualdocument.txt.bz2"):
            key = os.path.basename(root)

            if doc_id_to_uri[key] in uri_to_gold_class :

                #print(f"{root}{filename}")
                #print(f"{key}")
                #print(doc_id_to_uri[key])
                #print(uri_to_gold_class[doc_id_to_uri[key]])

                txt = " ".join([str(line.decode("utf-8")).strip("\n") for line in bz2.open(os.path.join(root, filename), "r")])
                data.append([key, doc_id_to_uri[key], domain_to_id[uri_to_gold_class[doc_id_to_uri[key]]], uri_to_gold_class[doc_id_to_uri[key]] , txt])
                #print(txt)



df = pd.DataFrame(data, columns=['Doc Key', 'Doc URI', 'Class ID', 'Class Label', 'Text'])

df = df[['Class Label', 'Text']]

#print(df)

#fig = plt.figure(figsize=(8,6))
#df.groupby('Class Label').count().plot(kind='bar')
#plt.show()

lemma_to_domain_dbs=["/Users/lgu/workspace/ekr/dome/resources/20211126_input_unified/lemma_to_domain_wn","wn", "/Users/lgu/workspace/ekr/dome/resources/20211126_input_unified/lemma_to_domain_bn","bn"]

lemma_dbs = []
i = 0
while i < len(lemma_to_domain_dbs):
    print(f"Loading DB from path {lemma_to_domain_dbs[i]} {lemma_to_domain_dbs[i + 1]}")
    db = RocksDBDomainDisambiguator(lemma_to_domain_dbs[i], lemma_to_domain_dbs[i + 1], id_to_domain, hierarchy={}, strategy=AggregationStrategy.MAX)
    lemma_dbs.append(db)
    i = i + 2

da = SimpleDocumentAnnotator(None, id_to_domain, lemma_dbs, [], strategy=DocumentAnnotatorAggregationStrategy.SUM_WORD_MAX)

#tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words=stop)

#features = tfidf.fit_transform(df['Text']).toarray()
#print(features)
#print(tfidf.get_feature_names())
#labels = df['Class Label']
#print(features.shape)

#docs = []
#feature_names = tfidf.get_feature_names()
#for doc_id, doc_tf_idfs in enumerate(features):
#    doc_words_all = {feature_names[word_id]: word_score for word_id, word_score in enumerate(doc_tf_idfs) if word_score}
#    domain_vector = da.get_domain_vector(doc_words_all)[0]
#    domain_dict = {id_to_domain[domain_id]: score for domain_id, score in enumerate(domain_vector) if score}
#    docs.append({**domain_dict, **doc_words_all})
#docs = pd.DataFrame(docs)
#print(docs)

#print(da.get_domain_vector(docs[0]))
#print(sorted(enumerate(da.get_domain_vector(docs[0])[0]), key=lambda item: item[1], reverse=True))

#domains_ordered = [(domain_id, score) for domain_id, score in sorted(enumerate(da.get_domain_vector(docs[0])[0]), key=lambda item: item[1], reverse=True) if score]

#for domain_id, score in domains_ordered:
#    print(id_to_domain[domain_id] + "\t" + str(score))

N = 5
#for doc_id, class_id in sorted(labels.items()):
    #print(f"{doc_id}{class_id}")
#    features_chi2 = chi2(features, labels == class_id)
#    indices = np.argsort(features_chi2[0])
#    feature_names = np.array(tfidf.get_feature_names())[indices]
#    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
#    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
#    print("# '{}':".format(doc_id))
#    print("  . Most correlated unigrams:\n. {}".format('\n    . '.join(unigrams[-N:])))
#    print("  . Most correlated bigrams:\n. {}".format('\n    . '.join(bigrams[-N:])))


#X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Class Label'], random_state=0, test_size=0.20)

#print(f"{len(X_train)} {len(X_test)}")

#X_train, y_train = resample(X_train, y_train, random_state=0)


#fig1 = plt.figure(figsize=(8,6))
#df.groupby('Class Label').count().plot(kind='bar')
#plt.show()

X, y = resample(df['Text'], df['Class Label'], random_state=0)

#fig2 = plt.figure(figsize=(8,6))
#d = {'Text': X, 'Class Label': y}
#df2 = pd.DataFrame(data=d)
#df2.groupby('Class Label').count().plot(kind='bar')
#plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.1)

tfidf = TfidfTransformer()
cv = CountVectorizer()
pipeline = Pipeline(
    [
        ("vect", cv),
        ("tfidf", tfidf),
        ("da", DomainTransformer(da, cv))
    ]
)

X_train_tfidf = pipeline.fit_transform(X_train)
X_test_tfidf = pipeline.transform(X_test)

X = pipeline.fit_transform(X)

models = [
    RandomForestClassifier(random_state=0),
    LinearSVC(class_weight='balanced'),
    MultinomialNB(),
    LogisticRegression(solver='newton-cg', class_weight='balanced', random_state=0),
    svm.SVC(decision_function_shape='ovo', class_weight='balanced'),
    SGDClassifier(class_weight='balanced'),
    Perceptron(random_state=0),
    OneVsRestClassifier(RandomForestClassifier(random_state=0)),
    OneVsRestClassifier(LinearSVC(class_weight='balanced')),
    OneVsRestClassifier(MultinomialNB()),
    OneVsRestClassifier(LogisticRegression(solver='newton-cg', class_weight='balanced', random_state=0)),
    OneVsRestClassifier(svm.SVC(decision_function_shape='ovo', class_weight='balanced')),
    OneVsRestClassifier(SGDClassifier(class_weight='balanced')),
    OneVsRestClassifier(Perceptron(random_state=0)),
]

models2 = models.copy()

for idx, clf in enumerate(models):

    scoring = "f1_weighted"

    # 10-fold cross validation of the model
    scores = cross_val_score(clf, X, y, cv=10, scoring=scoring)

    # fit model2 to get performance on individual classes
    models2[idx].fit(X_train_tfidf, y_train)
    y_pred = models2[idx].predict(X_test_tfidf)

    # print results
    estimator = clf.__class__.__name__
    if hasattr(clf, 'estimator'):
        estimator = f"{clf.__class__.__name__} {clf.estimator.__class__.__name__}"

    print(f"{estimator} {scores.mean()} {scoring} with a standard deviation of {scores.std()}\n")
    print(classification_report(y_test, y_pred))


#print(metrics.confusion_matrix(y_test, y_pred))
#cm = metrics.confusion_matrix(y_test, y_pred,  labels=clf.classes_)
#disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
#                              display_labels=clf.classes_)
#disp.plot(xticks_rotation="vertical")
#plt.show()



