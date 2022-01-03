import os
from utils.Utils import load_map_from_file, load_list_from_file,load_vectors_from_file
import bz2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils import resample
from sklearn.metrics import classification_report
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
from sklearn.model_selection import cross_val_score
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.getLogger("gensim").setLevel(logging.ERROR)
logging.getLogger("polyglot").setLevel(logging.ERROR)
trace = logging.getLogger("trace")
trace.setLevel(logging.INFO)
timings = logging.getLogger("timings")
timings.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import multilabel_confusion_matrix
from preprocessing.Tokenizer import LemmaTokenizer

virtual_documents = "/Users/lgu/Desktop/NOTime/EKR/Benchmark/virtual_documents"
uri_to_doc_id_file = "/Users/lgu/Desktop/NOTime/EKR/Benchmark/virtual_documents/index.tsv"
gold_standard = "/Users/lgu/Desktop/NOTime/EKR/Benchmark/GoldStandart-MultiTopic.tsv"
doc_ids_file = "/Users/lgu/Desktop/NOTime/EKR/Benchmark/dataset_ids"

doc_ids = load_list_from_file(doc_ids_file)
id2doc = {k: v for v, k in enumerate(doc_ids)}
print(id2doc)

uri_to_doc_id = load_map_from_file(uri_to_doc_id_file)
doc_id_to_uri = {k: v for v, k in uri_to_doc_id.items()}
print(doc_id_to_uri)

uri_to_gold_classes, headers = load_vectors_from_file(gold_standard, header=0, usecols=[0,1,2,3,4,5,6])
print(uri_to_gold_classes)
print(headers)

stop = get_stopwords("stopwords.txt")

data = []

for root, dirs, files in os.walk(virtual_documents):
    for filename in files:
        if (filename == "virtualdocument.txt.bz2"):
            key = os.path.basename(root)
            txt = " ".join(
                [str(line.decode("utf-8")).strip("\n") for line in bz2.open(os.path.join(root, filename), "r")])



            if doc_id_to_uri[key] not in uri_to_gold_classes:
                print(f"{key} {doc_id_to_uri[key]} not found in gold standard ")
                if doc_id_to_uri[key] + '/'  in uri_to_gold_classes:
                    print(f"{key} {doc_id_to_uri[key]} renamed to {key} {doc_id_to_uri[key] + '/'}")
                    old_uri = doc_id_to_uri[key]
                    doc_id_to_uri[key] = old_uri + '/'
                    uri_to_doc_id[old_uri + '/'] = key
                    uri_to_doc_id.pop(old_uri + '/', None)

            klasses = [headers[id] for id, flag in enumerate(uri_to_gold_classes[doc_id_to_uri[key]]) if flag]

            data.append([doc_id_to_uri[key], klasses, txt])

df = pd.DataFrame(data, columns=['URI', 'Class Labels', 'Text'])
X = df['Text']
y = df['Class Labels']

#X, y = resample(df['Text'], df['Class Labels'], random_state=0)


# Resampling
X, y = resample(X, y)

# Preprocessing
cv = CountVectorizer(lowercase=True, stop_words=stop, tokenizer=LemmaTokenizer(), binary=True)
pipeline = Pipeline(
    [
        ("vect", cv),
        # ("tfidf", TfidfTransformer()),
    ]
)
X = pipeline.fit_transform(X)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)

# Test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.1)

classifiers = [
    #MLPClassifier(solver='lbfgs'),
    RandomForestClassifier(class_weight="balanced"),
    #KNeighborsClassifier(),
    #RadiusNeighborsClassifier()
]

for clf in classifiers:

    estimator = clf.__class__.__name__
    if hasattr(clf, 'estimator'):
        estimator = f"{clf.__class__.__name__} {clf.estimator.__class__.__name__}"

    scoring = "f1_weighted"
    scores = cross_val_score(clf, X, y, cv=10, scoring=scoring)

    clf.fit(X_train, y_train)

    # Get predictions for test data
    y_test_pred = clf.predict(X_test)

    # Generate multiclass confusion matrices
    matrices = multilabel_confusion_matrix(y_test, y_test_pred)

    print(classification_report(y_test, y_test_pred))
    print(f"{estimator} {scores.mean()} {scoring} with a standard deviation of {scores.std()}\n")




#y = MultiLabelBinarizer().fit_transform(y)
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
#tfidf = TfidfTransformer()
#cv = CountVectorizer(lowercase=True, stop_words=stop, tokenizer=LemmaTokenizer())
#pipeline = Pipeline(
#    [
#        ("vect", cv),
#        ("tfidf", tfidf),
#    ]
#)
#
# X_train_tfidf = pipeline.fit_transform(X_train)
# X_test_tfidf = pipeline.transform(X_test)
# X = pipeline.fit_transform(X)
#
# multi_target = MultiOutputClassifier(RandomForestClassifier(random_state=0), n_jobs=-1)
# multi_target.fit(X_train_tfidf, y_train)
#
#
# # Get predictions for test data
# y_test_pred = multi_target.predict(X_test_tfidf)
#
# # Generate multiclass confusion matrices
# matrices = multilabel_confusion_matrix(y_test, y_test_pred)
#
# print(classification_report(y_test, y_test_pred))
#
# scoring = "f1_weighted"
# scores = cross_val_score(MultiOutputClassifier(RandomForestClassifier(random_state=0), n_jobs=-1), X, y, cv=10, scoring=scoring)
#
# print(f"{scores.mean()} {scoring} with a standard deviation of {scores.std()}\n")
