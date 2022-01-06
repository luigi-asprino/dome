import os

from scipy.sparse import csr_matrix

from utils.Utils import load_map_from_file, load_list_from_file, load_vectors_from_file
from utils.ml_utils import resample
import bz2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import multilabel_confusion_matrix
from preprocessing.Tokenizer import LemmaTokenizer
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
import pickle

# Logging configuration
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
trace = logging.getLogger("trace")
trace.setLevel(logging.INFO)
timings = logging.getLogger("timings")
timings.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def load_dataset(folder, doc_id_to_uri, uri_to_gold_classes, hierarchy=None):
    data = []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if (filename == "virtualdocument.txt.bz2"):
                key = os.path.basename(root)

                if doc_id_to_uri[key] in uri_to_gold_classes:
                    txt = " ".join(
                        [str(line.decode("utf-8")).strip("\n") for line in bz2.open(os.path.join(root, filename), "r")])

                    direct_klasses = uri_to_gold_classes[doc_id_to_uri[key]]
                    undirect_classes = [k for k in direct_klasses]

                    if hierarchy is not None:
                        for klass in undirect_classes:
                            if domain_to_id[klass] in hierarchy:
                                for super_klass in hierarchy[domain_to_id[klass]]:
                                    if id_to_domain[super_klass] not in undirect_classes:
                                        undirect_classes.append(id_to_domain[super_klass])
                    #data.append([key, doc_id_to_uri[key], undirect_classes, txt])
                    data.append([undirect_classes, txt])
    return data


def load_benchmark(virtual_documents, doc_id_to_uri, uri_to_gold_classes, headers, hierarchy):
    data = []

    for root, dirs, files in os.walk(virtual_documents):
        for filename in files:
            if (filename == "virtualdocument.txt.bz2"):
                key = os.path.basename(root)
                txt = " ".join(
                    [str(line.decode("utf-8")).strip("\n") for line in bz2.open(os.path.join(root, filename), "r")])

                if doc_id_to_uri[key] not in uri_to_gold_classes:
                    print(f"{key} {doc_id_to_uri[key]} not found in gold standard ")
                    if doc_id_to_uri[key] + '/' in uri_to_gold_classes:
                        print(f"{key} {doc_id_to_uri[key]} renamed to {key} {doc_id_to_uri[key] + '/'}")
                        old_uri = doc_id_to_uri[key]
                        doc_id_to_uri[key] = old_uri + '/'
                        uri_to_doc_id[old_uri + '/'] = key
                        uri_to_doc_id.pop(old_uri + '/', None)

                direct_klasses = [headers[id] for id, flag in enumerate(uri_to_gold_classes[doc_id_to_uri[key]]) if flag]

                #direct_klasses = uri_to_gold_classes[doc_id_to_uri[key]]
                undirect_classes = [k for k in direct_klasses]

                if hierarchy is not None:
                    for klass in undirect_classes:
                        if domain_to_id[klass] in hierarchy:
                            for super_klass in hierarchy[domain_to_id[klass]]:
                                if id_to_domain[super_klass] not in undirect_classes:
                                    undirect_classes.append(id_to_domain[super_klass])

                #data.append([doc_id_to_uri[key], klasses, txt])
                data.append([undirect_classes, txt])
    return data

# Arguments
id_to_domain = "/Users/lgu/workspace/ekr/dome/resources/20211126_input_unified/id2domain.tsv"
input_folder_corpus = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/Corpus_lov"
token_number = 8
uri_to_doc_id_file = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/output/index.tsv"
gold_standard = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/LOV_KD_annotations.tsv"
gold_standard_benchmark = "/Users/lgu/Desktop/NOTime/EKR/Benchmark/GoldStandart-MultiTopic.tsv"
hierarchy_file = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/KD_hierarchy.tsv"
resampling_strategy = "mlsmote_iterative"
virtual_documents_lov = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/output"
virtual_documents_benchmark = "/Users/lgu/Desktop/NOTime/EKR/Benchmark/virtual_documents"
uri_to_doc_id_file_benchmark = "/Users/lgu/Desktop/NOTime/EKR/Benchmark/virtual_documents/index.tsv"
folder = "/Users/lgu/Desktop/NOTime/EKR/partial/"
data_file = folder + "data_file.p"

if not os.path.exists(folder):
    os.mkdir(folder)

if not os.path.exists(data_file):
    # Load resources
    id_to_domain = load_map_from_file(id_to_domain)
    domain_to_id = {k: v for v, k in id_to_domain.items()}

    doc_ids = load_list_from_file(input_folder_corpus + "/doc_ids", token_number, extractid=True)
    id2doc = {k: v for v, k in enumerate(doc_ids)}

    uri_to_doc_id = load_map_from_file(uri_to_doc_id_file)
    doc_id_to_uri = {k: v for v, k in uri_to_doc_id.items()}

    uri_to_doc_id_benchmark = load_map_from_file(uri_to_doc_id_file_benchmark)
    doc_id_to_uri_benchmark = {k: v for v, k in uri_to_doc_id_benchmark.items()}

    uri_to_gold_classes = load_vectors_from_file(gold_standard,  usecols=[0,1,2,3], nullstring="-")



    hierarchy = {}
    for (k, v) in load_map_from_file(hierarchy_file).items():
        hierarchy[int(k)] = [int(kd.strip()) for kd in v.split(",")]

    data_lov = load_dataset(virtual_documents_lov, doc_id_to_uri, uri_to_gold_classes, hierarchy)
    uri_to_gold_classes_benchmark, headers_benchmark = load_vectors_from_file(gold_standard_benchmark, header=0, usecols=[0,1,2,3,4,5,6])
    data_benchmark = load_benchmark(virtual_documents_benchmark, doc_id_to_uri_benchmark, uri_to_gold_classes_benchmark, headers_benchmark, hierarchy)
    data = data_lov + data_benchmark
    df = pd.DataFrame(data, columns=['Class Label', 'Text'])
    pickle.dump(df, open(data_file, "wb"))
else:
    print("Loading data file")
    df = pickle.load(open(data_file, "rb"))

if not os.path.exists(folder+resampling_strategy):
    os.mkdir(folder+resampling_strategy)
    X = df['Text']
    y = df['Class Label']

    stop = get_stopwords("stopwords.txt")

    cv = CountVectorizer(lowercase=True, stop_words=stop, tokenizer=LemmaTokenizer(), binary=True)
    pipeline = Pipeline(
        [
            ("vect", cv),
            # ("tfidf", TfidfTransformer()),
        ]
    )
    # Preprocessing
    X = pd.DataFrame(pipeline.fit_transform(X).todense())
    mlb = MultiLabelBinarizer()
    y = pd.DataFrame(mlb.fit_transform(y))

    # Resampling
    X, y = resample(X, y, stategy=resampling_strategy)

    y = y.values
    X = csr_matrix(X.values)

    print("Dumping X and y")
    pickle.dump(X, open(folder+resampling_strategy+"/X.p", "wb"))
    pickle.dump(y, open(folder + resampling_strategy + "/y.p", "wb"))
else:
    print("Loading X and y")
    X = pickle.load(open(folder+resampling_strategy+"/X.p", "rb"))
    y = pickle.load(open(folder+resampling_strategy+"/y.p", "rb"))

# Test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.1)

classifiers = [
    MLPClassifier(solver='lbfgs'),
    #RandomForestClassifier(class_weight="balanced"),
    #KNeighborsClassifier(),
    #RadiusNeighborsClassifier()
]

for clf in classifiers:

    estimator = clf.__class__.__name__
    if hasattr(clf, 'estimator'):
        estimator = f"{clf.__class__.__name__}-{clf.estimator.__class__.__name__}"

    estimator_folder = folder+resampling_strategy+"/"+estimator

    if not os.path.exists(estimator_folder):
        os.mkdir(estimator_folder)

    scoring = "f1_weighted"
    scores = cross_val_score(clf, X, y, cv=10, scoring=scoring)

    clf.fit(X_train, y_train)

    # Get predictions for test data
    y_test_pred = clf.predict(X_test)

    # Generate multiclass confusion matrices
    matrices = multilabel_confusion_matrix(y_test, y_test_pred)

    f = open(estimator_folder + '/results.txt', 'w')

    to_print = classification_report(y_test, y_test_pred)
    f.write(to_print)
    print(to_print)

    to_print = f"{estimator} {scores.mean()} {scoring} with a standard deviation of {scores.std()}\n"
    f.write(to_print)
    print(to_print)

    clf.fit(X, y)
    pickle.dump(clf, open(estimator_folder + resampling_strategy + "/clf.p", "wb"))






