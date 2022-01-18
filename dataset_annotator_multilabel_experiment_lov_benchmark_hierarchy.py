import os

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.multiclass import OneVsRestClassifier
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier

from domainannotators.DocumentDomainAnnotators import SimpleDocumentAnnotator, DocumentAnnotatorAggregationStrategy
from domainannotators.WordAnnotators import RocksDBDomainDisambiguator, AggregationStrategy
from utils.Utils import load_map_from_file, load_list_from_file, load_vectors_from_file
from utils.ml_utils import resample, DomainTransformer
import bz2
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
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
from sklearn.neural_network import MLPClassifier
from preprocessing.Tokenizer import LemmaTokenizer
from sklearn.neighbors import KNeighborsClassifier
import pickle
from utils.ml_utils import get_irlb
from sklearn.metrics import hamming_loss, accuracy_score, f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn_hierarchical_classification.constants import ROOT
import shutil
from hiclass import LocalClassifierPerNode



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
virtual_documents_lov = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/output"
virtual_documents_benchmark = "/Users/lgu/Desktop/NOTime/EKR/Benchmark/virtual_documents"
uri_to_doc_id_file_benchmark = "/Users/lgu/Desktop/NOTime/EKR/Benchmark/virtual_documents/index.tsv"

class_hierarchy = {
        ROOT: ["Sport, Recreation", "Social Science", "History", "Linguistics", "Pure Science", "Life Science",
               "Astrology", "Applied Science", "Literature", "Engineering, Technology", "Time",
               "Art, architecture, and archaeology", "Religion", "Phylosophy, Psychology", "Bibliography",
               "Scientific Research & Academy"],
        "Sport, Recreation": ["Sport", "Recreation"],
        "Social Science": ["Political Science", "Sexuality", "Sociology", "Military", "Pedagogy", "Transport, Travel",
                           "Telecommunication(telegraphy, telephony, networking)", "Law", "Industry", "Fashion",
                           "Mediology(tv, radio, newspapers)", "Economy", "Anthropology"],
        "Pure Science": ["Physics, Astronomy", "Mathematics", "Chemistry", "Earth", "Biology"],
        "Life Science": ["Biology", "Medicine"],
        "Applied Science": ["Medicine", "Alimentation", "Computer Science", "Agriculture", "Architecture",
                            "Engineering"],
        "Engineering, Technology": ["Engineering", "Technology"],
        "Art, architecture, and archaeology": ["Architecture", "Archaeology", "Art & Culture"],
        "Phylosophy, Psychology": ["Psychology", "Philosophy"],
        "Sociology" : ["Social Networking"],
        "Military": ["Naval Science"],
        "Transport, Travel": ["Transport", "Tourism"],
        "Anthropology": ["Royalty and nobility"],
        "Physics, Astronomy": ["Physics", "Astronomy"],
        "Earth": ["Geography", "Geology", "Meteorology", "Oceanography", "Paleontology"],
        "Art & Culture": ["Drawing", "Music", "Plastic Arts", "Photography", "Theatre", "Dance"]
    }
all_classes = set([v for k, vs in class_hierarchy.items() for v in vs])
all_classes.update([k for k, v in class_hierarchy.items()])

resampling_strategy = "mlsmote_iterative"
use_hierarchy = False
use_tfidf = False
use_domain_annotator = False
on_LOV = True
on_Bench = True
flash = False
only_top_level = True

folder = "/Users/lgu/Desktop/NOTime/EKR/experiments/"

if on_LOV:
    folder += "lov_"
if on_Bench:
    folder += "bench_"
if use_hierarchy:
    folder += "hp_"
if use_tfidf:
    folder += "tf_"
else:
    folder += "bin_"
if use_domain_annotator:
    folder += "da_"
if only_top_level:
    folder += "top_"
folder = folder[:-1] + "/"

data_file = folder + "data_file.p"

if not os.path.exists(folder):
    os.mkdir(folder)
elif flash:
    shutil.rmtree(folder)
    os.mkdir(folder)

fc = open(folder + '/configuration.txt', 'w')
fc.write(f"Resampling strategy {resampling_strategy}\n")
fc.write(f"Use hierarchy in training/testing phase {use_hierarchy}\n")
fc.write(f"Use hierarchy TF-IDF {use_tfidf}\n")
fc.write(f"Use domain annotator {use_domain_annotator}\n")
fc.write(f"On LOV {on_LOV}\n")
fc.write(f"On Benchmark {on_Bench}\n")
fc.write(f"Only Top-Level {only_top_level}\n")

# Load resources
id_to_domain = load_map_from_file(id_to_domain)
domain_to_id = {k: v for v, k in id_to_domain.items()}

## Check classes
for domain in all_classes:
    if domain not in domain_to_id:
        print(f"Wrong domain {domain}")

tod_domains_ids = [domain_to_id[d] for d in class_hierarchy[ROOT]]

if not os.path.exists(data_file):

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

    if use_hierarchy:
        data_lov = load_dataset(virtual_documents_lov, doc_id_to_uri, uri_to_gold_classes, hierarchy)
    else:
        data_lov = load_dataset(virtual_documents_lov, doc_id_to_uri, uri_to_gold_classes, None)

    uri_to_gold_classes_benchmark, headers_benchmark = load_vectors_from_file(gold_standard_benchmark, header=0, usecols=[0,1,2,3,4,5,6])
    data_benchmark = load_benchmark(virtual_documents_benchmark, doc_id_to_uri_benchmark, uri_to_gold_classes_benchmark, headers_benchmark, hierarchy)
    
    data = []
    if on_LOV:
        data = data + data_lov
    if on_Bench:
        data = data + data_benchmark

    if only_top_level:
        data_new = []
        for klasses, txt in data:
            klasses_new = []
            for klass_label in klasses:
                #top_classes = [id_to_domain[klass_id] for klass_id in domain_to_id[klass_label] for super_ if klass_id in tod_domains_ids]
                top_classes = []
                if domain_to_id[klass_label] in tod_domains_ids:
                    if klass_label not in klasses_new:
                        top_classes.append(klass_label)
                elif domain_to_id[klass_label] in hierarchy:
                    for super_klass_id in hierarchy[domain_to_id[klass_label]]:
                        if super_klass_id in tod_domains_ids:
                            if id_to_domain[super_klass_id] not in klasses_new:
                                top_classes.append(id_to_domain[super_klass_id])
                klasses_new.extend(top_classes)
            data_new.append([klasses_new, txt])
            #print(f"{klasses} -> {klasses_new}")

        data = data_new

    df = pd.DataFrame(data, columns=['Class Label', 'Text'])
    pickle.dump(df, open(data_file, "wb"))
else:
    print("Loading data file")
    df = pickle.load(open(data_file, "rb"))

print(df)

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

if not os.path.exists(folder+resampling_strategy):
    os.mkdir(folder+resampling_strategy)
    X = df['Text']
    y = df['Class Label']

    stop = get_stopwords("stopwords.txt")

    cv = CountVectorizer(lowercase=True, stop_words=stop, tokenizer=LemmaTokenizer(), binary=not use_tfidf)

    transformers_pipeline = [("vect", cv)]

    if use_tfidf:
        transformers_pipeline.append(("tfidf", TfidfTransformer()))

    if use_domain_annotator:
        transformers_pipeline.append(("da", DomainTransformer(da, cv)))

    pipeline = Pipeline(transformers_pipeline)

    # Preprocessing
    X = pd.DataFrame(pipeline.fit_transform(X).todense())
    mlb = MultiLabelBinarizer()
    y = pd.DataFrame(mlb.fit_transform(y))

    # Resampling
    X, y = resample(X, y, stategy=resampling_strategy)

    irlb, irlb_mean_last = get_irlb(y)
    mess = f"IRLB mean {irlb_mean_last}"
    f = open(folder+resampling_strategy+"/irlb_mean.txt", 'w')
    f.write(mess)

    y = y.values
    X = csr_matrix(X.values)

    print("Dumping X and y")
    pickle.dump(X, open(folder+resampling_strategy+"/X.p", "wb"))
    pickle.dump(y, open(folder + resampling_strategy + "/y.p", "wb"))
    pickle.dump(mlb, open(folder + resampling_strategy + "/mlb.p", "wb"))
    if not use_domain_annotator:
        pickle.dump(pipeline, open(folder + resampling_strategy + "/pipeline.p", "wb"))
    pickle.dump(cv, open(folder + resampling_strategy + "/cv.p", "wb"))
else:
    print("Loading X and y")
    X = pickle.load(open(folder+resampling_strategy+"/X.p", "rb"))
    y = pickle.load(open(folder+resampling_strategy+"/y.p", "rb"))
    mlb = pickle.load(open(folder + resampling_strategy + "/mlb.p", "rb"))

# Test train split
classifiers = [
    RandomForestClassifier(class_weight="balanced"),
    KNeighborsClassifier(),
    MLPClassifier(solver='lbfgs'),
    # HierarchicalClassifier(
    #     base_estimator=MLPClassifier(solver='lbfgs'),
    #     class_hierarchy=class_hierarchy,
    #     mlb=mlb,
    #     use_decision_function=True
    # )
]

for clf in classifiers:

    estimator = clf.__class__.__name__
    if hasattr(clf, 'estimator'):
        estimator = f"{clf.__class__.__name__}-{clf.estimator.__class__.__name__}"

    estimator_folder = folder+resampling_strategy+"/"+estimator

    if not os.path.exists(estimator_folder):
        os.mkdir(estimator_folder)
    else:
        print(f"Skipping {estimator}")
        continue

    #scoring = "f1_weighted"
    mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    f = open(estimator_folder + '/results.txt', 'w')
    computed_metrics = {"hls": [], "accs": [],  "f1_micro": [], "f1_macro": [], "f1_weight": []}
    for train_index, test_index in mskf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)

        to_print = classification_report(y_test, y_test_pred)
        f.write(to_print)
        print(estimator)
        print(to_print)

        hl = hamming_loss(y_test, y_test_pred)
        computed_metrics["hls"].append(hl)

        acc = accuracy_score(y_test, y_test_pred)
        computed_metrics["accs"].append(acc)

        f1_micro = f1_score(y_test, y_test_pred, average="micro")
        computed_metrics["f1_micro"].append(f1_micro)

        f1_macro = f1_score(y_test, y_test_pred, average="macro")
        computed_metrics["f1_macro"].append(f1_macro)

        f1_weight = f1_score(y_test, y_test_pred, average="weighted")
        computed_metrics["f1_weight"].append(f1_weight)

        to_print = f"Hamming loss {hl} Accuracy {acc} F1: micro {f1_micro} macro {f1_macro} weight {f1_weight}\n"

        f.write(to_print)
        print(to_print)

    ## Print mean and std
    for metric_name, observations in computed_metrics.items():
        mean = np.array(observations).mean()
        std = np.array(observations).std()
        to_print = f"{metric_name} mean {mean} std {std}\n"
        f.write(to_print)
        print(to_print)


    #scores = cross_val_score(clf, X, y, cv=10, scoring=scoring)

    #clf.fit(X_train, y_train)

    # Get predictions for test data
    #y_test_pred = clf.predict(X_test)

    # Generate multilabel confusion matrices
    #matrices = multilabel_confusion_matrix(y_test, y_test_pred)

    #f = open(estimator_folder + '/results.txt', 'w')

    #to_print = classification_report(y_test, y_test_pred)
    #f.write(to_print)
    #print(to_print)

    #hl = hamming_loss(y_test, y_test_pred)
    #acc = accuracy_score(y_test, y_test_pred)
    #to_print = f"Hamming loss {hl} Accuracy {acc}"
    #f.write(to_print)
    #print(to_print)

    #to_print = f"{estimator} {scores.mean()} {scoring} with a standard deviation of {scores.std()}\n"
    #f.write(to_print)
    #print(to_print)

    clf.fit(X, y)
    pickle.dump(clf, open(estimator_folder + "/clf.p", "wb"))