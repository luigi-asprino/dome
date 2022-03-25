from domainannotators.DocumentDomainAnnotators import SimpleDocumentAnnotator, DocumentAnnotatorAggregationStrategy
from domainannotators.WordAnnotators import RocksDBDomainDisambiguator, AggregationStrategy
from gensim import corpora
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from loaders.load_utils import load_dataset, load_benchmark
import logging
import numpy as np
import os
import pandas as pd
import pickle
from preprocessing.Tokenizer import LemmaTokenizer
import random
from scipy.sparse import csr_matrix
import shutil
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.feature_selection import chi2, SelectKBest, SelectPercentile
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import hamming_loss, accuracy_score, f1_score
from sklearn_hierarchical_classification.constants import ROOT

from utils import ml_utils
from utils.Utils import load_map_from_file, load_list_from_file, load_vectors_from_file, get_stopwords
from utils.ml_utils import get_irlb, write_class_distribution_on_file, specialize_annotations, \
    get_indexes_of_items_with_labels, oversampling, \
    DomainTransformer, Strategies
from utils.corpora import get_doc
import warnings




warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
np.random.seed(0)

# Logging configuration
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
trace = logging.getLogger("trace")
trace.setLevel(logging.INFO)
timings = logging.getLogger("timings")
timings.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Arguments
id_to_domain = "/Users/lgu/workspace/ekr/dome/resources/20211126_input_unified/id2domain.tsv"
input_folder_corpus = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/Corpus_lov"
token_number = 8
n_sample = 100
uri_to_doc_id_file = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/output/index.tsv"
gold_standard = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/LOV_KD_annotations.tsv"
gold_standard_LOV_max = "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/LOV/TSVs/maximal_annotation_set.p"
gold_standard_LOV_int = "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/LOV/TSVs/intersection_annotations.p"
gold_standard_Laun_max = "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/Laundromat/maximal_annotation_set.p"
gold_standard_Laun_int = "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/Laundromat/intersection_annotations.p"

gold_standard_benchmark = "/Users/lgu/Desktop/NOTime/EKR/Benchmark/GoldStandart-MultiTopic.tsv"
hierarchy_file = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/KD_hierarchy.tsv"
virtual_documents_lov = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/output"
virtual_documents_benchmark = "/Users/lgu/Desktop/NOTime/EKR/Benchmark/virtual_documents"
uri_to_doc_id_file_benchmark = "/Users/lgu/Desktop/NOTime/EKR/Benchmark/virtual_documents/index.tsv"
laundromat_corpus = "/Users/lgu/Dropbox/Backups/Corpus_lod"
tfidf_corpus_file = laundromat_corpus + "/tfidf_corpus"
dictionary_file = laundromat_corpus + "/dictionary"

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
    "Sociology": ["Social Networking"],
    "Military": ["Naval Science"],
    "Transport, Travel": ["Transport", "Tourism"],
    "Anthropology": ["Royalty and nobility"],
    "Physics, Astronomy": ["Physics", "Astronomy"],
    "Earth": ["Geography", "Geology", "Meteorology", "Oceanography", "Paleontology"],
    "Art & Culture": ["Drawing", "Music", "Plastic Arts", "Photography", "Theatre", "Dance"]
}
all_classes = set([v for k, vs in class_hierarchy.items() for v in vs])
all_classes.update([k for k, v in class_hierarchy.items()])

resampling_strategy = Strategies.OVERSAMPLING
use_hierarchy = False
use_tfidf = True
use_domain_annotator = False
on_LOV = True
on_Bench = True
on_LOD_Laundromat = True
flash = False
only_top_level = False
use_feature_selection = False
MAXIMAL = "maximal"
INTERSECTION = "intersection"
intersection_or_maximal = MAXIMAL

folder = "/Users/lgu/Desktop/NOTime/EKR/experiments/"

if on_LOV:
    if intersection_or_maximal == INTERSECTION:
        folder += "lov_int_"
    elif intersection_or_maximal == MAXIMAL:
        folder += "lov_max_"
    else:
        folder += "lov_"
if on_Bench:
    folder += "bench_"
if on_LOD_Laundromat:
    folder += "laund_"
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
if use_feature_selection:
    folder += "fs_"
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
fc.write(f"On LOD Laundromat {on_Bench}\n")
fc.write(f"Only Top-Level {only_top_level}\n")
fc.write(f"Use feature selection {use_feature_selection}\n")
fc.write(f"Intersection/Maximal {intersection_or_maximal}\n")

# Load resources
id_to_domain = load_map_from_file(id_to_domain)
domain_to_id = {k: v for v, k in id_to_domain.items()}

# Check classes
for domain in all_classes:
    if domain not in domain_to_id:
        print(f"Wrong domain {domain}")

# Compute Top level domains
tod_domains_ids = [domain_to_id[d] for d in class_hierarchy[ROOT]]

# Load ids corpus
doc_ids_laun = load_list_from_file(laundromat_corpus + "/doc_ids", token_number, extractid=True)
id2doc_laun = {k: v for v, k in enumerate(doc_ids_laun)}


# Initialize rule-based domain annotators
lemma_to_domain_dbs = [
    "/Users/lgu/workspace/ekr/dome/resources/20211126_input_unified/lemma_to_domain_wn",
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

# loading hierarchy
hierarchy = {}
for (k, v) in load_map_from_file(hierarchy_file).items():
    hierarchy[int(k)] = [int(kd.strip()) for kd in v.split(",")]
print(hierarchy)

if not os.path.exists(folder + "X_pre.p"):

    # loading ids LOV corpus
    doc_ids = load_list_from_file(input_folder_corpus + "/doc_ids", token_number, extractid=True)
    id2doc = {k: v for v, k in enumerate(doc_ids)}

    # loading URI LOV corpus
    uri_to_doc_id = load_map_from_file(uri_to_doc_id_file)
    doc_id_to_uri = {k: v for v, k in uri_to_doc_id.items()}

    # loading annotations Benchmark corpus
    uri_to_doc_id_benchmark = load_map_from_file(uri_to_doc_id_file_benchmark)
    doc_id_to_uri_benchmark = {k: v for v, k in uri_to_doc_id_benchmark.items()}

    # loading annotations LOV annotations
    if intersection_or_maximal == INTERSECTION:
        uri_to_gold_classes = pickle.load(open(gold_standard_LOV_int, "rb"))
    elif intersection_or_maximal == MAXIMAL:
        uri_to_gold_classes = pickle.load(open(gold_standard_LOV_max, "rb"))
    else:
        uri_to_gold_classes = load_vectors_from_file(gold_standard, usecols=[0, 1, 2, 3], nullstring="-")

    # loading annotations Laundromat annotations
    if intersection_or_maximal == INTERSECTION:
        uri_to_gold_classes_laun = pickle.load(open(gold_standard_Laun_int, "rb"))
    else:
        uri_to_gold_classes_laun = pickle.load(open(gold_standard_Laun_max, "rb"))

    # loading data Benchmark annotations
    uri_to_gold_classes_benchmark, headers_benchmark = \
        load_vectors_from_file(gold_standard_benchmark, header=0, usecols=[0, 1, 2, 3, 4, 5, 6])

    # loading data LOV
    if use_hierarchy:
        data_lov = load_dataset(virtual_documents_lov, doc_id_to_uri, uri_to_gold_classes,
                                domain_to_id, id_to_domain, hierarchy)
        data_benchmark = load_benchmark(virtual_documents_benchmark, doc_id_to_uri_benchmark,
                                        uri_to_gold_classes_benchmark,
                                        headers_benchmark, hierarchy, uri_to_doc_id_benchmark,
                                        domain_to_id, id_to_domain)
    else:
        data_lov = load_dataset(virtual_documents_lov, doc_id_to_uri, uri_to_gold_classes,
                                domain_to_id, id_to_domain, None)
        data_benchmark = load_benchmark(virtual_documents_benchmark, doc_id_to_uri_benchmark,
                                        uri_to_gold_classes_benchmark,
                                        headers_benchmark, None, uri_to_doc_id_benchmark, domain_to_id, id_to_domain)

    # combine data
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
                # top_classes = [id_to_domain[klass_id] for klass_id
                # in domain_to_id[klass_label] for super_ if klass_id in tod_domains_ids]
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
            # print(f"{klasses} -> {klasses_new}")

        data = data_new

    print(f"Number of examples {len(data)}")

    df = pd.DataFrame(data, columns=['Class Label', 'Text'])
    pickle.dump(df, open(data_file, "wb"))

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

    if on_LOD_Laundromat:

        laun_annotations = [{0: annotations} for doc_id, annotations in uri_to_gold_classes_laun.items()]
        y = pd.concat([y, pd.DataFrame(laun_annotations)], ignore_index=True)

        dictionary = corpora.Dictionary.load(dictionary_file)
        corpus_tfidf = corpora.MmCorpus(tfidf_corpus_file)

        features = cv.get_feature_names()
        feature_to_id = {feature: feature_id for feature_id, feature in enumerate(features)}

        docs = [get_doc(corpus_tfidf[id2doc_laun[doc_id]], dictionary, use_word=False, as_dictionary=True,
                        binary_score=not use_tfidf)
                for doc_id, annotations in uri_to_gold_classes_laun.items()]

        words_laun_unique = set([dictionary[w] for d in docs for w, s in d.items() if w >= 0])

        words_laun_to_add = [w for w in words_laun_unique if w not in feature_to_id]

        for w in words_laun_to_add:
            features.append(w)

        dic_id_to_features_id = {dictionary.token2id[feature]: feature_id for feature_id, feature in enumerate(features)
                                 if feature in dictionary.token2id}

        pickle.dump(dic_id_to_features_id, open(folder + "/dic_id_to_features_id.p", "wb"))
        pickle.dump(features, open(folder + "/features.p", "wb"))

        docs = [{dic_id_to_features_id[w]: s for w, s in d.items()} for d in docs]

        X = pd.concat([X, pd.DataFrame(docs)], ignore_index=True)
        X = X.fillna(0)

    if not use_domain_annotator:
        pickle.dump(pipeline, open(folder + "/pipeline.p", "wb"))

    pickle.dump(X, open(folder + "X_pre.p", "wb"))
    pickle.dump(y, open(folder + "y_pre.p", "wb"))
    pickle.dump(cv, open(folder + "/cv.p", "wb"))

else:
    print("Loading X and y..")
    X = pickle.load(open(folder + "X_pre.p", "rb"))
    y = pickle.load(open(folder + "y_pre.p", "rb"))
    print("Loaded..")

specialize_annotations(y[0], hierarchy, domain_to_id)

counter, counter_ordered, label_to_annotation_set = \
        write_class_distribution_on_file(y, folder +"class_distribution.tsv")

counter, counter_ordered, label_to_annotation_set = \
        write_class_distribution_on_file(y, folder + "class_distribution_after_downsampling.tsv")

X, y = ml_utils.undersampling(X, y)

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y[0])
X = csr_matrix(X.values)

# if not os.path.exists(folder + resampling_strategy):
#
#     specialize_annotations(y[0], hierarchy, domain_to_id)
#
#     counter, counter_ordered, label_to_annotation_set = \
#         write_class_distribution_on_file(y, folder +"class_distribution.tsv")
#
#     X, y = ml_utils.undersampling(X, y)
#
#     counter, counter_ordered, label_to_annotation_set = \
#         write_class_distribution_on_file(y, folder + "class_distribution_after_downsampling.tsv")
#
#     print(f"len(X): {len(X)} len(y): {len(y)}")
#
#     os.mkdir(folder+resampling_strategy)
#     mlb = MultiLabelBinarizer()
#     y = pd.DataFrame(mlb.fit_transform(y[0]))
#     # irlb, irlb_mean = get_irlb(y)
#     # print(irlb_mean)
#     #
#     # if use_feature_selection:
#     #     X = csr_matrix(X.values)
#     #     X = SelectPercentile(chi2).fit_transform(X, y)
#     #     X = pd.DataFrame(X.toarray())
#     #
#     # irlb, irlb_mean_last = get_irlb(y)
#     # mess = f"IRLB mean {irlb_mean_last} number of samples {n_sample}"
#     # f = open(folder + resampling_strategy + "/irlb_mean.txt", 'w')
#     # f.write(mess)
#
#     y = y.values
#     X = csr_matrix(X.values)
#
#     print("Dumping X and y")
#     pickle.dump(X, open(folder + resampling_strategy + "/X.p", "wb"))
#     pickle.dump(y, open(folder + resampling_strategy + "/y.p", "wb"))
#     pickle.dump(mlb, open(folder + resampling_strategy + "/mlb.p", "wb"))
# else:
#     print("Loading X and y")
#     X = pickle.load(open(folder + resampling_strategy + "/X.p", "rb"))
#     y = pickle.load(open(folder + resampling_strategy + "/y.p", "rb"))
#     mlb = pickle.load(open(folder + resampling_strategy + "/mlb.p", "rb"))

# Test train split
classifiers = [
    RandomForestClassifier(class_weight="balanced"),
    KNeighborsClassifier(),
    MLPClassifier(solver='lbfgs'),
]

for clf in classifiers:

    estimator = clf.__class__.__name__
    if hasattr(clf, 'estimator'):
        estimator = f"{clf.__class__.__name__}-{clf.estimator.__class__.__name__}"

    estimator_folder = folder + "/" + estimator

    if not os.path.exists(estimator_folder):
        os.mkdir(estimator_folder)
    else:
        print(f"Skipping {estimator}")
        continue

    mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    f = open(estimator_folder + '/results.txt', 'w')
    computed_metrics = {"hls": [], "accs": [], "f1_micro": [], "f1_macro": [], "f1_weight": []}
    n_fold = 0
    for train_index, test_index in mskf.split(X, y):
        print(f"Training {estimator} FOLD {n_fold}")
        n_fold += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = pd.DataFrame(X_train.toarray()).reset_index(drop=True)
        y_train = pd.DataFrame(y_train).reset_index(drop=True)

        X_train, y_train = oversampling(X_train, y_train)
        X_train = csr_matrix(X_train.values)
        y_train = y_train.values

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

        to_print = f"Hamming loss {hl} Accuracy {acc} F1: micro {f1_micro} " \
                   f"macro {f1_macro} weight {f1_weight}\n"

        f.write(to_print)
        print(to_print)

    # Print mean and std
    for metric_name, observations in computed_metrics.items():
        mean = np.array(observations).mean()
        std = np.array(observations).std()
        to_print = f"{metric_name} mean {mean} std {std}\n"
        f.write(to_print)
        print(to_print)

    to_print = f"{folder.split('/')[-2]}\t" \
               f"{np.array(computed_metrics['f1_micro']).mean():.3f}\t" \
               f"{np.array(computed_metrics['f1_micro']).std():.3f}\t" \
               f"{np.array(computed_metrics['accs']).mean():.3f}\t{np.array(computed_metrics['accs']).std():.3f}\t" \
               f"{np.array(computed_metrics['hls']).mean():.3f}\t{np.array(computed_metrics['hls']).std():.3f}\t" \
               f"\n".replace(".", ",")

    f.write(to_print)
    print(to_print)

    # scores = cross_val_score(clf, X, y, cv=10, scoring=scoring)

    # clf.fit(X_train, y_train)

    # Get predictions for test data
    # y_test_pred = clf.predict(X_test)

    # Generate multilabel confusion matrices
    # matrices = multilabel_confusion_matrix(y_test, y_test_pred)

    # f = open(estimator_folder + '/results.txt', 'w')

    # to_print = classification_report(y_test, y_test_pred)
    # f.write(to_print)
    # print(to_print)

    # hl = hamming_loss(y_test, y_test_pred)
    # acc = accuracy_score(y_test, y_test_pred)
    # to_print = f"Hamming loss {hl} Accuracy {acc}"
    # f.write(to_print)
    # print(to_print)

    # to_print = f"{estimator} {scores.mean()} {scoring} with a standard deviation of {scores.std()}\n"
    # f.write(to_print)
    # print(to_print)

    clf.fit(X, y)
    pickle.dump(clf, open(estimator_folder + "/clf.p", "wb"))
