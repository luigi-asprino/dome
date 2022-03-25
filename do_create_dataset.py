from domainannotators.DocumentDomainAnnotators import SimpleDocumentAnnotator, DocumentAnnotatorAggregationStrategy
from domainannotators.WordAnnotators import RocksDBDomainDisambiguator, AggregationStrategy
from gensim import corpora
from loaders.load_utils import load_dataset, load_benchmark
import logging
import numpy as np
import os
import pandas as pd
import pickle
from preprocessing.Tokenizer import LemmaTokenizer
import shutil
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn_hierarchical_classification.constants import ROOT
from utils.Utils import load_map_from_file, load_list_from_file, load_vectors_from_file, get_stopwords
from utils.ml_utils import DomainTransformer, specialize_annotations, generalise_annotations, \
    restrict_to_top_level_domains
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

use_hierarchy = True
use_tfidf = True
use_domain_annotator = False
on_LOV = True
on_Bench = True
on_LOD_Laundromat = True
flash = False
only_top_level = False
MAXIMAL = "maximal"
INTERSECTION = "intersection"
intersection_or_maximal = MAXIMAL

folder = "/Users/lgu/Desktop/NOTime/EKR/experiments/"

if on_LOV:
   folder += "lov_"
if on_Bench:
    folder += "bench_"
if on_LOD_Laundromat:
    folder += "laund_"
if intersection_or_maximal == INTERSECTION:
    folder += "int_"
elif intersection_or_maximal == MAXIMAL:
    folder += "max_"
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
fc.write(f"Use hierarchy in training/testing phase {use_hierarchy}\n")
fc.write(f"Use hierarchy TF-IDF {use_tfidf}\n")
fc.write(f"Use domain annotator {use_domain_annotator}\n")
fc.write(f"On LOV {on_LOV}\n")
fc.write(f"On Benchmark {on_Bench}\n")
fc.write(f"On LOD Laundromat {on_Bench}\n")
fc.write(f"Only Top-Level {only_top_level}\n")
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
tod_domains_labels = [d for d in class_hierarchy[ROOT]]

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

df = pd.DataFrame(data, columns=['Class Label', 'Text'])
# pickle.dump(df, open(data_file, "wb"))

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

    if not use_hierarchy:
        specialize_annotations(y[0], hierarchy, domain_to_id)
    else:
        generalise_annotations(y[0], hierarchy, domain_to_id)

    if only_top_level:
        restrict_to_top_level_domains(y[0], hierarchy, domain_to_id, tod_domains_labels)
else:
    if not use_hierarchy:
        specialize_annotations(y, hierarchy, domain_to_id)
    else:
        generalise_annotations(y, hierarchy, domain_to_id)

    if only_top_level:
        restrict_to_top_level_domains(y, hierarchy, domain_to_id, tod_domains_labels)

if not use_domain_annotator:
    pickle.dump(pipeline, open(folder + "/pipeline.p", "wb"))

pickle.dump(X, open(folder + "X_pre.p", "wb"))
pickle.dump(y, open(folder + "y_pre.p", "wb"))
pickle.dump(cv, open(folder + "/cv.p", "wb"))

