import os
from utils.Utils import load_map_from_file, load_list_from_file
import bz2
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
from stop_words import get_stop_words, AVAILABLE_LANGUAGES
import logging
import pycountry
import pandas as pd
import pickle
import numpy as np
np.random.seed(0)
from transformers import pipeline


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.getLogger("gensim").setLevel(logging.ERROR)
logging.getLogger("polyglot").setLevel(logging.ERROR)
trace = logging.getLogger("trace")
trace.setLevel(logging.INFO)
timings = logging.getLogger("timings")
timings.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

def get_stopwords(stopwords_file):
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
                        "boolean", "xml", "httpd", "https"]

    for w in words_to_exclude:
        stop.add(w)

    logger.info(f"Number of Stopwords {len(stop)}")
    return stop


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
                txt = " ".join([str(line.decode("utf-8")).strip("\n") for line in bz2.open(os.path.join(root, filename), "r")])
                data.append([key, doc_id_to_uri[key], domain_to_id[uri_to_gold_class[doc_id_to_uri[key]]], uri_to_gold_class[doc_id_to_uri[key]] , txt])


pickle.dump(data, open("/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/corpus.p", "wb"))

#df = pd.DataFrame(data, columns=['Doc Key', 'Doc URI', 'Class ID', 'Class Label', 'Text'])



#df = df[['Class Label', 'Text']]

#classifier = pipeline("zero-shot-classification")

#candidates_tags = [label for id, label in id_to_domain.items()]
#hit = 0

#for index, row in df.iterrows():
#    print(f"Annotating {index}")
#    print(row['Text'])
#    if classifier(row['Text'], candidates_tags)["labels"][0] == row['Class Label']:
#        print("hit!")
#        hit += 0
#print(hit)


