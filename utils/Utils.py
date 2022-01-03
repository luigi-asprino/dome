import pickle
import pandas as pd
import os
import pycountry
from stop_words import get_stop_words, AVAILABLE_LANGUAGES
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def load_list_from_file(filename, token_number=0, extractid=False):
    if (os.path.exists(filename + ".p")):
        return pickle.load(open(filename + ".p", "rb"))
    df = pd.read_csv(filename, sep='\t', header=None, usecols=[0])
    if extractid:
        result = [row[0].split("/")[token_number] for index, row in df.iterrows()]
    else:
        result = [row[0] for index, row in df.iterrows()]
    pickle.dump(result, open(filename + ".p", "wb"))
    return result


def load_map_from_file(filename):
    if os.path.exists(filename + ".p"):
        return pickle.load(open(filename + ".p", "rb"))
    df = pd.read_csv(filename, sep='\t', header=None, usecols=[0, 1])
    res = {row[0]: row[1] for index, row in df.iterrows()}
    pickle.dump(res, open(filename + ".p", "wb"))
    return res


def load_vectors_from_file(filename, id_col=0, sep='\t', header=None, usecols=None, nullstring=None):
    df = pd.read_csv(filename, sep=sep, header=header, usecols=usecols)
    res = {row[id_col]: [el for el in row[id_col+1:]] for index, row in df.iterrows()}
    if nullstring is not None:
        for k in res:
            vals = res[k]
            res[k] = [v for v in vals if v is not nullstring]
    if header is not None:
        return res, [lab for lab in df.columns[id_col+1:]]
    else:
        return res


def load_matrix_from_file(filename, loadscore=False, exclude_null=False, lowercase=False):
    if (os.path.exists(filename + ".p")):
        return pickle.load(open(filename + ".p", "rb"))

    # print(f"Rows {rows} Cols {cols}")
    matrix = {}
    if loadscore:
        df = pd.read_csv(filename, sep='\t', header=None, usecols=[0, 1, 2])
    else:
        df = pd.read_csv(filename, sep='\t', header=None, usecols=[0, 1])
    for index, row in df.iterrows():
        if exclude_null and row[1] == "null":
            print("null" + row[0])
            continue
        k = row[0]
        if lowercase:
            if isinstance(k, str):
                k = k.lower()

        if k in matrix:
            if loadscore:
                matrix[k].append((row[1], float(row[2])))
            else:
                matrix[k].append((row[1], 1.0))
        else:
            if loadscore:
                matrix[k] = [(row[1], float(row[2]))]
            else:
                matrix[k] = [(row[1], 1.0)]

    pickle.dump(matrix, open(filename + ".p", "wb"))

    return matrix


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