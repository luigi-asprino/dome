import pickle
from gensim import corpora
import numpy as np
from utils.corpora import get_doc
from utils.Utils import load_list_from_file, load_map_from_file


def get_doc_feature(dic_id_to_feature_id_map, doc_in, features_provided):
    doc_res = np.zeros(len(features_provided))
    for w, s in doc_in:
        if w in dic_id_to_feature_id_map:
            doc_res[dic_id_to_feature_id_map[w]] = 1.0
    return doc_res


experiment_dir = "/Users/lgu/Desktop/NOTime/EKR/experiments/lov_bench_laund_max_hp_bin/"
dic_id_to_feature_id_file = experiment_dir + "dic_id_to_features_id.p"
features_file = experiment_dir + "features.p"
svd_file = experiment_dir + "svd.p"
laundromat_corpus = "/Users/lgu/Dropbox/Backups/Corpus_lod"
tfidf_corpus_file = laundromat_corpus + "/tfidf_corpus"
dictionary_file = laundromat_corpus + "/dictionary"
pipe_file = experiment_dir + "7/MLSMOTE/MultiOutputClassifier-XGBClassifier_pipe.p"
mlb_file = experiment_dir + "mlb.p"

dictionary = corpora.Dictionary.load(dictionary_file)
corpus_tfidf = corpora.MmCorpus(tfidf_corpus_file)
dic_id_to_feature_id = pickle.load(open(dic_id_to_feature_id_file, "rb"))
svd = pickle.load(open(svd_file, "rb"))
features = pickle.load(open(features_file, "rb"))
pipe = pickle.load(open(pipe_file, "rb"))
mlb = pickle.load(open(mlb_file, "rb"))
doc_ids_laun = load_list_from_file(laundromat_corpus + "/doc_ids", 8, extractid=True)
source_to_doc_id = load_map_from_file("/Users/lgu/Desktop/NOTime/EKR/Corpus_lod_4/sources.txt")
doc_id_to_source = {v: k for k, v in source_to_doc_id.items()}

f = open(experiment_dir + "annotations.tsv", "w")
result = {}

for idx, d in enumerate(corpus_tfidf):
    svd_d = svd.transform([get_doc_feature(dic_id_to_feature_id, d, features)])
    labels = mlb.inverse_transform(pipe.predict(svd_d))
    doc = [word for word, score in get_doc(d, dictionary)]
    if len(doc) > 50:
        description = " ".join(doc[:51])
    else:
        description = " ".join(doc)
    ll = " ".join(labels[0])
    result[doc_ids_laun[idx]] = labels[0]
    print(f"#{idx} dataset annotated {ll}")
    if doc_ids_laun[idx] in doc_id_to_source:
        f.write(f"{doc_ids_laun[idx]} {doc_id_to_source[doc_ids_laun[idx]]} ::::::: {ll} ::::::: {description}\n")
    else:
        f.write(f"{doc_ids_laun[idx]} ::::::: {ll} ::::::: {description}\n")
pickle.dump(result, open(experiment_dir + "annotated_laundromat.p", "wb"))

f.flush()
f.close()

