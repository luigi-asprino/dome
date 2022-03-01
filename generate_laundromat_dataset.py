from utils.Utils import load_map_from_file, load_list_from_file
from utils.corpora import CorpusFiltered, get_doc
from gensim import corpora
import random
import logging
logging.getLogger("utils.corpora").setLevel(logging.DEBUG)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

source_to_cluster = load_map_from_file("/Users/lgu/Desktop/NOTime/EKR/Corpus_lod_4/KMeansOnSource/clusters_128.txt")
source_to_id = load_map_from_file("/Users/lgu/Desktop/NOTime/EKR/Corpus_lod_4/sources.txt")
doc_ids = load_list_from_file("/Users/lgu/Dropbox/Backups/Corpus_lod/doc_ids", 8, extractid=True)
id_to_docn = {iddoc: docn for docn, iddoc in enumerate(doc_ids)}
corpus_tfidf = corpora.MmCorpus("/Users/lgu/Dropbox/Backups/Corpus_lod/tfidf_corpus")
dictionary = corpora.Dictionary.load("/Users/lgu/Dropbox/Backups/Corpus_lod/dictionary")
corpus_tfidf_filtered = CorpusFiltered(corpus_tfidf, 0.9)

cluster_to_sources = {}
counter = 0
empty_docs = 0
i = 0

for source, cluster_id in source_to_cluster.items():
    i += 1
    if i%10000==0:
        print(i)
    if source not in source_to_id:
        print(f"Not in source_to_id {source}")
        continue
    elif source_to_id[source] not in id_to_docn:
        print(f"Not in id_to_docn {source} {source_to_id[source]}")
        counter += 1
        continue

    if not corpus_tfidf[id_to_docn[source_to_id[source]]]:
        print(f"Empty DOC {source} {source_to_id[source]}")
        empty_docs += 1
        continue

    if cluster_id in cluster_to_sources:
        cluster_to_sources[cluster_id].append(source)
    else:
        cluster_to_sources[cluster_id] = [source]

print(f"Number of sources not in id_to_docn {counter} Empty docs {empty_docs}")
for cluster_id, cluster in cluster_to_sources.items():
    print(f"Cluster #{cluster_id}\t{len(cluster)}")

import pickle
pickle.dump(cluster_to_sources, open("/Users/lgu/Desktop/cluster_to_sources.p", "wb"))

rows = []

for cluster_id, sources in sorted(cluster_to_sources.items(), key=lambda x: x[0]):
    #print(f"Cluster {cluster_id} {len(sources)} {random.choices(sources, k=6)}")
    random.shuffle(sources)
    if len(sources) > 10000:
        chosen_sources = sources[:7]
    else:
        chosen_sources = sources[:6]
    for chosen_source in chosen_sources:
        if chosen_source in source_to_id:
            doc = [word for word, score in get_doc(corpus_tfidf_filtered[id_to_docn[source_to_id[chosen_source]]], dictionary)]
            if len(doc) > 50:
                description = " ".join(doc[:51])
            else:
                description = " ".join(doc)

            rows.append([cluster_id, len(sources), source_to_id[chosen_source], id_to_docn[source_to_id[chosen_source]], chosen_source, description])


            #f.write(f"{cluster_id}\t{len(sources)}\t{source_to_id[chosen_source]}\t{chosen_source}\t{id_to_docn[source_to_id[chosen_source]]}\t{description}\n")






random.shuffle(rows)


f = open(f"/Users/lgu/Desktop/annotation_all.tsv", "w")
for row in rows:
    for col in row:
        f.write(str(col))
        f.write("\t")
    f.write("\n")
f.close()

i = 0
for chunk in chunks(rows, 250):
    f = open(f"/Users/lgu/Desktop/annotation_{i}.tsv", "w")
    for row in chunk:
        for col in row:
            f.write(str(col))
            f.write("\t")
        f.write("\n")
    f.close()
    i += 1

