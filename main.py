import numpy as np
import faiss  # make faiss available
from nltk import everygrams
import pandas as pd
import os
from gensim import corpora
from multiprocessing.dummy import Pool as ThreadPool



def test_faiss():
    d = 64  # dimension
    nb = 100000  # database size
    nq = 10000  # nb of queries
    np.random.seed(1234)  # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.
    index = faiss.IndexFlatL2(d)  # build the index
    print(index.is_trained)
    index.add(xb)  # add vectors to the index
    print(index.ntotal)
    k = 4  # we want to see 4 nearest neighbors
    D, I = index.search(xb[:5], k)  # sanity check
    print(I)
    print(D)
    D, I = index.search(xq, k)  # actual search
    print(I[:5])  # neighbors of the 5 first queries
    print(I[-5:])  # neighbors of the 5 last queries

def word_list_to_trigram_matrix(word_list,  trigramToId):
    xb = np.zeros((len(word_list), len(trigramToId)), dtype="float32")

    for i in range(0, len(word_list)):

        # if len(word_list[i])< 3:
        #     if word_list[i] in trigramToId:
        #         xb[i][trigramToId[word_list[i]]] =  1.0

        #print(f"Getting ngrams of {word_list[i]}")

        trigrams = get_ngrams(word_list[i])
        for trigram in trigrams:
            k = trigram[0] + trigram[1] + trigram[2]
            if k in trigramToId:
                xb[i][trigramToId[k]] = xb[i][trigramToId[k]] + (1.0 / len(trigrams))

    return xb

def get_ngrams(word):
    return list(everygrams(str(word).lower(), 3, 3))

def get_trigram_matrix(word_list):
    m = dict()
    trigramToId = dict()
    trigramId = 0
    for w in word_list:

        # if len(w)<3:
        #     if w not in m:
        #         m[w] = 1
        #         trigramToId[w] = trigramId
        #         trigramId += 1
        #     else:
        #         m[w] = m[w] + 1
        #
        #     break
        #print(f"Getting n-grams of {w}")
        for trigram in get_ngrams(w):
            k = trigram[0] + trigram[1] + trigram[2]
            if k not in m:
                m[k] = 1
                trigramToId[k] = trigramId
                trigramId+=1
            else:
                m[k] = m[k] + 1
    return m, trigramToId

def create_index(xb, d):
    index = faiss.IndexFlatL2(d)  # build the index
    index.add(xb)  # add vectors to the index
    print("Index created")
    return index

def create_index_from_file(filename):
    df = pd.read_csv(filename, sep='\t', header=None, usecols=[0])
    word_list = [row[0] for index, row in df.iterrows()]
    #print(word_list)
    #f = open(filename)
    #word_list = [line.rstrip() for line in f]
    trigram_matrix, trigramToId = get_trigram_matrix(word_list)
    print("Trigram matrix computed")

    d = len(trigramToId)  # dimension
    nb = len(word_list)  # database size
    xb = word_list_to_trigram_matrix(word_list,  trigramToId)

    index = create_index(xb,d)

    return index, word_list, trigramToId

def load_map_from_file(filename):
    df = pd.read_csv(filename, sep='\t', header=None, usecols=[0,1])
    map = {row[0]:row[1] for index, row in df.iterrows()}
    return map

def print_domains_of_word(word, word2id, id2domains, word_domain_matrix):
    domain_distribution = word_domain_matrix[word2id[word]]
    for i in range(0,len(domain_distribution)):
        if domain_distribution[i]>0:
            print(f"{word} {id2domains[i]}")


def load_matrix_from_file(filename, rows, cols):
    #print(f"Rows {rows} Cols {cols}")
    matrix = np.zeros((rows, cols), dtype="float32")
    df = pd.read_csv(filename, sep='\t', header=None, usecols=[0,1])
    for index, row in df.iterrows():
        matrix[row[0]][row[1]] = 1.0
    return matrix

def load_domain_hierarchy(filename, uri2id):
    df = pd.read_csv(filename, quotechar="\"", header=0)
    domain_hierarchy_names = {row[0].strip(): row[1].split() for index, row in df.iterrows()}
    domain_hierarchy_ids = dict()
    for domain in domain_hierarchy_names.keys():
        if domain in uri2id:
            domain_hierarchy_ids[uri2id[domain]]=[uri2id[sub_domain]   for sub_domain in domain_hierarchy_names[domain] if sub_domain in uri2id]
    return domain_hierarchy_ids



if __name__ == '__main__':
    # test_faiss()
    input_folder = "input/"
    domain_to_id = load_map_from_file(input_folder + "domain2id")
    id_to_domain = {v: k for k, v in domain_to_id.items()}
    print("domain2id map loaded")

    top_level = ["https://w3id.org/framester/wn/wn30/wndomains/wn-domain-factotum",
                 "https://w3id.org/framester/wn/wn30/wndomains/wn-domain-free_time",
                 "https://w3id.org/framester/wn/wn30/wndomains/wn-domain-pure_science",
                 "https://w3id.org/framester/wn/wn30/wndomains/wn-domain-applied_science",
                 "https://w3id.org/framester/wn/wn30/wndomains/wn-domain-social_science",
                 "https://w3id.org/framester/wn/wn30/wndomains/wn-domain-doctrines"]

    domain_hierarchy_ids = load_domain_hierarchy(input_folder + "domain_hierarchy.csv", domain_to_id)

    index, word_list, trigramToId = create_index_from_file(input_folder + "word2id")

    tfidf_corpus_file = input_folder + "tfidf_corpus"
    dictionary_file = input_folder + "dictionary"
    corpus_tfidf = corpora.MmCorpus(tfidf_corpus_file)
    print("corpus tf-idf loaded")
    dictionary = corpora.Dictionary.load(dictionary_file)
    id_to_dictionary_token = {v: k for k, v in dictionary.token2id.items()}
    print("dictionary loaded")
    k=1

    wnword_to_id = load_map_from_file(input_folder + "word2id")
    id_to_wnword = {v: k for k, v in wnword_to_id.items()}
    print("word2id map loaded")

    #wnworduri_to_id = load_map_from_file(input_folder + "uri2id")
    #id_to_wnworduri = {v: k for k, v in wnworduri_to_id.items()}
    #print("uri2id map loaded")

    num_domains = max(id_to_domain.keys()) + 1
    num_wn_words = max(id_to_wnword.keys()) + 1
    word_domain_matrix = load_matrix_from_file(input_folder + "word_domain_matrix", num_wn_words, num_domains)
    print("word_domain_matrix map loaded")

    doc_domain_matrix = np.zeros((len(corpus_tfidf),num_domains))

    uriOntology2Id = load_map_from_file(input_folder + "uriOntology2Id.tsv")
    Id2OntologyUri = {v: k for k, v in uriOntology2Id.items()}
    print("Id2OntologyUri map loaded")

    df = pd.read_csv(input_folder + "doc_ids", sep='\t', header=None, usecols=[0])
    doc_ids = [row[0] for index, row in df.iterrows()]

    outfolder = "annotated_domains"
    if (not os.path.exists(outfolder)):
        os.mkdir(outfolder)

    words_to_exclude = ["property", "label", "comment", "class", "restriction", "ontology", "nil", "individual", "value", "domain", "range", "first", "rest", "datatype", "integer"]

    ### TODO Convert Dictionary Not Documents!

    for doc_id in range(0, len(corpus_tfidf)):
        doc_words = [id_to_dictionary_token[tf[0]] for tf in corpus_tfidf[doc_id]]
        print(f"Processing file {doc_id} size: {len(doc_words)} {Id2OntologyUri[doc_ids[doc_id]]}")
        xq = word_list_to_trigram_matrix(doc_words, trigramToId)
        D, I = index.search(xq, k)
        for i in range(0, len(xq)):
            for ii in range(0, k):
                wn_word_id = I[i][ii] # wn word_id
                sim = 1 - D[i][ii]
                if not wn_word_id in id_to_wnword:
                    print(f"Couldn't find wordnet word with id {wn_word_id}")
                    continue

                #if not id_to_wnword[wn_word_id] in words_to_exclude: # exclude words from domain computation
                for domain_id in range(0, num_domains):
                    if word_domain_matrix[wn_word_id][domain_id]>0:
                        doc_domain_matrix[doc_id][domain_id] += sim * corpus_tfidf[doc_id][i][1] # word simiarity * tf-idf of word

        # use domain hierarchy to reinforce top-level domains
        for domain_id in range(0, num_domains):
            if domain_id in domain_hierarchy_ids:
                for sub_domain in domain_hierarchy_ids[domain_id]:
                    doc_domain_matrix[doc_id][domain_id] += doc_domain_matrix[doc_id][sub_domain]

        doc_domain_matrix[doc_id][domain_to_id["https://w3id.org/framester/wn/wn30/wndomains/wn-domain-factotum"]] = 0.0 # exclude factotum domain

        # normalize vector
        #  FIXME /Users/lgu/workspace/ekr/dome/main.py:213: RuntimeWarning: invalid value encountered in true_divide
        #   doc_domain_matrix[doc_id] = doc_domain_matrix[doc_id] / np.linalg.norm(doc_domain_matrix[doc_id])
        doc_domain_matrix[doc_id] = doc_domain_matrix[doc_id] / np.linalg.norm(doc_domain_matrix[doc_id])

    print("domains computed")



    fw_summary = open(outfolder + "/summary" , 'w')
    for doc_id in range(0, len(corpus_tfidf)):
        domain_distribution_sorted = sorted([(doc_domain_matrix[doc_id][domain_id], id_to_domain[domain_id]) for domain_id in range(0, num_domains)], reverse=True)
        fw = open(outfolder+"/"+str(doc_id), 'w')
        fw.write(f"Ontology {doc_ids[doc_id]} URI {Id2OntologyUri[doc_ids[doc_id]]}\n\n")

        for s, d in domain_distribution_sorted:
            if d in top_level:
                fw.write(f"{d}\t{s}\tTOP LEVEL\n")
            else:
                fw.write(f"{d}\t{s}\n")
        fw.close()

        fw_summary.write(f"{doc_ids[doc_id]}\t{Id2OntologyUri[doc_ids[doc_id]]}\t{domain_distribution_sorted[0][1]}\t{domain_distribution_sorted[0][0]}\n")

    fw_summary.close()


        # for domain_id in range(0, num_domains):
        #     #print(f"{doc_id}\t{id2domain[domain_id]}\t{doc_domain_matrix[doc_id][domain_id]}")
        #     fw.write(f"{id2domain[domain_id]}\t{doc_domain_matrix[doc_id][domain_id]}\n")
        # fw.close()

    print_domains_of_word("ontology", wnword_to_id, id_to_domain, word_domain_matrix)
