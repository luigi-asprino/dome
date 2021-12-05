from gensim import corpora
import pickle
import pandas as pd
import numpy as np
import os
import simplemma
from rocksdb.merge_operators import StringAppendOperator
import rocksdb

langdata = simplemma.load_data('en', 'it', 'de', 'es', 'fr', 'nl', 'ru')

import logging as logger
logger.basicConfig( level=logger.INFO)


class SimpleDomainDisambiguator:

    def __init__(self, word_to_id, domain_matrices, id_to_domain):
        self.word_to_id = word_to_id
        self.domain_matrices = domain_matrices
        self.id_to_domain = id_to_domain

    def get_domains(self, word, word_weight=1.0):
        result = np.zeros((len(self.id_to_domain)))
        w = word
        if not w in self.word_to_id:
            w = simplemma.lemmatize(w, langdata)

        if w in self.word_to_id:
            for wd_words in self.word_to_id[w]:
                for domain_matrix in self.domain_matrices:
                    if wd_words[0] in domain_matrix:
                        for d in domain_matrix[wd_words[0]]:
                            result[int(d[0])] += word_weight * d[1]
        return result

    def print_domains(self, domains):
        print(domains)
        domains_extracted = {self.id_to_domain[domain_id]: score for domain_id, score in enumerate(domains)}
        domains_extracted_ordered = {k: v for k, v in sorted(domains_extracted.items(), key=lambda item: item[1])}
        for d in domains_extracted_ordered:
            print(f"{d}\t{domains_extracted_ordered[d]}")


class RocksDBDomainDisambiguator(SimpleDomainDisambiguator):

    def __init__(self, dbpath, id_to_domain):
        self.dbpath = dbpath
        self.id_to_domain = id_to_domain
        self.take_only_max_domain = True
        opts = rocksdb.Options()
        opts.create_if_missing = True
        opts.merge_operator = StringAppendOperator()
        self.db = rocksdb.DB(self.dbpath, opts)
        self.domain_to_id = {v: k for k, v in id_to_domain.items()}

    def extract_domains(self, domain_string):
        result = {}
        split = domain_string.decode().split()
        i = 0
        while i < len(split):
            score = float(split[i + 1])
            if split[i] in result:
                ## domain already within result take the maximum score
                if score > result[split[i]]:
                    result[split[i].replace(",","")] = score
            else:
                result[split[i].replace(",","")] = score
            i += 2
        return result

    def get_domains(self, word, word_weight=1.0):
        result = np.zeros((len(self.id_to_domain)))
        domain_string = self.db.get(word.encode())
        if domain_string == None:
            domain_string = self.db.get(simplemma.lemmatize(word, langdata).encode())

        if domain_string != None:
            domains = self.extract_domains(domain_string)
            if self.take_only_max_domain:
                domain = max(domains, key=domains.get)
                if (domains[domain] <= 1.0 and domains[domain] > 0.0):
                    result[self.domain_to_id[domain]] += word_weight * domains[domain]
            else:
                for domain in domains:
                    if (domains[domain] <= 1.0 and domains[domain] > 0.0):
                        result[self.domain_to_id[domain]] += word_weight * domains[domain]
        return result


def load_corpus(input_folder):

    tfidf_corpus_file = input_folder + "/tfidf_corpus"
    print(f"loading tf-idf corpus from {tfidf_corpus_file}")
    dictionary_file = input_folder + "/dictionary"
    corpus_tfidf = corpora.MmCorpus(tfidf_corpus_file)
    print(f"tf-idf corpus loaded length: {len(corpus_tfidf)}")
    print(f"loading dictionary from {dictionary_file}")
    dictionary = corpora.Dictionary.load(dictionary_file)
    id_to_dictionary_token = {v: k for k, v in dictionary.token2id.items()}
    print("dictionary loaded")
    return corpus_tfidf, dictionary, id_to_dictionary_token


def load_map_from_file(filename):
    if (os.path.exists(filename + ".p")):
        return pickle.load(open(filename + ".p", "rb"))
    df = pd.read_csv(filename, sep='\t', header=None, usecols=[0, 1])
    map = {row[0]: row[1] for index, row in df.iterrows()}
    pickle.dump(map, open(filename + ".p", "wb"))
    return map


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


def print_domains(wd_item, wditem2id, domain_matrix, id_to_domain):
    if wd_item in wditem2id:
        if wditem2id[wd_item] in domain_matrix:
            for domain in domain_matrix[wditem2id[wd_item]]:
                print(id_to_domain[domain])


def annotate(id_to_dictionary_token, doc_id, corpus_tfidf, sdds,
             verbose, words_to_exclude, id_to_domain, compute_word_per_domain=False):

    minus_one=0
    for tf in corpus_tfidf[doc_id]:
        if tf[0]<0:
            minus_one+=1
        if tf[0]> 0  and tf[0] not in id_to_dictionary_token:
            print(f"Not in dictionary: {tf[0]}")

    if minus_one > 0:
        print(f"Words not found {minus_one} doc length {len(corpus_tfidf[doc_id])}")

    doc_words_all = {id_to_dictionary_token[tf[0]]: tf[1] for tf in corpus_tfidf[doc_id] if
                     (tf[0]> 0 and id_to_dictionary_token[tf[0]] not in words_to_exclude)}

    if compute_word_per_domain:
        word_per_domain = {}

    if (verbose):
        print(doc_words_all)

    domain_vector = np.zeros((len(id_to_domain)))
    for w in doc_words_all:
        for sdd in sdds:
            word_domains = sdd.get_domains(w, doc_words_all[w])
            if compute_word_per_domain:
                max_domain = np.argmax(word_domains)
                if max_domain in word_per_domain:
                    word_per_domain[max_domain].append((w,word_domains[max_domain]))
                else:
                    word_per_domain[max_domain] = [(w,word_domains[max_domain])]
            domain_vector += word_domains

    norm = np.linalg.norm(domain_vector)
    if norm > 0:
        domain_vector = domain_vector / norm

    if compute_word_per_domain:
        return domain_vector, doc_words_all, word_per_domain

    return domain_vector, doc_words_all




def load_domain_hierarchy(filename, uri2id):
    df = pd.read_csv(filename, header=None)
    domain_hierarchy_names = {row[0].strip(): row[1].split() for index, row in df.iterrows()}
    domain_hierarchy_ids = dict()
    for domain in domain_hierarchy_names.keys():
        if domain in uri2id:
            domain_hierarchy_ids[uri2id[domain]] = [uri2id[sub_domain] for sub_domain in domain_hierarchy_names[domain]
                                                    if sub_domain in uri2id]
    return domain_hierarchy_ids


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')

    # --input_folder resources/input_wd_6/ --input_folder_corpus /Users/lgu/Desktop/NOTime/EKR/Corpus_lod --input_wn_folder /Users/lgu/workspace/ekr/dome/resources/wn_resources --out_folder /Users/lgu/workspace/ekr/dome/outputs/output_8

    parser.add_argument('--input_folder', dest='input_folder', default='resources/input_wd_6/', help='The folder containing ... ')
    parser.add_argument('--input_folder_corpus', dest='input_folder_corpus', default='resources/input_wd_6/', help='The folder containing ... ')
    parser.add_argument('--input_wn_folder', dest='input_wn_folder', default='resources/wn_resources/', help='The folder containing ... ')
    parser.add_argument('--out_folder', dest='out_folder', help='The folder containing ... ')
    parser.add_argument('--limit', dest='limit', help='The folder containing ... ')

    args = parser.parse_args()

    #input_folder = "resources/input_wd_6/"
    input_folder = args.input_folder

    #input_folder_corpus = "/Users/lgu/Desktop/NOTime/EKR/Corpus_lod/"
    #input_folder_corpus = sys.argv[2]

    input_folder_corpus = args.input_folder_corpus

    #input_wn_folder = "resources/wn_resources/"
    #input_wn_folder = sys.argv[3]

    input_wn_folder = args.input_wn_folder

    #out_folder = "outputs/output_7/"
    #out_folder = sys.argv[4]
    out_folder = args.out_folder

    if args.limit:
        limit = int(args.limit)
        print(f"limit: {limit}")


    print(f"input_folder: {input_folder}\ninput_folder_corpus: {input_folder_corpus} \ninput_wn_folder: {input_wn_folder}\nout_folder{out_folder}")

    output_folder_wiki = out_folder + "/wiki/"
    output_folder_bn = out_folder + "/bn/"
    output_folder_wn_bn = out_folder + "/wn_bn/"
    output_folder_wiki_wn_bn = out_folder + "/wiki_wn_bn/"
    red = False
    verbose = False
    test = False
    write_detailed_results = True

    if (not os.path.isdir(output_folder_wiki)):
        os.makedirs(output_folder_wiki)

    if (not os.path.isdir(output_folder_bn)):
        os.makedirs(output_folder_bn)

    if (not os.path.isdir(output_folder_wiki_wn_bn)):
        os.makedirs(output_folder_wiki_wn_bn)

    if (not os.path.isdir(output_folder_wn_bn)):
        os.makedirs(output_folder_wn_bn)

    corpus_tfidf, dictionary, id_to_dictionary_token = load_corpus(input_folder_corpus)

    print("loading word_domain_matrix_wn")
    domain_matrix_wn = load_matrix_from_file(input_folder + "word_domain_matrix_wn")
    print("word_domain_matrix_wn loaded")

    print("loading wordnet word domain matrix")
    domain_matrix_wordnet = load_matrix_from_file(input_wn_folder + "word_domain_matrix_ekr", exclude_null=True)
    print("wordnet word domain matrix loaded")

    print("loading word_domain_matrix_bn")
    domain_matrix_bn = load_matrix_from_file(input_folder + "word_domain_matrix_bn", True)
    print("word_domain_matrix_bn loaded")

    print("loading word_domain_matrix")
    if red:
        domain_matrix = load_matrix_from_file(input_folder + "word_domain_matrix_red")
    else:
        domain_matrix = load_matrix_from_file(input_folder + "word_domain_matrix")
    print("word_domain_matrix loaded")

    print("loading wordIDS")
    if red:
        wditem2id = load_map_from_file(input_folder + "wordIDs_red")
    else:
        wditem2id = load_map_from_file(input_folder + "wordIDs")
    print("wordIDS loaded")

    print("loading domain2id map")
    domain_to_id = load_map_from_file(input_folder + "domain2id")
    id_to_domain = {v: k for k, v in domain_to_id.items()}
    print("domain2id map loaded")

    print("loading word2id map")
    if (not red):
        word_to_id = load_matrix_from_file(input_folder + "word2id")
    else:
        word_to_id = load_matrix_from_file(input_folder + "word2id_red")
    # id_to_word = {v: k for k, v in word_to_id.items()}
    print("word2id map loaded")

    wn_word_to_id = load_matrix_from_file(input_wn_folder + "word2id", lowercase=True)

    print("loading domain hierarchy")
    domain_hierarchy_ids = load_domain_hierarchy(input_folder + "domain_hierarchy.csv", domain_to_id)
    print("domain hierarchy loaded")

    if args.uri_to_ontology_id:
        print("Loading Id2OntologyUri")
        uriOntology2Id = load_map_from_file(input_folder + "uriOntology2Id.tsv")
        Id2OntologyUri = {v: k for k, v in uriOntology2Id.items()}
        print("Id2OntologyUri map loaded")

        print("Loading doc_ids")
        df = pd.read_csv(input_folder + "doc_ids", sep='\t', header=None, usecols=[0])
        doc_ids = [row[0] for index, row in df.iterrows()]
        print("doc_ids loaded")

    else:
        Id2OntologyUri = {}
        doc_ids = None



    words_to_exclude = ["property", "label", "comment", "class", "restriction", "ontology", "nil", "individual",
                        "value", "domain", "range", "first", "rest", "resource", "datatype", "integer", "equivalent",
                        "title", "thing", "creator", "disjoint", "predicate", "dublin", "taxonomy", "axiom", "foaf", "dc",
                        "boolean", "xml", "httpd", "https"]

    sdd_wn = SimpleDomainDisambiguator(wn_word_to_id, [domain_matrix_wordnet], id_to_domain)

    sdd_wd = SimpleDomainDisambiguator(word_to_id, [domain_matrix, domain_matrix_wn, domain_matrix_bn], id_to_domain)

    dd_db = RocksDBDomainDisambiguator("resources/bn", id_to_domain)

    domain_vectors = np.zeros((len(corpus_tfidf),len(id_to_domain)))

    if not args.limit:
        limit = len(corpus_tfidf)

    #dv, words = annotate(id_to_dictionary_token, 7, corpus_tfidf, [sdd_wn, dd_db],
    #         verbose, words_to_exclude, id_to_domain)

    #dd_db.print_domains(dv)

    #print(words)
    #quit()

    for doc_id in range(0, min(len(corpus_tfidf), limit)):

        print(f"Annotating {doc_id}")

        domain_vectors[doc_id],doc_words_all = annotate(id_to_dictionary_token, doc_id, corpus_tfidf, [sdd_wn, dd_db],
                 verbose, words_to_exclude, id_to_domain)

        if test:
            annotate(id_to_dictionary_token, doc_id, corpus_tfidf, [sdd_wd],
                     verbose, words_to_exclude, id_to_domain)
            annotate(id_to_dictionary_token, doc_id, corpus_tfidf, [sdd_wd, sdd_wn, dd_db],
                     verbose, words_to_exclude, id_to_domain)

        if write_detailed_results:

            domains = {i: score for i, score in enumerate(domain_vectors[doc_id]) if score > 0.0}

            # getting URIs of top scoring domains
            if (len(domains) > 0):
                domains_extracted = {id_to_domain[domain_id]: domains[domain_id] for domain_id in domains}
                # writing results TODO
                domains_extracted_ordered = {k: v for k, v in sorted(domains_extracted.items(), key=lambda item: item[1], reverse=True)}

                fw_summary = open(output_folder_wn_bn + "/summary_" + str(doc_id), 'w')

                if args.uri_to_ontology_id:
                    fw_summary.write(f"Ontology {doc_ids[doc_id]}")
                    if doc_ids[doc_id] in Id2OntologyUri:
                        fw_summary.write(f" URI {Id2OntologyUri[doc_ids[doc_id]]}\n")
                    else:
                        fw_summary.write(f"\n")

                for d in domains_extracted_ordered:
                    # print(f"{d}\t{domains_extracted_ordered[d]}")
                    fw_summary.write(f"{d}\t{domains_extracted_ordered[d]}\n")
                fw_summary.write(f"\n\nVDOC\n\n")
                words_ordered = {k: v for k, v in sorted(doc_words_all.items(), key=lambda item: item[1], reverse=True)}

                for word in words_ordered:
                    fw_summary.write(f"\t{word}\t{words_ordered[word]}\n")

                fw_summary.close()


    pickle.dump(domain_vectors, open(out_folder+"/domain_model.p", "wb"))
    pickle.dump(id_to_domain, open(out_folder + "/id_to_domain.p", "wb"))
