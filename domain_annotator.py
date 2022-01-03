from gensim import corpora
import pandas as pd
import numpy as np
import os
import simplemma
from domainannotators.WordAnnotators import RocksDBDomainDisambiguator, AggregationStrategy
from utils.Utils import load_map_from_file, load_list_from_file
from domainannotators.DocumentDomainAnnotators import SimpleDocumentAnnotator, DocumentAnnotatorAggregationStrategy

langdata = simplemma.load_data('en', 'it', 'de', 'es', 'fr', 'nl', 'ru')

import logging as logger

logger.basicConfig(level=logger.INFO)


def load_domain_hierarchy(filename, uri2id):
    df = pd.read_csv(filename, header=None)
    domain_hierarchy_names = {row[0].strip(): row[1].split() for index, row in df.iterrows()}
    domain_hierarchy_ids = dict()
    for domain in domain_hierarchy_names.keys():
        if domain in uri2id:
            domain_hierarchy_ids[uri2id[domain]] = [uri2id[sub_domain] for sub_domain in domain_hierarchy_names[domain]
                                                    if sub_domain in uri2id]
    return domain_hierarchy_ids


def get_top_domain(vector, id_to_domain):
    return id_to_domain[np.argmax(vector)]


def args():
    import argparse

    parser = argparse.ArgumentParser(description='Annotate each document of the corpus with its knowledge domain.')

    parser.add_argument('--id_to_domain', dest='id_to_domain', help='A TSV file mapping each domain id to a knowledge '
                                                                    'domain.')
    parser.add_argument('--corpus_folder', dest='corpus_folder', help='The folder containing the corpus.')
    parser.add_argument('--out_folder', dest='out_folder', help='The path to the output folder.')
    parser.add_argument('--lemma_to_domain_db', dest='lemma_to_domain_db', nargs='+',
                        help='The path to the RocksDB containing mappings of the lemmas to knowledge domains.')
    parser.add_argument('--uri_to_doc_id', dest='uri_to_doc_id',
                        help='The filepath to a TSV file containing the mapping of the URIs of the datasets to the ids '
                             'of the document.')
    parser.add_argument('--token_number', dest='token_number',
                        help='The number of token of the path containing the id of the doc.', type=int)
    parser.add_argument('--inspect', dest='inspect',
                        help='A comma separated list of document ids to inspect e.g. 0,10,23,42')
    parser.add_argument('--gold_standard', dest='gold_standard',
                        help='A TSV containing gold standard annotations for each dataset URI.')
    parser.add_argument('--limit', dest='limit', help='The maximum number of documents to process.')
    parser.add_argument('--hierarchy', dest='hierarchy_path',
                        help='A TSV the hierarchical relations among knowledge domains.')
    parser.add_argument('--words_to_inspect', dest='words_to_inspect', nargs='+',
                        help='A list of words for which it will be printed the their distribution over knowledge domains.')

    args = parser.parse_args()

    docs_to_inspect = []
    if args.inspect:
        for d in args.inspect.split(","):
            docs_to_inspect.append(int(d.strip()))

    return args, args.corpus_folder, args.out_folder, docs_to_inspect, args.lemma_to_domain_db


def load_resources(input_folder_corpus, args):
    tfidf_corpus_file = input_folder_corpus + "/tfidf_corpus"
    print(f"loading tf-idf corpus from {tfidf_corpus_file}")
    dictionary_file = input_folder_corpus + "/dictionary"
    corpus_tfidf = corpora.MmCorpus(tfidf_corpus_file)
    print(f"tf-idf corpus loaded length: {len(corpus_tfidf)}")
    print(f"loading dictionary from {dictionary_file}")
    dictionary = corpora.Dictionary.load(dictionary_file)
    id_to_dictionary_token = {v: k for k, v in dictionary.token2id.items()}
    print("dictionary loaded")

    id_to_domain = load_map_from_file(args.id_to_domain)
    domain_to_id = {k: v for v, k in id_to_domain.items()}
    print(domain_to_id)

    hierarchy = {}
    if (args.hierarchy_path):
        for (k, v) in load_map_from_file(args.hierarchy_path).items():
            hierarchy[int(k)] = [int(kd.strip()) for kd in v.split(",")]

    doc_ids = load_list_from_file(input_folder_corpus + "/doc_ids", args.token_number, extractid=True)
    id2doc = {k: v for v, k in enumerate(doc_ids)}

    uri_to_doc_id = load_map_from_file(args.uri_to_doc_id)
    doc_id_to_uri = {k: v for v, k in uri_to_doc_id.items()}

    words_to_exclude = ["property", "label", "comment", "class", "restriction", "ontology", "nil", "individual",
                        "value", "domain", "range", "first", "rest", "resource", "datatype", "integer", "equivalent",
                        "title", "thing", "creator", "disjoint", "predicate", "dublin", "taxonomy", "axiom", "foaf",
                        "dc", "uri", "void", "dataset", "subject", "term", "agent",
                        "boolean", "xml", "httpd", "https"]

    lemma_dbs = []
    i = 0
    while i < len(lemma_to_domain_dbs):
        print(f"Loading DB from path {lemma_to_domain_dbs[i]} {lemma_to_domain_dbs[i + 1]}")
        db = RocksDBDomainDisambiguator(lemma_to_domain_dbs[i], lemma_to_domain_dbs[i + 1], id_to_domain, hierarchy=hierarchy, strategy=AggregationStrategy.MAX)
        lemma_dbs.append(db)
        i = i + 2

    if args.words_to_inspect:
        for word_to_inspect in args.words_to_inspect:
            for ldb in lemma_dbs:
                print(f"DB name {ldb.name}")
                print(ldb.print_domains(ldb.get_domains(word_to_inspect)))
                #print(ldb.print_domains(ldb.get_domains_normalised(word_to_inspect)))
                print("\n\n")

    for ldb in lemma_dbs:
        print(ldb.get_number_of_words_per_domain())
        print(ldb.get_number_of_words_per_domain()[0])

    uri_to_gold_class = {}
    if (args.gold_standard):
        uri_to_gold_class = load_map_from_file(args.gold_standard)

    da = SimpleDocumentAnnotator(id_to_dictionary_token, id_to_domain, lemma_dbs, words_to_exclude, strategy=DocumentAnnotatorAggregationStrategy.SUM_WORD_MAX)

    return corpus_tfidf, dictionary, id_to_dictionary_token, id_to_domain, domain_to_id, doc_ids, id2doc, uri_to_doc_id, doc_id_to_uri, words_to_exclude, lemma_dbs, uri_to_gold_class, hierarchy, da


def inspect_document(domain_vectors, doc_id,  corpus_tfidf, id_to_domain, da):
    domain_vectors[doc_id], doc_words_all = da.get_domain_vector(corpus_tfidf[doc_id])
    word_per_domain, not_domain = da.inspect_document(doc_words_all)

    print("\n\n")
    uri = doc_id_to_uri[doc_ids[doc_id]]
    predicted_category = get_top_domain(domain_vectors[doc_id], id_to_domain)
    gold_category = uri_to_gold_class[uri] if uri in uri_to_gold_class else ""
    print(
        f"\n\n{uri} pred:'{predicted_category}' {domain_vectors[doc_id][domain_to_id[predicted_category]]} :: {gold_category}")
    print(f"{sorted(not_domain, key=lambda item: item[1], reverse=True)}")
    print(f"{sorted(doc_words_all.items(), key=lambda item: item[1], reverse=True)}")

    domains_ordered = [(domain_id, score) for domain_id, score in
                       sorted(enumerate(domain_vectors[doc_id]), key=lambda item: item[1], reverse=True) if score]

    for domain_id, score in domains_ordered:
        print(id_to_domain[domain_id] + "\t" + str(score))
        for word in sorted(word_per_domain[domain_id], key=lambda item: item[1], reverse=True):
            print("\t", str(word), end=" ")
        print()


def annotate_documents(corpus_tfidf, limit, id_to_domain, doc_id_to_uri, doc_ids, domain_vectors, uri_to_gold_class,
                       hierarchy, da):
    doc_in_gold_standard = 0
    correct = 0
    subsumed = 0

    for doc_id in range(0, min(len(corpus_tfidf), limit)):

        domain_vectors[doc_id], doc_words_all = da.get_domain_vector(corpus_tfidf[doc_id])

        uri = doc_id_to_uri[doc_ids[doc_id]]
        predicted_category = get_top_domain(domain_vectors[doc_id], id_to_domain)
        gold_category = uri_to_gold_class[uri] if uri in uri_to_gold_class else ""

        if gold_category == "" or gold_category == "Time" or gold_category == "Scientific Research & Academy":
            continue

        print(f"{doc_id}) {uri} {predicted_category}::{gold_category}")
        if uri in uri_to_gold_class:
            doc_in_gold_standard += 1
        if predicted_category == gold_category:
            print("Correct")
            correct += 1
        if gold_category != "" and domain_to_id[gold_category] in hierarchy and domain_to_id[predicted_category] in \
                hierarchy[domain_to_id[gold_category]]:
            ## check whether the predicted category subumes the gold category
            print("subsumes")
            subsumed += 1

    print(f"in GS {doc_in_gold_standard} {correct} {subsumed}")


if __name__ == '__main__':

    args, input_folder_corpus, out_folder, docs_to_inspect, lemma_to_domain_dbs = args()

    output_folder_wiki = out_folder + "/wiki/"
    verbose = False

    if (not os.path.isdir(output_folder_wiki)):
        os.makedirs(output_folder_wiki)

    corpus_tfidf, dictionary, id_to_dictionary_token, id_to_domain, domain_to_id, doc_ids, id2doc, uri_to_doc_id, doc_id_to_uri, words_to_exclude, lemma_dbs, uri_to_gold_class, hierarchy, da = load_resources(
        input_folder_corpus, args)

    domain_vectors = np.zeros((len(corpus_tfidf), len(id_to_domain)))

    for doc_id in docs_to_inspect:
        inspect_document(domain_vectors, doc_id,  corpus_tfidf, id_to_domain, da)

    if not args.limit:
        limit = len(corpus_tfidf)
    else:
        limit = args.limit

    if (not len(docs_to_inspect)):
        annotate_documents(corpus_tfidf, limit, id_to_domain, doc_id_to_uri, doc_ids, domain_vectors, uri_to_gold_class,
                           hierarchy, da)

    # pickle.dump(domain_vectors, open(out_folder+"/domain_model.p", "wb"))
    # pickle.dump(id_to_domain, open(out_folder + "/id_to_domain.p", "wb"))
