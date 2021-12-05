import pandas as pd
import pickle
import os
import numpy as np

import logging as logger

logger.basicConfig(level=logger.INFO)


def load_list_from_file(filename, token_number, extractid=False):
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
    if (os.path.exists(filename + ".p")):
        return pickle.load(open(filename + ".p", "rb"))
    df = pd.read_csv(filename, sep='\t', header=None, usecols=[0, 1])
    map = {row[0]: row[1] for index, row in df.iterrows()}
    pickle.dump(map, open(filename + ".p", "wb"))
    return map


def order_and_print_domain_vector(vector, id_to_domain):
    domains_extracted = {id: value for id, value in enumerate(vector)}
    domains_extracted_ordered = {k: v for k, v in
                                 sorted(domains_extracted.items(), key=lambda item: item[1], reverse=True)}
    for id in domains_extracted_ordered:
        print(f"{id_to_domain[id]} {domains_extracted_ordered[id]}")


def get_top_domain(vector, id_to_domain):
    return id_to_domain[np.argmax(vector)]


def nomralize_category(dbpedia_cat):
    return dbpedia_cat.replace('http://dbpedia.org/resource/Category:','').replace('Computer_science', 'Computer Science').replace('Political_science','Political Science')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--input_folder_corpus', dest='input_folder_corpus', help=' ... ')
    parser.add_argument('--domain_model', dest='domain_model', help=' ... ')
    parser.add_argument('--id_to_domain', dest='id_to_domain', help=' ... ')
    parser.add_argument('--token_number', dest='token_number',
                        help='The number of token of the path containing the id of the doc.', type=int)
    parser.add_argument('--uri_to_doc_id', dest='uri_to_doc_id',
                        help='The filepath to a TSV file containing the mapping of the URIs of the datasets to the ids of the document.')
    parser.add_argument('--gold_standard', dest='gold_standard',
                        help='A TSV containing gold standard annotations for each dataset URI.')
    parser.add_argument('--datasets_descriptions', dest='datasets_descriptions',
                        help='A TSV containing a natural language description of the datasets.')


    args = parser.parse_args()

    doc_ids = load_list_from_file(args.input_folder_corpus + "/doc_ids", args.token_number, extractid=True)
    id2doc = {k: v for v, k in enumerate(doc_ids)}
    domain_model = pickle.load(open(args.domain_model, "rb"))
    id_to_domain = pickle.load(open(args.id_to_domain, "rb"))

    if(args.uri_to_doc_id):
        uri_to_doc_id = load_map_from_file(args.uri_to_doc_id)
        doc_id_to_uri = {k : v for v,k in uri_to_doc_id.items()}

    if (args.gold_standard):
        uri_to_gold_class = load_map_from_file(args.gold_standard)

    if (args.datasets_descriptions):
        datasets_descriptions = load_map_from_file(args.datasets_descriptions)

    counter_datasets = {id_to_domain[id]: 0 for id in id_to_domain}

    for vd in domain_model:
        counter_datasets[get_top_domain(vd, id_to_domain)] += 1

    count_equals  = 0
    count_total = 0
    count_total_6 = 0
    count_equals_6 = 0

    if (args.uri_to_doc_id):
        for i, e in enumerate(domain_model):
            uri = doc_id_to_uri[doc_ids[i]]
            if args.gold_standard and uri in uri_to_gold_class:
                category_predicted = nomralize_category(get_top_domain(e, id_to_domain))
                gold_category = uri_to_gold_class[doc_id_to_uri[doc_ids[i]]]
                top_score = np.max(e)
                count_total = count_total + 1
                if top_score > 0.6:
                    count_total_6 = count_total_6 + 1

                if (category_predicted == gold_category):
                    count_equals = count_equals + 1
                    if top_score > 0.6:
                        count_equals_6 = count_equals_6 + 1
                if(args.datasets_descriptions):
                    print(
                        f"{count_total-1}) {uri}\t{category_predicted}\t{top_score}\t{gold_category}\t{datasets_descriptions[doc_id_to_uri[doc_ids[i]]]}")
                else:
                    print(f"{uri}\t{category_predicted}\t{gold_category}")
            #else: Do not print if it is not in the gold_standard
            #    print( f"{doc_id_to_uri[doc_ids[i]]}\t{nomralize_category(get_top_domain(e, id_to_domain))}")




    print("\n\nCounter datasets\n\n")

    for domain in counter_datasets:
        print(f"{domain} {counter_datasets[domain]}")

    print(f"Total {count_total}")
    print(f"Equals {count_equals}")
    print(f"Equals {count_equals/count_total}")

    print(f"Total > .6 {count_total_6}")
    print(f"Equals > .6 {count_equals_6}")
    print(f"Equals > .6 {count_equals_6 / count_total_6}")