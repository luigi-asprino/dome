import pandas as pd
import pickle
import rocksdb
import os
import numpy as np
from rocksdb.merge_operators import StringAppendOperator

import logging as logger
logger.basicConfig( level=logger.INFO)

def load_list_from_file(filename, extractid=False):
    if (os.path.exists(filename + ".p")):
        return pickle.load(open(filename + ".p", "rb"))
    df = pd.read_csv(filename, sep='\t', header=None, usecols=[0])
    if extractid:
        result = [row[0].split("/")[4] for index, row in df.iterrows()]
    else:
        result = [row[0] for index, row in df.iterrows()]
    pickle.dump(result, open(filename + ".p", "wb"))
    return result

def order_and_print_domain_vector(vector, id_to_domain):
    domains_extracted = {id: value for id, value in enumerate(vector)}
    domains_extracted_ordered = {k: v for k, v in
                                 sorted(domains_extracted.items(), key=lambda item: item[1], reverse=True)}
    for id in domains_extracted_ordered:
        print(f"{id_to_domain[id]} {domains_extracted_ordered[id]}")

def get_top_domain(vector, id_to_domain):
    return id_to_domain[np.argmax(vector)]


def get_domain_vector(entity, domain_model, id2doc):
    dataset_string = db.get(entity.encode())
    if dataset_string is not None:
        datasets = dataset_string.decode().split(",")
        avg_vectors = np.average([domain_model[id2doc[d]] for d in datasets if d in id2doc], axis=0)
        return avg_vectors

    return np.zeros(domain_model[0].shape[0])

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--input_folder_corpus', dest='input_folder_corpus',
                        help=' ... ')
    parser.add_argument('--domain_model', dest='domain_model',
                        help=' ... ')
    parser.add_argument('--id_to_domain', dest='id_to_domain',
                        help=' ... ')
    parser.add_argument('--entity2dataset_db', dest='entity2dataset_db',
                        help=' ... ')
    parser.add_argument('--empty_entities_path', dest='empty_entities_path',
                        help=' ... ')
    parser.add_argument('--empty_entities_domain_vectors', dest='empty_entities_domain_vectors',
                        help=' ... ')
    parser.add_argument('--limit', dest='limit', help=' ... ')

    parser.add_argument('--sum', dest='sum', action='store_true')


    args = parser.parse_args()



    empty_entities = load_list_from_file(args.empty_entities_path )
    doc_ids = load_list_from_file(args.input_folder_corpus + "/doc_ids", extractid=True)
    #logger.debug(doc_ids)
    id2doc = {k : v for v,k in enumerate(doc_ids)}
    domain_model = pickle.load(open(args.domain_model, "rb"))
    id_to_domain = pickle.load(open(args.id_to_domain, "rb"))

    opts = rocksdb.Options()
    opts.create_if_missing = True
    opts.merge_operator = StringAppendOperator()
    db = rocksdb.DB(args.entity2dataset_db, opts)

    zero_vector_entities = 0
    not_found = 0
    zero_vector_datasets = []
    zero_vector_dataset_ids = []

    for id, v in enumerate(domain_model):
        if not np.any(v):
            zero_vector_datasets.append(doc_ids[id])
            zero_vector_dataset_ids.append(id)

    counter = {id_to_domain[id] : 0  for id in id_to_domain}

    counter_datasets = {id_to_domain[id]: 0 for id in id_to_domain}

    if not args.limit:
        limit = len(empty_entities)
    else:
        limit = int(args.limit)
    print(f"limit: {limit}")

    cnt = 0

    for id_entity in range(0,limit):

        if cnt%10000==0:
            logger.info(f"{cnt}/{limit}")

        cnt = cnt +1

        datasets = db.get(empty_entities[id_entity].encode())

        if datasets is  None:
            not_found +=1
            continue

        datasets = datasets.decode().split(",")

        d = np.zeros((len(datasets),len(id_to_domain)))

        for idx, id_doc in enumerate(datasets):
            if id_doc in id2doc:
                #logger.debug(f"{id_doc} {id2doc[id_doc]} {domain_model[id2doc[id_doc]]}")
                d[idx] = domain_model[id2doc[id_doc]]
            else:
                logger.debug(f"Couldn't find {id_doc}")

        if args.sum:
            avg = sum(d)
        else:
            avg = sum(d)/len(datasets)

        if not np.any(avg):
            zero_vector_entities += 1
            logger.debug(f"zero")
            logger.debug(f"{datasets}")
        else:
            counter[get_top_domain(avg,id_to_domain)]+=1

        logger.debug(f"{empty_entities[id_entity].encode()} {datasets}")


    print(f"entities with zero domain vector {zero_vector_entities}/{len(empty_entities)}")
    print(f"not found {not_found}")
    print(f"zero vector datasets {len(zero_vector_datasets)} {zero_vector_datasets[0]}")
    print(f"example of zero vector dataset {doc_ids[zero_vector_dataset_ids[0]]} {zero_vector_dataset_ids[0]}")
    #order_and_print_domain_vector(domain_vectors[zero_vector_dataset_ids[0]], id_to_domain)


    print("Counter entities")
    for domain in counter:
        print(f"{domain} {counter[domain]}")

    print("Counter datasets")

    for vd in domain_model:
        counter_datasets[get_top_domain(vd, id_to_domain)] += 1

    for domain in counter_datasets:
        print(f"{domain} {counter_datasets[domain]}")