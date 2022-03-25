import pickle
import rocksdb
from rocksdb.merge_operators import StringAppendOperator

import logging as logger

from utils.Utils import load_list_from_file

logger.basicConfig(level=logger.INFO)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--input_folder_corpus', dest='input_folder_corpus',
                        help=' ... ')
    parser.add_argument('--annotations', dest='annotations',
                        help=' ... ')
    parser.add_argument('--entity2dataset_db', dest='entity2dataset_db',
                        help=' ... ')
    parser.add_argument('--empty_entities_path', dest='empty_entities_path',
                        help=' ... ')
    parser.add_argument('--entities_annotated', dest='entities_annotated',
                        help=' ... ')
    parser.add_argument('--limit', dest='limit', help=' ... ')

    args = parser.parse_args()

    empty_entities = load_list_from_file(args.empty_entities_path)
    doc_ids = load_list_from_file(args.input_folder_corpus + "/doc_ids", extractid=True)
    id2doc = {k: v for v, k in enumerate(doc_ids)}

    opts = rocksdb.Options()
    opts.create_if_missing = True
    opts.merge_operator = StringAppendOperator()
    db = rocksdb.DB(args.entity2dataset_db, opts)

    laundromat_annotations = pickle.load(open(args.annotations, "rb"))

    if not args.limit:
        limit = len(empty_entities)
    else:
        limit = int(args.limit)
    print(f"limit: {limit}")

    cnt = 0

    entity_to_domains = {entity: set() for entity in empty_entities}

    not_found = 0

    for id_entity in range(0, limit):

        if cnt % 10000 == 0:
            logger.info(f"{cnt}/{limit}")

        cnt = cnt + 1

        datasets = db.get(empty_entities[id_entity].encode())

        if datasets is None:
            not_found += 1
            continue

        datasets = datasets.decode().split(",")

        for idx, id_doc in enumerate(datasets):
            if id_doc in laundromat_annotations:
                entity_to_domains[empty_entities[id_entity]].update(laundromat_annotations[id_doc])

    print(f"Entities not found {not_found}")
    pickle.dump(entity_to_domains, open(args.entities_annotated, "wb"))
