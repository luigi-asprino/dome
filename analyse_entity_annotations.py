import pickle
from rocksdb.merge_operators import StringAppendOperator
import rocksdb

from utils.Utils import load_map_from_file


def increase_counter_dict(d, key):
    if key in d:
        d[key] = d[key] + 1
    else:
        d[key] = 1

experiment_dir = "/Users/lgu/Desktop/NOTime/EKR/experiments/lov_bench_laund_max_hp_bin/"
entities_annotated = pickle.load(open(experiment_dir + "entities_annotated.p", "rb"))
entity_to_dataset_db = "/Users/lgu/Desktop/NOTime/EKR/entity2dataset11"

opts = rocksdb.Options()
opts.create_if_missing = True
opts.merge_operator = StringAppendOperator()
db = rocksdb.DB(entity_to_dataset_db, opts)

counter = {}
counter_domain = {}
datasets_without_annotations_containing_empty_entities = set()
datasets_containing_empty_entities = set()
without_annotations = []
skolemized_entities = []
number_of_empty_entities_per_dataset = {}

f = open(experiment_dir + "results_of_annotated_entities.txt", "w")

for entity, annotation_set in entities_annotated.items():
    datasets = db.get(entity.encode())
    if datasets is not None:
        datasets = datasets.decode().split(",")
        for dataset in datasets:
            increase_counter_dict(number_of_empty_entities_per_dataset, dataset)
    else:
        datasets = []
    datasets_containing_empty_entities.update(datasets)
    if len(annotation_set) == 0:
        without_annotations.append(entity)
        datasets_without_annotations_containing_empty_entities.update(datasets)
        if entity.startswith("http://lodlaundromat.org/.well-known"):
            skolemized_entities.append(entity)

    for annotation in annotation_set:
        increase_counter_dict(counter_domain, annotation)

    ann_set = " ".join(annotation_set)
    increase_counter_dict(counter, ann_set)


for ann_set, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
    print(f"{ann_set} {count}")
    f.write(f"{ann_set} {count}\n")

print("\n\n\n")
f.write(f"\n\n\n")

for ann_set, count in sorted(counter_domain.items(), key=lambda x: x[1], reverse=True):
    print(f"{ann_set} {count}")
    f.write(f"{ann_set} {count}\n")

print("\n\n\n")
f.write(f"\n\n\n")
print(f"{len(without_annotations)} empty entities coming from "
      f"{len(datasets_without_annotations_containing_empty_entities)} datasets without annotations")
f.write(f"{len(without_annotations)} empty entities coming from "
      f"{len(datasets_without_annotations_containing_empty_entities)} datasets without annotations")
print(f"Datasets containing empty annotations {len(datasets_containing_empty_entities)}")
f.write(f"Datasets containing empty annotations {len(datasets_containing_empty_entities)}")
print(f"Skolemized entities {len(skolemized_entities)}")
f.write(f"Skolemized entities {len(skolemized_entities)}")

source_to_doc_id = load_map_from_file("/Users/lgu/Desktop/NOTime/EKR/Corpus_lod_4/sources.txt")
doc_id_to_source = {v: k for k, v in source_to_doc_id.items()}

f.write(f"\n\n\n")
sources_without_annotations_containing_empty_entities = \
    [doc_id_to_source[dataset_id] for dataset_id in datasets_without_annotations_containing_empty_entities]
for dataset_id in sorted(sources_without_annotations_containing_empty_entities):
    f.write(f"{dataset_id} {source_to_doc_id[dataset_id]} "
            f"{number_of_empty_entities_per_dataset[source_to_doc_id[dataset_id]]}\n")