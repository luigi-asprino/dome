import pickle
from sklearn.preprocessing import MultiLabelBinarizer

from utils.Utils import load_map_from_file
from utils.ml_utils import generalise_annotations


def counter(annotations):
    counter_singular = {}
    counter_combined = {}
    counter_singular_and_combined = {}
    for annotation_set in annotations:
        annotation_lab = " ".join(annotation_set)

        if annotation_lab in counter_combined:
            counter_combined[annotation_lab] = counter_combined[annotation_lab] + 1
        else:
            counter_combined[annotation_lab] = 1

        for annotation in annotation_set:
            if annotation in counter_singular:
                counter_singular[annotation] = counter_singular[annotation] + 1
            else:
                counter_singular[annotation] = 1

    print("\n\nCounter combined\n\n")
    for domain, count in sorted(counter_combined.items(), key=lambda item: item[1], reverse=True):
        print(f"{domain}\t{count}")

    print("\n\nCounter singular\n\n")
    for domain, count in sorted(counter_singular.items(), key=lambda item: item[1], reverse=True):
        print(f"{domain}\t{count}")


#y = pickle.load(open("/Users/lgu/Desktop/NOTime/EKR/experiments/lov_bench_laund_max_hp_bin/y_pre.p", "rb"))
#mlb = MultiLabelBinarizer()
#y = mlb.fit_transform(y[0])
#y = mlb.inverse_transform(y)
#counter(y)

hierarchy_file = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/KD_hierarchy.tsv"
id_to_domain = "/Users/lgu/workspace/ekr/dome/resources/20211126_input_unified/id2domain.tsv"
id_to_domain = load_map_from_file(id_to_domain)
domain_to_id = {k: v for v, k in id_to_domain.items()}


al = pickle.load(open("/Users/lgu/Desktop/NOTime/EKR/experiments/lov_bench_laund_max_hp_bin/annotated_laundromat.p", "rb"))

ann_sets = [[a for a in an_set] for d, an_set in al.items()]




# loading hierarchy
hierarchy = {}
for (k, v) in load_map_from_file(hierarchy_file).items():
    hierarchy[int(k)] = [int(kd.strip()) for kd in v.split(",")]

for id, domain in hierarchy.items():
    print(f"{id} {domain}")

generalise_annotations(ann_sets, hierarchy, domain_to_id)

ann_sets = [list(set(ann_set)) for ann_set in ann_sets]

counter(ann_sets)