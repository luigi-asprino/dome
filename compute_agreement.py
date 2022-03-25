#!/usr/bin/env python
import krippendorff
import numpy as np
from utils.Utils import load_map_from_file
import pickle


def annotation_set_to_vector(domain2id, annotation_set, hierarchy={}):
    if len(annotation_set) == 0:
        return np.full((len(domain2id),), np.nan)
    result = np.zeros(len(domain2id))
    for annotation in annotation_set:
        #print(annotation)
        #print(domain2id[annotation])
        result[domain2id[annotation]] = 1
        if len(hierarchy) > 0:
            if domain2id[annotation] in hierarchy:
                for super_kd in hierarchy[domain2id[annotation]]:
                    result[super_kd] = 1
    return result


def code_coder(vocs, dict, domain2id, hierarchy={}):
    res = []
    for voc in vocs:
        if voc in dict:
            vec = annotation_set_to_vector(domain2id, dict[voc], hierarchy)
        else:
            vec = annotation_set_to_vector(domain2id, [])
        res.extend(vec)
    return res


def counter(annotations):
    counter = {}
    counter_combined = {}
    for dataset, annotation_set in annotations.items():
        annotation_lab = " ".join(annotation_set)

        if annotation_lab in counter_combined:
            counter_combined[annotation_lab] = counter_combined[annotation_lab] + 1
        else:
            counter_combined[annotation_lab] = 1

        for annotation in annotation_set:
            if annotation in counter:
                counter[annotation] = counter[annotation] + 1
            else:
                counter[annotation] = 1

    print("\n\nCounter singular\n\n")
    for domain, count in sorted(counter_combined.items(), key=lambda item: item[1], reverse=True):
        print(f"{domain}\t{count}")

    print("\n\nCounter singular\n\n")
    for domain, count in sorted(counter.items(), key=lambda item: item[1], reverse=True):
        print(f"{domain}\t{count}")



def compute_krippendorff(vocs, coders, domain2id, hierarchy, outfolder, verbose, threshold_reliability = 0.66):
    vocs_reliable = []
    vocs_annotated = set()
    cross_domain_by_all = 0

    for voc in vocs:
        vector_exists = False
        has_annotations = False
        for c in coders:
            if voc in c:
                vector_exists = True
                if len(c[voc]) > 0:
                    has_annotations = True
                    vocs_annotated.add(voc)

        if not vector_exists or not has_annotations:
            cross_domain_by_all = cross_domain_by_all + 1
            if verbose:
                print(f"Krippendorff's alpha for {voc}: 1.0")
        else:
            reliability_data = [annotation_set_to_vector(domain2id, c[voc], hierarchy)
                                if voc in c else annotation_set_to_vector(domain2id, [], hierarchy) for c in coders]

            alpha = krippendorff.alpha(reliability_data=reliability_data)
            if verbose:
                print(f"Krippendorff's alpha for {voc}: {alpha}")
                for c in coders:
                    if voc in c:
                        print(f"\t{c[voc]}")

            if alpha > threshold_reliability:
                vocs_reliable.append(voc)
                if verbose:
                    print("RELIABLE")

    reliability_data = np.array([code_coder(vocs, c, domain2id, hierarchy) for c in coders])
    reliability_data_reliable_only = np.array([code_coder(vocs_reliable, c, domain2id, hierarchy) for c in coders])

    empty_intersection = 0
    intersection_annotations = {}
    maximal_annotation_set = {}
    for voc in vocs_reliable:

        reliability_data_l = [annotation_set_to_vector(domain2id, c[voc], hierarchy)
                if voc in c else annotation_set_to_vector(domain2id, [], hierarchy) for c in coders]

        alpha = krippendorff.alpha(reliability_data=reliability_data_l)

        if verbose:
            print(f"Krippendorff's alpha for {voc}: {alpha}")

        annotations = []
        max_set = set()
        for c in coders:
            if voc in c:
                if verbose:
                    print(f"\t{c[voc]}")
                if len(c[voc]) > 0:
                    annotations.append(set(c[voc]))
                    max_set.update(c[voc])

        intersection_set = set.intersection(*annotations)
        if len(intersection_set) == 0:
            empty_intersection += 1
        else:
            intersection_annotations[voc] = [ann for ann in intersection_set]
        maximal_annotation_set[voc] = [ann for ann in max_set]
        if verbose:
            print(f"Maximal Set of annotations {max_set}")
            print(f"Intersection set {intersection_set}")


    pickle.dump(intersection_annotations, open(outfolder+"intersection_annotations.p", "wb"))
    pickle.dump(maximal_annotation_set, open(outfolder+"maximal_annotation_set.p", "wb"))



    #print("\nIntersection set")
    #counter(intersection_annotations)

    print("\nMaximal set")
    counter(maximal_annotation_set)

    #for dataset in vocs:
    #    print(dataset)

    print(f"\n\nNumber of datasets identified as cross domain by all the annotators {cross_domain_by_all}")
    print(f"Number of reliable datasets {len(vocs_reliable)}/{len(vocs)}")
    # print(f"Number of annotated datasets  {len(vocs_annotated)}/{len(vocs)}")
    print(f"Number of datates with annotations and reliable "
          f"{len(set(vocs_reliable).intersection(set(vocs_annotated)))}/{len(vocs)}")
    # print(f"Reliable vocabularies with empty intersection among {empty_intersection}")
    print(f"Krippendorff's alpha for all vocabulaires: {krippendorff.alpha(reliability_data=reliability_data)}")
    print(f"Krippendorff's alpha for reliable vocabulaires: "
          f"{krippendorff.alpha(reliability_data=reliability_data_reliable_only)} ")


def main():

    c1 = load_map_from_file(
        "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/LOV/TSVs/LOV-all - LOV Ontologies.tsv",
        [0, 3, 4, 5], 0, [3, 4, 5], skiprows=[0])

    c2 = load_map_from_file(
        "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/LOV/TSVs/LOV-2to251-1 - LOV Ontologies.tsv",
        [0, 3, 4, 5], 0,
        [3, 4, 5], skiprows=[0])

    c3 = load_map_from_file(
        "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/LOV/TSVs/LOV-2to251-2 - LOV Ontologies.tsv",
        [0, 3, 4, 5], 0,
        [3, 4, 5], skiprows=[0])

    c4 = load_map_from_file(
        "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/LOV/TSVs/LOV-252to501-1 - LOV Ontologies.tsv",
        [0, 3, 4, 5], 0,
        [3, 4, 5], skiprows=[0])

    c5 = load_map_from_file(
        "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/LOV/TSVs/LOV-252to501-2 - LOV Ontologies.tsv",
        [0, 3, 4, 5], 0,
        [3, 4, 5], skiprows=[0])

    c6 = load_map_from_file(
        "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/LOV/TSVs/LOV-502to752-1 - LOV Ontologies.tsv",
        [0, 3, 4, 5], 0,
        [3, 4, 5], skiprows=[0])

    c7 = load_map_from_file(
        "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/LOV/TSVs/LOV-502to752-2 - LOV Ontologies.tsv",
        [0, 3, 4, 5], 0,
        [3, 4, 5], skiprows=[0])

    lod_c01 = load_map_from_file(
        "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/Laundromat/Laundromat-0-1 - Foglio2.tsv",
        [2, 6, 7, 8], 2,
        [6, 7, 8], skiprows=[0])

    lod_c02 = load_map_from_file(
        "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/Laundromat/Laundromat-0-2 - Foglio2.tsv",
        [2, 6, 7, 8], 2,
        [6, 7, 8], skiprows=[0])

    lod_c03 = load_map_from_file(
        "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/Laundromat/Laundromat-0-3 - Foglio2.tsv",
        [2, 6, 7, 8], 2,
        [6, 7, 8], skiprows=[0])

    lod_c11 = load_map_from_file(
        "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/Laundromat/Laundromat-1-1 - Foglio2.tsv",
        [2, 6, 7, 8], 2,
        [6, 7, 8], skiprows=[0])

    lod_c12 = load_map_from_file(
        "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/Laundromat/Laundromat-1-2 - Foglio2.tsv",
        [2, 6, 7, 8], 2,
        [6, 7, 8], skiprows=[0])

    lod_c13 = load_map_from_file(
        "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/Laundromat/Laundromat-1-3 - Foglio2.tsv",
        [2, 6, 7, 8], 2,
        [6, 7, 8], skiprows=[0])

    lod_c21 = load_map_from_file(
        "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/Laundromat/Laundromat-2-1 - Foglio2.tsv",
        [2, 6, 7, 8], 2,
        [6, 7, 8], skiprows=[0])

    lod_c22 = load_map_from_file(
        "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/Laundromat/Laundromat-2-2 - Foglio2.tsv",
        [2, 6, 7, 8], 2,
        [6, 7, 8], skiprows=[0])

    lod_c23 = load_map_from_file(
        "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/Laundromat/Laundromat-2-3 - Foglio2.tsv",
        [2, 6, 7, 8], 2,
        [6, 7, 8], skiprows=[0])

    datasets_ids = set.union(set([k[0] for k in lod_c01.items()]),
                             set([k[0] for k in lod_c11.items()]),
                             set([k[0] for k in lod_c21.items()]))

    vocs = [c[0] for c in c1.items() if c[0] != "URI"]

    id2domain = load_map_from_file("/Users/lgu/Google Drive/Lavoro/Progetti/EKR/domain2id.tsv")
    domain2id = {id2d[1]: id2d[0] for id2d in id2domain.items()}

    hierarchy = {}
    hierarchy_file = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/KD_hierarchy.tsv"
    for (k, v) in load_map_from_file(hierarchy_file).items():
        hierarchy[int(k)] = [int(kd.strip()) for kd in v.split(",")]

    coders = [c1, c2, c3, c4, c5, c6, c7]
    coders_lod = [lod_c11, lod_c12, lod_c13, lod_c21, lod_c22, lod_c23, lod_c01, lod_c02, lod_c03]

    compute_krippendorff(vocs, coders, domain2id, hierarchy,
                         "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/LOV/", False)
    #compute_krippendorff(datasets_ids, coders_lod, domain2id, hierarchy,
    #                     "/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/Laundromat/", False, 0.66)

    # vocs_reliable=[]
    # vocs_annotated=[]
    #
    # for voc in vocs:
    #     vector_exists = False
    #     has_annotations = False
    #     for c in coders:
    #         if voc in c:
    #             vector_exists = True
    #             if len(c[voc]) > 0:
    #                 has_annotations = True
    #                 vocs_annotated.append(voc)
    #
    #
    #     if not vector_exists or not has_annotations:
    #         print(f"Krippendorff's alpha for {voc}: 1.0")
    #     else:
    #         reliability_data = [annotation_set_to_vector(domain2id, c[voc], hierarchy)
    #                             if voc in c else annotation_set_to_vector(domain2id, [], hierarchy) for c in coders]
    #
    #         alpha = krippendorff.alpha(reliability_data=reliability_data)
    #         print(f"Krippendorff's alpha for {voc}: {alpha}")
    #         for c in coders:
    #             if voc in c:
    #                 print(f"\t{c[voc]}")
    #
    #         if alpha > 0.66:
    #             vocs_reliable.append(voc)
    #             print("RELIABLE")
    #
    # c22 = dict(c2, **c4, **c6)
    # print(len(c22))
    #
    # c33 = dict(c3, **c5, **c7)
    # print(len(c33))
    #
    #
    # reliability_data = np.array([code_coder(vocs, c1, domain2id, hierarchy), code_coder(vocs, c2, domain2id, hierarchy),
    #                     code_coder(vocs, c3, domain2id, hierarchy), code_coder(vocs, c4, domain2id, hierarchy),
    #                     code_coder(vocs, c5, domain2id, hierarchy), code_coder(vocs, c6, domain2id, hierarchy),
    #                     code_coder(vocs, c7, domain2id, hierarchy)])
    # reliability_data_2 = np.array([code_coder(vocs, c1, domain2id, hierarchy), code_coder(vocs, c22, domain2id, hierarchy),
    #                              code_coder(vocs, c33, domain2id, hierarchy)])
    # reliability_data_3 = np.array([code_coder(vocs_reliable, c1, domain2id, hierarchy), code_coder(vocs_reliable, c22, domain2id, hierarchy),
    #                                code_coder(vocs_reliable, c33, domain2id, hierarchy)])
    #
    # empty_intersection = 0
    # intersection_annotations = {}
    # maximal_annotation_set = {}
    # for voc in vocs_reliable:
    #     reliability_data_l = [annotation_set_to_vector(domain2id, c[voc], hierarchy)
    #             if voc in c else annotation_set_to_vector(domain2id, [], hierarchy) for c in coders]
    #     alpha = krippendorff.alpha(reliability_data=reliability_data_l)
    #     print(f"Krippendorff's alpha for {voc}: {alpha}")
    #     annotations = []
    #     max_set = set()
    #     for c in coders:
    #         if voc in c:
    #             print(f"\t{c[voc]}")
    #             if len(c[voc]) > 0:
    #                 annotations.append(set(c[voc]))
    #                 max_set.update(c[voc])
    #     intersection_set = set.intersection(*annotations)
    #     if len(intersection_set) == 0:
    #         empty_intersection += 1
    #     else:
    #         intersection_annotations[voc] = [ann for ann in intersection_set]
    #     maximal_annotation_set[voc] = [ann for ann in max_set]
    #     print(f"Maximal Set of annotations {max_set}")
    #     print(f"Intersection set {intersection_set}")
    #
    # import pickle
    # pickle.dump(intersection_annotations, open("/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/LOV/TSVs/intersection_annotations.p", "wb"))
    # pickle.dump(maximal_annotation_set, open("/Users/lgu/Google Drive/Lavoro/Progetti/EKR/Annotation/LOV/TSVs/maximal_annotation_set.p", "wb"))
    #
    # print()
    # print(f"Number of reliable vocabularies {len(vocs_reliable)}")
    # print(f"Number of vocabularies with annotations and reliable {len(set(vocs_reliable).intersection(set(vocs_annotated)))}")
    # print(f"Reliable vocabularies with empty intersection among {empty_intersection}")
    #
    # print(f"Krippendorff's alpha for all vocabulaires: {krippendorff.alpha(reliability_data=reliability_data_2)} "
    #       f"{krippendorff.alpha(reliability_data=reliability_data)}")
    # print(f"Krippendorff's alpha for reliable vocabulaires: {krippendorff.alpha(reliability_data=reliability_data_3)} ")

if __name__ == '__main__':
    main()