from sklearn import utils
from sklearn.base import BaseEstimator, TransformerMixin
from utils.mlsmote import MLSMOTE_iterative, MLSMOTE, get_irlb
from utils.Utils import transform_to_powerlabel, transform_to_binary
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
import numpy as np
import scipy.sparse as sp
import random

def resample_powerlabel_smoteen(X, y):
    y_powerlabel, powerlabel_to_bin = transform_to_powerlabel(y.values)
    ros = SMOTEENN(random_state=42)
    X_res, y_res_powerlabel = ros.fit_resample(X, pd.DataFrame(y_powerlabel))
    y_res = pd.DataFrame(transform_to_binary(y_res_powerlabel.values, powerlabel_to_bin))
    return X_res, y_res


def resample_powerlabel_ros(X, y):
    y_powerlabel, powerlabel_to_bin = transform_to_powerlabel(y.values)
    ros = RandomOverSampler(random_state=42)
    X_res, y_res_powerlabel = ros.fit_resample(X, pd.DataFrame(y_powerlabel))
    y_res = pd.DataFrame(transform_to_binary(y_res_powerlabel.values, powerlabel_to_bin))
    return X_res, y_res


def resample_powerlabel(X, y):
    y_powerlabel, powerlabel_to_bin = transform_to_powerlabel(y.values)
    X_res, y_powerlabel_res = utils.resample(X,pd.DataFrame(y_powerlabel))
    y_res = pd.DataFrame(transform_to_binary(y_powerlabel_res.values, powerlabel_to_bin))
    return X_res, y_res

class Strategies:
    MLSMOTE = "MLSMOTE"
    MLSMOTE_IT = "MLSMOTE_IT"
    DEFAULT = "DEFAULT"
    NO_RESAMPLE = "NO_RESAMPLE"
    OVERSAMPLING = "OVERSAMPLING"
    MLSMOTE_IT_UNDERSAMPLING = "MLSMOTE_IT_UNDERSAMPLING"
    MLSMOTE_CP_IT_UNDERSAMPLING = "MLSMOTE_CP_IT_UNDERSAMPLING"
    MLSMOTE_CP_IT = "MLSMOTE_CP_IT"
    UNDERSAMPLING_OVERSAMPLING = "UNDERSAMPLING_OVERSAMPLING"
    MLSMOTE_IT_THRESHOLD_10 = "MLSMOTE_IT_THRESHOLD_10"

def resample(X, y, strategy=None, n_sample=None):
    irlb, irlb_mean_last = get_irlb(y)
    print(f"IRMean before resampling {irlb_mean_last} number of examples {len(X)} number of target {len(y)}")
    print(f"Applying Strategy {strategy}")

    if strategy == Strategies.MLSMOTE_IT_UNDERSAMPLING or strategy == Strategies.MLSMOTE_CP_IT_UNDERSAMPLING or \
            strategy == Strategies.UNDERSAMPLING_OVERSAMPLING:
        X, y = undersampling_ml(X, y)

    if strategy is None:
        X_res, y_res = utils.resample(X, y)
    elif strategy == "powerlabel":
        X_res, y_res = resample_powerlabel(X, y)
    elif strategy == "powerlabel_ros":
        X_res, y_res = resample_powerlabel_ros(X, y)
    elif strategy == "powerlabel_smoteen":
        X_res, y_res = resample_powerlabel_smoteen(X, y)
    elif strategy == Strategies.MLSMOTE_IT or strategy == Strategies.MLSMOTE_IT_UNDERSAMPLING:
        X_res, y_res = MLSMOTE_iterative(X, y)
    elif strategy == Strategies.MLSMOTE_IT_THRESHOLD_10:
        X_res, y_res = MLSMOTE_iterative(X, y, threshold=10)
    elif strategy == Strategies.MLSMOTE_CP_IT or strategy == Strategies.MLSMOTE_CP_IT_UNDERSAMPLING:
        X_res, y_res = MLSMOTE_iterative(X, y, cp=True)
    elif strategy == Strategies.MLSMOTE:
        X_res, y_res = MLSMOTE(X, y, n_sample)
    elif strategy == Strategies.OVERSAMPLING or strategy == Strategies.UNDERSAMPLING_OVERSAMPLING:
        X_res, y_res = oversampling(X, y)
    elif strategy == Strategies.DEFAULT:
        X_res, y_res = utils.resample(X, y)
    elif strategy == Strategies.NO_RESAMPLE:
        X_res = X
        y_res = y

    irlb, irlb_mean_last = get_irlb(y_res)
    print(f"IRMean after resampling {irlb_mean_last} number of examples {len(X_res)} number of target {len(y_res)}")
    return X_res, y_res


def print_counters(y, multilabel_binarizer, clab=None):
    counter, lab_to_ann_set, counter_singular = get_class_distribution(multilabel_binarizer.inverse_transform(y))
    print("\n\nCounter\n\n")
    for lab, n in sorted(counter.items(), key=lambda item: item[1], reverse=True):
        if clab is not None and lab in clab:
            print(f"{lab}\t{n}\tformer: {clab[lab]}")
        else:
            print(f"{lab}\t{n}")

    print("\n\nCounter Singular\n\n")
    for lab, n in sorted(counter_singular.items(), key=lambda item: item[1], reverse=True):
        print(f"{lab}\t{n}")
    return counter



def get_class_distribution(y):
    counter = {}
    counter_singular = {}
    label_to_annotation_set = {}
    for i in y:
        k = " ".join([str(ii) for ii in i])
        label_to_annotation_set[k] = i
        if k in counter:
            counter[k] = counter[k] + 1
        else:
            counter[k] = 1
        for lab in i:
            if lab in counter_singular:
                counter_singular[lab] = counter_singular[lab] + 1
            else:
                counter_singular[lab] = 1
    return counter, label_to_annotation_set, counter_singular


def write_class_distribution_on_file(y, filepath):
    counter, label_to_annotation_set, counter_singular = get_class_distribution(y[0])
    fcounter = open(filepath, 'w')
    counter_ordered = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    for k, v in counter_ordered:
        fcounter.write(f"{k}\t{v}\n")

    fcounter.write(f"\n\nSINGULAR\n\n")
    counter_singular_ordered = sorted(counter_singular.items(), key=lambda item: item[1], reverse=True)
    for k, v in counter_singular_ordered:
        fcounter.write(f"{k}\t{v}\n")
    fcounter.flush()
    fcounter.close()
    return counter, counter_ordered, label_to_annotation_set


def specialize_annotations(y, hierarchy, domain_to_id):
    for idx, i in enumerate(y):
        to_exclude_from_i = set([])
        for ii in i:
            for jj in i:
                if domain_to_id[ii] in hierarchy and domain_to_id[jj] in hierarchy[domain_to_id[ii]]: # jj is superclass of ii
                    to_exclude_from_i.add(jj)
        for r in to_exclude_from_i:
            i.remove(r)


def generalise_annotations(y, hierarchy, domain_to_id):
    id_to_domain = {v: k for k, v in domain_to_id.items()}
    for idx, i in enumerate(y):
        annotation_set_new = set(i)
        to_add = set(i)
        for ii in i:
            if domain_to_id[ii] in hierarchy:
                for super_domain in hierarchy[domain_to_id[ii]]:
                    if id_to_domain[super_domain] not in annotation_set_new:
                        annotation_set_new.add(id_to_domain[super_domain])
                        to_add.add(id_to_domain[super_domain])
        for a in to_add:
            i.append(a)


def restrict_to_top_level_domains(y, hierarchy, domain_to_id, top_level_domains):
    generalise_annotations(y, hierarchy, domain_to_id)

    for idx, i in enumerate(y):
        to_remove = [ii for ii in i if ii not in top_level_domains]
        for a in to_remove:
            i.remove(a)
        y[idx] = list(set(i))


def get_indexes_of_items_with_labels(y, annotation_set):
    return [idx for idx, ann in enumerate(y) if ann == annotation_set]


def get_indexes_of_items_with_labels_ml(y, annotation_set):
    return [idx for idx, ann in enumerate(y) if (ann == annotation_set).all()]


def oversampling(X, y):
    counter, label_to_annotation_set, counter_singular = get_class_distribution(y.values)
    counter_ordered = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    sizes = np.array([counter_ordered[i][1] for i in range(0, len(counter_ordered))])
    mean = sizes.mean()
    print(f"{sizes.mean()} {sizes.std()}")
    #upsize = np.ceil(mean)
    upsize = sizes.max()

    annotation_sets_to_upsize = [label_to_annotation_set[counter_ordered[k][0]]
                                 for k in range(0, len(counter_ordered)) if counter_ordered[k][1] < mean]
    print(f"#{len(annotation_sets_to_upsize)} annotation sets to upsize  up to {upsize}")

    X_to_add = [X]
    y_to_add = [y]

    for annotation_set_to_upsize in annotation_sets_to_upsize:
        examples_having_the_annotation_set = get_indexes_of_items_with_labels_ml(y.values, annotation_set_to_upsize)
        #n_replicas = np.ceil(upsize / len(examples_having_the_annotation_set))
        n_of_examples_to_add = upsize - len(examples_having_the_annotation_set)
        indexes_to_add = []

        while n_of_examples_to_add > 0:
            # X_to_add.append(X.loc[examples_having_the_annotation_set])
            # y_to_add.append(y.loc[examples_having_the_annotation_set])
            indexes_to_add.append(random.choice(examples_having_the_annotation_set))
            n_of_examples_to_add = n_of_examples_to_add - 1

        #print(f"{len(indexes_to_add)} examples taken from {annotation_set_to_upsize} "
        #      f"of length {len(examples_having_the_annotation_set)}")
        X_to_add.append(X.loc[indexes_to_add])
        y_to_add.append(y.loc[indexes_to_add])

    X = pd.concat(X_to_add)
    y = pd.concat(y_to_add)

    return X, y



def undersampling(X, y):
    counter, label_to_annotation_set, counter_singular = get_class_distribution(y[0])
    counter_ordered = sorted(counter.items(), key=lambda item: item[1], reverse=True)

    sizes = np.array([counter_ordered[i][1] for i in range(0, len(counter_ordered))])
    mean = sizes.mean()
    downsize = int(np.ceil(mean + sizes.std()))
    print(f"{sizes.mean()} {np.median(sizes)} {sizes.std()}")

    annotation_sets_to_downsize = [label_to_annotation_set[counter_ordered[k][0]]
                                   for k in range(0, len(counter_ordered))
                                   if counter_ordered[k][1] > downsize]

    print(f"Annotations to downsize {annotation_sets_to_downsize} down to {downsize}")
    all_indexes_to_remove = []

    for annotation_set_to_downsize in annotation_sets_to_downsize:
        majority_class_indexes = get_indexes_of_items_with_labels(y[0], annotation_set_to_downsize)
        examples_to_keep = random.sample(majority_class_indexes, downsize)
        indexes_to_remove = [idx for idx in majority_class_indexes if idx not in examples_to_keep]
        all_indexes_to_remove.extend(indexes_to_remove)
        print(f"Removing {len(indexes_to_remove)} from the class {annotation_set_to_downsize}")

    X = X.drop(all_indexes_to_remove).reset_index(drop=True)
    y = y.drop(all_indexes_to_remove).reset_index(drop=True)

    print(f"Removing total number of indexes {len(all_indexes_to_remove)} len(X): {len(X)} len(y): {len(y)}")

    return X, y


def undersampling_ml(X, y):
    counter, label_to_annotation_set, counter_singular = get_class_distribution(y.values)
    counter_ordered = sorted(counter.items(), key=lambda item: item[1], reverse=True)

    sizes = np.array([counter_ordered[i][1] for i in range(0, len(counter_ordered))])
    mean = sizes.mean()
    downsize = int(np.ceil(mean + sizes.std()))
    print(f"{sizes.mean()} {np.median(sizes)} {sizes.std()}")

    annotation_sets_to_downsize = [label_to_annotation_set[counter_ordered[k][0]]
                                   for k in range(0, len(counter_ordered))
                                   if counter_ordered[k][1] > downsize]

    # print(f"Annotations to downsize {annotation_sets_to_downsize} down to {downsize}")
    all_indexes_to_remove = []

    for annotation_set_to_downsize in annotation_sets_to_downsize:
        majority_class_indexes = get_indexes_of_items_with_labels_ml(y.values, annotation_set_to_downsize)
        examples_to_keep = random.sample(majority_class_indexes, downsize)
        indexes_to_remove = [idx for idx in majority_class_indexes if idx not in examples_to_keep]
        all_indexes_to_remove.extend(indexes_to_remove)
        # print(f"Removing {len(indexes_to_remove)} from the class {label_to_annotation_set}")

    X = X.drop(all_indexes_to_remove).reset_index(drop=True)
    y = y.drop(all_indexes_to_remove).reset_index(drop=True)

    print(f"Removing total number of indexes {len(all_indexes_to_remove)} len(X): {len(X)} len(y): {len(y)}")

    return X, y


class DomainTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, da, cv):
        self.da = da
        self.cv = cv

    def fit(self, X, y=None):
        self.words = self.cv.get_feature_names()
        print(f"Number of words {len(self.words)}")
        return self

    def transform(self, X, y=None):
        d = np.zeros((X.shape[0], len(self.da.id_to_domain)))
        for row in range(0, X.shape[0]):
            doc_words_all = {self.words[col]: X[row, col] for col in X[row, :].nonzero()[1]}
            d[row] = self.da.get_domain_vector(doc_words_all)[0]
        return sp.hstack([X, d])
