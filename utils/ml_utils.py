from sklearn import utils
from sklearn.base import BaseEstimator, TransformerMixin
from utils.mlsmote import MLSMOTE_iterative, MLSMOTE, get_irlb
from utils.Utils import transform_to_powerlabel, transform_to_binary
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
import numpy as np
import scipy.sparse as sp

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
    MLSMOTE = "mlsmote"
    MLSMOTE_IT = "mlsmote_iterative"
    DEFAULT = "default"

def resample(X, y, stategy=None, n_sample=None):
    irlb, irlb_mean_last = get_irlb(y)
    print(f"IRMean before resampling {irlb_mean_last} number of examples {len(X)} number of target {len(y)}")
    print(f"strategy {stategy}")
    if stategy is None:
        X_res, y_res = utils.resample(X, y)
    elif stategy == "powerlabel":
        X_res, y_res = resample_powerlabel(X, y)
    elif stategy == "powerlabel_ros":
        X_res, y_res = resample_powerlabel_ros(X, y)
    elif stategy == "powerlabel_smoteen":
        X_res, y_res = resample_powerlabel_smoteen(X, y)
    elif stategy == Strategies.MLSMOTE_IT:
        X_res, y_res = MLSMOTE_iterative(X, y)
    elif stategy == Strategies.MLSMOTE:
        X_res, y_res = MLSMOTE(X, y, n_sample)
    elif stategy == Strategies.DEFAULT:
        X_res, y_res = utils.resample(X, y)
    irlb, irlb_mean_last = get_irlb(y_res)
    print(f"IRMean after resampling {irlb_mean_last} number of examples {len(X_res)} number of target {len(y_res)}")
    return X_res, y_res


def get_class_distribution(y):
    counter = {}
    for i in y:
        k = " ".join(i)
        if k in counter:
            counter[k] = counter[k] + 1
        else:
            counter[k] = 1
    return counter


def write_class_distribution_on_file(y, filepath):
    counter = get_class_distribution(y[0])
    fcounter = open(filepath, 'w')
    counter_ordered = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    for k, v in counter_ordered:
        fcounter.write(f"{k}\t{v}\n")
    fcounter.flush()
    fcounter.close()
    return counter, counter_ordered


def specialize_annotations(y, hierarchy, domain_to_id):
    for i in y:
        to_exclude_from_i = set([])
        for ii in i:
            for jj in i:
                if domain_to_id[ii] in hierarchy and domain_to_id[jj] in hierarchy[domain_to_id[ii]]: # jj is superclass of ii
                    to_exclude_from_i.add(jj)
        for r in to_exclude_from_i:
            i.remove(r)


def get_indexes_of_items_with_labels(y, label):
    return [idx for idx, ann in enumerate(y) if ann == label]


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
