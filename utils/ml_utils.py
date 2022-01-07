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

def resample(X, y, stategy=None):
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
    elif stategy == "mlsmote_iterative":
        X_res, y_res = MLSMOTE_iterative(X, y)
    elif stategy == "mlsmote":
        X_res, y_res = MLSMOTE(X, y)
    else:
        X_res, y_res = utils.resample(X, y)
    irlb, irlb_mean_last = get_irlb(y_res)
    print(f"IRMean after resampling {irlb_mean_last} number of examples {len(X_res)} number of target {len(y_res)}")
    return X_res, y_res


class DomainTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, da, cv):
        self.da = da
        self.cv = cv

    def fit(self, X, y=None):
        self.words = self.cv.get_feature_names()
        print(f"Number of words {len(self.words)}")
        return self

    def transform(self, X, y=None):
        #print("Transform called")
        #print(type(X))
        #print(X.shape)
        d = np.zeros((X.shape[0], len(self.da.id_to_domain)))
        #print(f"Number of documents {X.shape[0]}")
        for row in range(0,X.shape[0]):
            doc_words_all = {self.words[col]: X[row, col] for col in X[row, :].nonzero()[1]}
            d[row] = self.da.get_domain_vector(doc_words_all)[0]

        return sp.hstack([X,d])
