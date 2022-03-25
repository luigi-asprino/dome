import os
import pickle
from scipy.sparse import csr_matrix
from utils.ml_utils import Strategies, resample, print_counters, get_class_distribution
import pandas as pd

folder = "/Users/lgu/Desktop/NOTime/EKR/experiments/lov_bench_laund_max_hp_tf/"

mlb = pickle.load(open(folder + "mlb.p", "rb"))

resampling_strategy = Strategies.MLSMOTE

for i in range(0, 10):

    print(f"Resampling fold {i}")

    fold_dir = folder + str(i) + "/"

    X_train = pickle.load(open(fold_dir + "X_train.p", "rb"))
    y_train = pickle.load(open(fold_dir + "y_train.p", "rb"))

    counter, lab_to_ann_set, counter_singular = get_class_distribution(mlb.inverse_transform(y_train))

    if not os.path.exists(fold_dir + resampling_strategy):
        os.mkdir(fold_dir + resampling_strategy)

    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    y_train = pd.DataFrame(y_train).reset_index(drop=True)

    X_train, y_train = resample(X_train, y_train, strategy=resampling_strategy)

    X_train = csr_matrix(X_train.values)
    y_train = y_train.values

    print_counters(y_train, mlb, clab=counter)

    pickle.dump(X_train, open(fold_dir + resampling_strategy+"/X_train.p", "wb"))
    pickle.dump(y_train, open(fold_dir + resampling_strategy+"/y_train.p", "wb"))
