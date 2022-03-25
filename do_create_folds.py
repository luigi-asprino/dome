import os
import pickle
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer

folder = "/Users/lgu/Desktop/NOTime/EKR/experiments/lov_bench_laund_max_hp_tf/"
n_components = 100

print("Loading X and y..")
X = pickle.load(open(folder + "X_pre.p", "rb"))
y = pickle.load(open(folder + "y_pre.p", "rb"))
print("Loaded..")

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y[0])
X = csr_matrix(X.values)
pickle.dump(mlb, open(folder+"mlb.p", "wb"))

if n_components is None:
    svd = TruncatedSVD(n_components=min(X.shape[0], X.shape[1]))
else:
    svd = TruncatedSVD(n_components=n_components)

X = svd.fit_transform(X)
pickle.dump(svd, open(folder+"svd.p", "wb"))

mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)

n_fold = 0

for train_index, test_index in mskf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    fold_dir = folder + "/" + str(n_fold) + "/"
    if not os.path.exists(fold_dir):
        os.mkdir(fold_dir)
    pickle.dump(X_train, open(fold_dir+"/X_train.p", "wb"))
    pickle.dump(y_train, open(fold_dir+"/y_train.p", "wb"))
    pickle.dump(X_test, open(fold_dir+"/X_test.p", "wb"))
    pickle.dump(y_test, open(fold_dir+"/y_test.p", "wb"))
    n_fold = n_fold + 1