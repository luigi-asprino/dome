import pickle
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, hamming_loss, accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from utils.ml_utils import Strategies
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import make_pipeline

folder = "/Users/lgu/Desktop/NOTime/EKR/experiments/lov_bench_laund_max_hp_tf/"
use_scaler = True
resampling_strategy = Strategies.MLSMOTE


classifiers = [
    RandomForestClassifier(random_state=42),
    KNeighborsClassifier(),
    ExtraTreesClassifier(),
    MultiOutputClassifier(xgb.XGBClassifier()),
    MultiOutputClassifier(AdaBoostClassifier()),
    MLPClassifier(max_iter=1000, solver='lbfgs', random_state=42)
]

mlb = pickle.load(open(folder + "mlb.p", "rb"))

estimator_strings = []

f = open(folder + '/classification_results.txt', 'w')

for clf in classifiers:

    estimator = clf.__class__.__name__
    if hasattr(clf, 'estimator'):
        estimator = f"{clf.__class__.__name__}-{clf.estimator.__class__.__name__}"

    estimator_folder = folder + "/" + estimator

    if use_scaler:
        pipe = make_pipeline(StandardScaler(), Normalizer(), clf)
    else:
        pipe = clf

    computed_metrics = {"hls": [], "accs": [], "f1_micro": [], "f1_macro": [], "f1_weight": [],
                        "p_micro": [], "r_micro": []}

    for i in range(0, 10):
        fold_dir = folder + str(i) + "/"
        resampling_dir = fold_dir + resampling_strategy + "/"

        X_train = pickle.load(open(resampling_dir + "X_train.p", "rb"))
        X_test = pickle.load(open(fold_dir + "X_test.p", "rb"))
        y_train = pickle.load(open(resampling_dir + "y_train.p", "rb"))
        y_test = pickle.load(open(fold_dir + "y_test.p", "rb"))

        print(f"Training {estimator} FOLD {i}")

        X_train = X_train.todense()

        pipe.fit(X_train, y_train)
        y_test_pred = pipe.predict(X_test)
        pickle.dump(pipe, open(resampling_dir + f"{estimator}_pipe.p", "wb"))

        hl = hamming_loss(y_test, y_test_pred)
        computed_metrics["hls"].append(hl)

        acc = accuracy_score(y_test, y_test_pred)
        computed_metrics["accs"].append(acc)

        computed_metrics["f1_micro"].append(f1_score(y_test, y_test_pred, average="micro", zero_division=0))
        computed_metrics["f1_macro"].append(f1_score(y_test, y_test_pred, average="macro", zero_division=0))
        computed_metrics["f1_weight"].append(f1_score(y_test, y_test_pred, average="weighted", zero_division=0))
        computed_metrics["p_micro"].append(precision_score(y_test, y_test_pred, average="micro", zero_division=0))
        computed_metrics["r_micro"].append(recall_score(y_test, y_test_pred, average="micro", zero_division=0))

        to_print = classification_report(y_test, y_test_pred, target_names=mlb.classes_, zero_division=0)
        print(estimator)
        print(to_print)
        f.write(f"{estimator} FOLD #{i}\n")
        f.write(to_print)

        to_print = f"Accuracy {acc:.3f} Hamming loss {hl:.3f}"
        print(to_print)
        f.write(to_print)

    # Print mean and std
    metric_means = {}
    metric_std = {}
    for metric_name, observations in computed_metrics.items():
        mean = np.array(observations).mean()
        metric_means[metric_name] = mean
        std = np.array(observations).std()
        metric_std[metric_name] = std
        to_print = f"{metric_name} mean {mean:.3f} std {std:.3f}\n"
        print(to_print)

    estimator_string = f"{estimator}\t" \
               f"{np.array(computed_metrics['f1_micro']).mean():.3f}\t" \
               f"{np.array(computed_metrics['f1_micro']).std():.3f}\t" \
               f"{np.array(computed_metrics['p_micro']).mean():.3f}\t" \
               f"{np.array(computed_metrics['p_micro']).std():.3f}\t" \
               f"{np.array(computed_metrics['r_micro']).mean():.3f}\t" \
               f"{np.array(computed_metrics['r_micro']).std():.3f}\t" \
               f"{np.array(computed_metrics['accs']).mean():.3f}\t{np.array(computed_metrics['accs']).std():.3f}\t" \
               f"{np.array(computed_metrics['hls']).mean():.3f}\t{np.array(computed_metrics['hls']).std():.3f}\t" \
               f"\n".replace(".", ",")
    f.write(estimator_string)
    estimator_strings.append(estimator_string)

for estimator_string in estimator_strings:
    print(estimator_string)
    f.write(estimator_string)

