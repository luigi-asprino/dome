import os
from utils.Utils import load_map_from_file, load_list_from_file, load_vectors_from_file
import bz2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
from sklearn.ensemble import RandomForestClassifier
from utils.Utils import get_stopwords, print_counter
import logging
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
np.random.seed(0)
from sklearn.model_selection import cross_val_score
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.getLogger("gensim").setLevel(logging.ERROR)
logging.getLogger("polyglot").setLevel(logging.ERROR)
trace = logging.getLogger("trace")
trace.setLevel(logging.INFO)
timings = logging.getLogger("timings")
timings.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
from sklearn.metrics import multilabel_confusion_matrix
from preprocessing.Tokenizer import LemmaTokenizer
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from utils.mlsmote import get_minority_instace, MLSMOTE, get_irlb, MLSMOTE_iterative
from sklearn.neural_network import MLPClassifier


id_to_domain = "/Users/lgu/workspace/ekr/dome/resources/20211126_input_unified/id2domain.tsv"
input_folder_corpus = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/Corpus_lov"
token_number = 8
uri_to_doc_id_file = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/output/index.tsv"
gold_standard = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/LOV_KD_annotations.tsv"
hierarchy_file = "/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/KD_hierarchy.tsv"

id_to_domain = load_map_from_file(id_to_domain)
domain_to_id = {k: v for v, k in id_to_domain.items()}

print(id_to_domain)

doc_ids = load_list_from_file(input_folder_corpus + "/doc_ids", token_number, extractid=True)
id2doc = {k: v for v, k in enumerate(doc_ids)}

print(id2doc)

uri_to_doc_id = load_map_from_file(uri_to_doc_id_file)
doc_id_to_uri = {k: v for v, k in uri_to_doc_id.items()}

print(doc_id_to_uri)

uri_to_gold_classes = load_vectors_from_file(gold_standard,  usecols=[0,1,2,3], nullstring="-")
print(uri_to_gold_classes)

stop = get_stopwords("stopwords.txt")

hierarchy = {}
for (k, v) in load_map_from_file(hierarchy_file).items():
    hierarchy[int(k)] = [int(kd.strip()) for kd in v.split(",")]

print(hierarchy)

data = []

for root, dirs, files in os.walk("/Users/lgu/Desktop/NOTime/EKR/LOV_experiment/output"):
    for filename in files:
        if (filename == "virtualdocument.txt.bz2"):
            key = os.path.basename(root)

            if doc_id_to_uri[key] in uri_to_gold_classes :
                txt = " ".join([str(line.decode("utf-8")).strip("\n") for line in bz2.open(os.path.join(root, filename),"r")])

                direct_klasses = uri_to_gold_classes[doc_id_to_uri[key]]
                undirect_classes = [k for k in direct_klasses]

                for klass in undirect_classes:
                    if domain_to_id[klass] in hierarchy:
                        for super_klass in hierarchy[domain_to_id[klass]]:
                            if id_to_domain[super_klass] not in undirect_classes:
                                undirect_classes.append(id_to_domain[super_klass])

                data.append([key, doc_id_to_uri[key],  undirect_classes, txt])

df = pd.DataFrame(data, columns=['Doc Key', 'Doc URI', 'Class Label', 'Text'])

X = df['Text']
y = df['Class Label']

# Preprocessing
cv = CountVectorizer(lowercase=True, stop_words=stop, tokenizer=LemmaTokenizer(), binary=True)
pipeline = Pipeline(
    [
        ("vect", cv),
        # ("tfidf", TfidfTransformer()),
    ]
)
X = pipeline.fit_transform(X)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)

## Resampling
print_counter(mlb.inverse_transform(y), id_to_domain)
X_df = pd.DataFrame(data=X.todense())
y_df = pd.DataFrame(data=y, columns=mlb.classes_)

X, y = MLSMOTE_iterative(X_df, y_df)

X = X.values
y = y.values

print("\n\n RESAMPLING \n\n")
print_counter(mlb.inverse_transform(y), id_to_domain)

# Test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.1)

classifiers = [
    MLPClassifier(solver='lbfgs'),
    #RandomForestClassifier(class_weight="balanced"),
    #KNeighborsClassifier(),
    #RadiusNeighborsClassifier()
]

for clf in classifiers:

    estimator = clf.__class__.__name__
    if hasattr(clf, 'estimator'):
        estimator = f"{clf.__class__.__name__} {clf.estimator.__class__.__name__}"

    scoring = "f1_weighted"
    scores = cross_val_score(clf, X, y, cv=10, scoring=scoring)

    clf.fit(X_train, y_train)

    # Get predictions for test data
    y_test_pred = clf.predict(X_test)

    # Generate multiclass confusion matrices
    matrices = multilabel_confusion_matrix(y_test, y_test_pred)

    print(classification_report(y_test, y_test_pred))
    print(f"{estimator} {scores.mean()} {scoring} with a standard deviation of {scores.std()}\n")




