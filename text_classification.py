## for data
import json
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for processing
import re
import nltk
## for bag-of-words
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing, \
    feature_selection, metrics
## for explainer
from lime import lime_text
## for word embedding
import gensim
import gensim.downloader as gensim_api
## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K
## for bert language model
import transformers


def attention_layer(inputs, neurons):
    ## code attention layer

    x = layers.Permute((2,1))(inputs)
    x = layers.Dense(neurons, activation="softmax")(x)
    x = layers.Permute((2,1), name="attention")(x)
    x = layers.multiply([inputs, x])
    return x

def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    '''
    Preprocess a string.
    :parameter
        :param text: string - name of column containing text
        :param lst_stopwords: list - list of stopwords to remove
        :param flg_stemm: bool - whether stemming is to be applied
        :param flg_lemm: bool - whether lemmitisation is to be applied
    :return
        cleaned text
    '''

    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in
    lst_stopwords]

    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    ## back to string from list
    text = " ".join(lst_text)

    return text

if __name__ == '__main__':

    lst_dics = []
    with open('data.json', mode='r', errors='ignore') as json_file:
        for dic in json_file:
            lst_dics.append( json.loads(dic) )
    ## print the first one
    lst_dics[0]

    ## create dtf
    dtf = pd.DataFrame(lst_dics)
    ## filter categories
    dtf = dtf[ dtf["category"].isin(['ENTERTAINMENT','POLITICS','TECH']) ][["category","headline"]]
    ## rename columns
    dtf = dtf.rename(columns={"category":"y", "headline":"text"})
    ## print 5 random rows
    dtf.sample(5)

    fig, ax = plt.subplots()
    fig.suptitle("y", fontsize=12)
    dtf["y"].reset_index().groupby("y").count().sort_values(by=
           "index").plot(kind="barh", legend=False,
            ax=ax).grid(axis='x')
    #plt.show()

    lst_stopwords = nltk.corpus.stopwords.words("english")

    dtf["text_clean"] = dtf["text"].apply(lambda x:
              utils_preprocess_text(x, flg_stemm=False, flg_lemm=True,
              lst_stopwords=lst_stopwords))
    print(dtf.head())

    ## split dataset
    dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.3)
    ## get target
    y_train = dtf_train["y"].values
    y_test = dtf_test["y"].values

    ## Count (classic BoW)
    vectorizer = feature_extraction.text.CountVectorizer(max_features=10000, ngram_range=(1, 2))

    ## Tf-Idf (advanced variant of BoW)
    vectorizer = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

    corpus = dtf_train["text_clean"]

    vectorizer.fit(corpus)
    X_train = vectorizer.transform(corpus)
    dic_vocabulary = vectorizer.vocabulary_

    #sns.heatmap(X_train.todense()[:, np.random.randint(0, X_train.shape[1], 100)] == 0, vmin=0, vmax=1, cbar=False).set_title(
    #    'Sparse Matrix Sample')

    word = "new york"
    print(dic_vocabulary[word])

    y = dtf_train["y"]
    X_names = vectorizer.get_feature_names()
    p_value_limit = 0.95
    dtf_features = pd.DataFrame()
    for cat in np.unique(y):
        chi2, p = feature_selection.chi2(X_train, y == cat)
        dtf_features = dtf_features.append(pd.DataFrame(
            {"feature": X_names, "score": 1 - p, "y": cat}))
        dtf_features = dtf_features.sort_values(["y", "score"],
                                                ascending=[True, False])
        dtf_features = dtf_features[dtf_features["score"] > p_value_limit]
    X_names = dtf_features["feature"].unique().tolist()

    for cat in np.unique(y):
        print("# {}:".format(cat))
        print("  . selected features:",
              len(dtf_features[dtf_features["y"] == cat]))
        print("  . top features:", ",".join(
            dtf_features[dtf_features["y"] == cat]["feature"].values[:10]))
        print(" ")

    vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
    vectorizer.fit(corpus)
    X_train = vectorizer.transform(corpus)
    dic_vocabulary = vectorizer.vocabulary_

    classifier = naive_bayes.MultinomialNB()

    ## pipeline
    model = pipeline.Pipeline([("vectorizer", vectorizer),
                               ("classifier", classifier)])
    ## train classifier
    model["classifier"].fit(X_train, y_train)
    ## test
    X_test = dtf_test["text_clean"].values
    predicted = model.predict(X_test)
    predicted_prob = model.predict_proba(X_test)

    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values

    ## Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(y_test, predicted)
    auc = metrics.roc_auc_score(y_test, predicted_prob,
                                multi_class="ovr")
    print("Accuracy:", round(accuracy, 2))
    print("Auc:", round(auc, 2))
    print("Detail:")
    print(metrics.classification_report(y_test, predicted))

    ## Plot confusion matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
           yticklabels=classes, title="Confusion matrix")
    plt.yticks(rotation=0)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ## Plot roc
    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:, i],
                                                 predicted_prob[:, i])
        ax[0].plot(fpr, tpr, lw=3,
                   label='{0} (area={1:0.2f})'.format(classes[i],
                                                      metrics.auc(fpr, tpr))
                   )
    ax[0].plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    ax[0].set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05],
              xlabel='False Positive Rate',
              ylabel="True Positive Rate (Recall)",
              title="Receiver operating characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(True)

    ## Plot precision-recall curve
    for i in range(len(classes)):
        precision, recall, thresholds = metrics.precision_recall_curve(
            y_test_array[:, i], predicted_prob[:, i])
        ax[1].plot(recall, precision, lw=3,
                   label='{0} (area={1:0.2f})'.format(classes[i],
                                                      metrics.auc(recall, precision))
                   )
    ax[1].set(xlim=[0.0, 1.05], ylim=[0.0, 1.05], xlabel='Recall',
              ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="best")
    ax[1].grid(True)
    plt.show()

    nlp = gensim_api.load("word2vec-google-news-300")

    # start the matrix (length of vocabulary x vector size) with all 0s
    embeddings = np.zeros((len(dic_vocabulary) + 1, 300))
    for word, idx in dic_vocabulary.items():
        ## update the row with vector
        try:
            embeddings[idx] = nlp[word]
        ## if word not in model then skip and the row stays all 0s
        except:
            pass

    ## input
    x_in = layers.Input(shape=(15,))
    ## embedding
    x = layers.Embedding(input_dim=embeddings.shape[0],
                         output_dim=embeddings.shape[1],
                         weights=[embeddings],
                         input_length=15, trainable=False)(x_in)
    ## apply attention
    x = attention_layer(x, neurons=15)
    ## 2 layers of bidirectional lstm
    x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2,
                                         return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2))(x)
    ## final dense layers
    x = layers.Dense(64, activation='relu')(x)
    y_out = layers.Dense(3, activation='softmax')(x)
    ## compile
    model = models.Model(x_in, y_out)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.summary()

    ## encode y
    dic_y_mapping = {n: label for n, label in
                     enumerate(np.unique(y_train))}
    inverse_dic = {v: k for k, v in dic_y_mapping.items()}
    y_train = np.array([inverse_dic[y] for y in y_train])
    ## train
    training = model.fit(x=X_train, y=y_train, batch_size=256,
                         epochs=10, shuffle=True, verbose=0,
                         validation_split=0.3)
    ## plot loss and accuracy
    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_' + metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    plt.show()

    ## test
    predicted_prob = model.predict(X_test)
    predicted = [dic_y_mapping[np.argmax(pred)] for pred in
                 predicted_prob]


