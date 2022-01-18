from gensim import corpora, models
import pickle


class CorpusFiltered:

    def __init__(self, input_corpus, dic):
        self.input_corpus = input_corpus
        self.dictionary = dic

    def __iter__(self):
        for doc in self.input_corpus:
            yield [e for e in doc if e[0] in self.dictionary]

#laundromat_corpus = "/Users/lgu/Dropbox/Backups/Corpus_lod"
laundromat_corpus = "/media/hd1000/Corpus_Lod"
tfidf_corpus_file = laundromat_corpus + "/tfidf_corpus"
dictionary_file = laundromat_corpus + "/dictionary"
corpus_tfidf = corpora.MmCorpus(tfidf_corpus_file)
dictionary = corpora.Dictionary.load(dictionary_file)
dictionary.filter_extremes(no_below=50, no_above=0.3, keep_n=None)
print(f"Dictionary filterd {len(dictionary)}")
#corpus_tfidf_filtered = [[e for e in doc if e[0] in dictionary] for doc in corpus_tfidf]
corpus_tfidf_filtered = CorpusFiltered(corpus_tfidf, dictionary)
print(f"Corpus TF-iDF filtered")
#pickle.dump(corpus_tfidf_filtered, open(laundromat_corpus+"/corpus_tf_idf_0.3.p","wb"))
corpora.MmCorpus.serialize(laundromat_corpus+"/corpus_tf_idf_0.3.p", corpus_tfidf_filtered)
print(f"Corpus dumped")
lda = models.LdaModel(corpus_tfidf_filtered, id2word=dictionary, num_topics=300)
print("Model LDA computed")
lda.save(laundromat_corpus+"/lda_model_100k")
print("Model LDA saved")
corpus_lda = lda[corpus_tfidf_filtered]
print("Corpus LDA computed")
corpora.MmCorpus.serialize(laundromat_corpus+"/corpus_lda_100k", corpus_lda)
print("Corpus LDA saved")

from utils.Utils import load_list_from_file
token_number = 8
doc_ids = load_list_from_file(laundromat_corpus + "/doc_ids", token_number, extractid=True)
