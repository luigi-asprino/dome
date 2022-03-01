import numpy as np
import logging

logging.basicConfig(level=logging.ERROR)
log = logging.getLogger(__name__)


def get_doc(doc, dictionary, ordered=True, as_dictionary=False, binary_score=False, use_word=True, remapping=None):
    if not as_dictionary:
        if ordered:
            return [(remap_word(word_id, dictionary, use_word, remapping), binarize(score, binary_score))
                    for word_id, score in sorted(doc, key=lambda x:x[1], reverse=True) if word_id in dictionary]
        else:
            return [(remap_word(word_id, dictionary, use_word, remapping), binarize(score, binary_score))
                    for word_id, score in doc if word_id in dictionary]
    else:
        return {remap_word(word_id, dictionary, use_word, remapping): binarize(score, binary_score)
                for word_id, score in doc if word_id in dictionary}


def binarize(i, bin):
    if bin:
        return 0 if not i else 1
    else:
        return i


def remap_word(word_id, dictionary, use_word, remapping):
    if use_word:
        return dictionary[word_id]
    else:
        if remapping is not None:
            if word_id in remapping:
                return remapping[word_id]
        else:
            return word_id





def print_doc(doc, dictionary, print_score=False):
    if print_score:
        print(" ".join([f"{word} {score}" for word, score in get_doc(doc, dictionary)]))
    else:
        print(" ".join([word for word, score in get_doc(doc, dictionary)]))


class CorpusFiltered:

    def __init__(self, input_corpus, quantile, d=None):
        self.input_corpus = input_corpus
        self.quantile = quantile
        self.dictionary = d

    def filter_doc(self, doc):
        if len(doc) == 0:
            return doc
        threshold = np.quantile([e[1] for e in doc], self.quantile)
        log.debug(f"Threshold {threshold}")
        if self.dictionary is None:
            return [e for e in doc if e[1] >= threshold and e[0] > 0]
        return [e for e in [ee for ee in doc if e[0] in self.dictionary] if e[1] > threshold and e[0] > 0]

    def __iter__(self):
        count = 0
        for doc in self.input_corpus:
            if count % 10000 == 0:
                print(f"{count} document processed")
            count += 1
            yield self.filter_doc(doc)

    def __getitem__(self, item):
        return self.filter_doc(self.input_corpus[item])