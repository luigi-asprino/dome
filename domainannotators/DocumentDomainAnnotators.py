import numpy as np
from enum import Enum

class DocumentAnnotatorAggregationStrategy(Enum):
    SUM = 1
    SUM_NORM = 2
    CENTROID = 3

class SimpleDocumentAnnotator:

    def __init__(self, id_to_dictionary_token, id_to_domain, sdds, words_to_exclude=[], strategy=DocumentAnnotatorAggregationStrategy.CENTROID):
        self.id_to_dictionary_token = id_to_dictionary_token
        self.words_to_exclude = words_to_exclude
        self.id_to_domain = id_to_domain
        self.sdds = sdds
        self.strategy = strategy

    def inspect_document(self, doc_words_all):
        word_per_domain = {}
        not_domain = []
        for w in doc_words_all:
            word_domains = np.zeros((len(self.id_to_domain)))
            for sdd in self.sdds:
                word_domains_retrieved = sdd.get_domains(w, doc_words_all[w])
                word_domains += word_domains_retrieved

                if not word_domains_retrieved.any():
                    not_domain.append((w, doc_words_all[w], sdd.name))
                else:
                    for domain_id, score in enumerate(word_domains):
                        if score:
                            if domain_id in word_per_domain:
                                word_per_domain[domain_id].append((w, score))
                            else:
                                word_per_domain[domain_id] = [(w, score)]

        return word_per_domain, not_domain

    def get_domain_vector(self, doc):
        if (self.strategy == DocumentAnnotatorAggregationStrategy.SUM):
            return self.get_domain_vector_sum(doc)
        elif (self.strategy == DocumentAnnotatorAggregationStrategy.SUM_NORM):
            return self.get_domain_vector_sum_norm(doc)
        elif (self.strategy == DocumentAnnotatorAggregationStrategy.CENTROID):
            return self.get_domain_vector_centroid(doc)

    def get_doc_words_all(self, doc):
        return {self.id_to_dictionary_token[tf[0]]: tf[1] for tf in doc
                if (tf[0] > 0 and self.id_to_dictionary_token[tf[0]] not in self.words_to_exclude)}

    def get_domain_vector_sum(self, doc):
        doc_words_all = self.get_doc_words_all(doc)
        domain_vector = np.zeros((len(self.id_to_domain)))
        for w in doc_words_all:
            word_domains = np.zeros((len(self.id_to_domain)))
            for sdd in self.sdds:
                word_domains += sdd.get_domains(w, doc_words_all[w])
            domain_vector += word_domains
        return domain_vector, doc_words_all

    def get_domain_vector_sum_norm(self, doc):
        domain_vector, doc_words_all = self.get_domain_vector_sum(doc)
        norm = np.linalg.norm(domain_vector)
        if norm > 0:
            domain_vector = domain_vector / norm
        return domain_vector, doc_words_all

    def get_domain_vector_centroid(self, doc):
        doc_words_all = self.get_doc_words_all(doc)
        domain_vector = np.mean([np.mean([sdd.get_domains(w, doc_words_all[w]) for sdd in self.sdds], axis=0) for w in doc_words_all], axis=0)
        return domain_vector, doc_words_all
