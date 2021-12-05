import numpy as np
import simplemma
from rocksdb.merge_operators import StringAppendOperator
import rocksdb
import json
import math
from enum import Enum

langdata = simplemma.load_data('en', 'it', 'de', 'es', 'fr', 'nl', 'ru')


class AggregationStrategy(Enum):
    SUM = 1
    MAX = 2
    AVG = 3
    TF_IDF = 4
    MAX_HIERARCHY = 5


class RocksDBDomainDisambiguator:

    def __init__(self, dbpath, name, id_to_domain, threshold=0.0, hierarchy=None, strategy=AggregationStrategy.MAX):
        if hierarchy is None:
            hierarchy = {}
        self.dbpath = dbpath
        self.id_to_domain = id_to_domain
        self.name = name
        self.threshold = threshold
        self.hierarchy = hierarchy
        opts = rocksdb.Options()
        opts.create_if_missing = True
        opts.merge_operator = StringAppendOperator()
        self.db = rocksdb.DB(self.dbpath, opts)
        self.domain_to_id = {v: k for k, v in id_to_domain.items()}
        self.number_of_words_per_domain = self.get_number_of_words_per_domain()
        self.strategy = strategy

    def get_number_of_words_per_domain(self):

        number_of_words_per_domain = self.db.get("number-of-words-per-domain".encode())

        if number_of_words_per_domain is not None:
            return {int(k): v for k, v in json.loads(number_of_words_per_domain).items()}

        print(f"Computing Number of Words per Domain Map for DB {self.name}")

        it = self.db.iteritems()
        it.seek_to_first()

        result = {int(k): 0 for k, v in self.id_to_domain.items()}

        for item in it:
            split = item[1].decode().split()
            i = 0
            while i < len(split):
                result[int(split[i].replace(",", ""))] += 1
                i += 2

        self.db.put("number-of-words-per-domain".encode(), json.dumps(result).encode())

        return result

    def extract_domains_with_duplicates(self, word):
        result = {}

        domain_string = self.db.get(word.encode())
        if domain_string is None:
            domain_string = self.db.get(simplemma.lemmatize(word, langdata).encode())

        if domain_string is None:
            return result

        split = domain_string.decode().split()
        i = 0
        while i < len(split):
            score = float(split[i + 1])
            if split[i] in result:
                result[int(split[i].replace(",", ""))].append(score)
            else:
                result[int(split[i].replace(",", ""))] = [score]
            i += 2
        return result

    def get_domains(self, word, word_weight=1.0):
        if self.strategy == AggregationStrategy.SUM:
            return self.get_domains_sum(word, word_weight)
        elif self.strategy == AggregationStrategy.MAX:
            return self.get_domains_max(word, word_weight)
        elif self.strategy == AggregationStrategy.AVG:
            return self.get_domains_avg(word, word_weight)
        elif self.strategy == AggregationStrategy.TF_IDF:
            return self.get_domains_tfidf(word, word_weight)
        elif self.strategy == AggregationStrategy.MAX_HIERARCHY:
            return self.get_domains_max_hierarchy(word, word_weight)

    def get_domains_max(self, word, word_weight=1.0):
        result = np.zeros((len(self.id_to_domain)))
        for domain_id, score_list in self.extract_domains_with_duplicates(word).items():
            result[int(domain_id)] = word_weight * max(score_list)
        return result

    def get_domains_sum(self, word, word_weight=1.0):
        result = np.zeros((len(self.id_to_domain)))
        for domain_id, score_list in self.extract_domains_with_duplicates(word).items():
            result[int(domain_id)] = word_weight * sum(score_list)
        return result

    def get_domains_avg(self, word, word_weight=1.0):
        result = np.zeros((len(self.id_to_domain)))
        for domain_id, score_list in self.extract_domains_with_duplicates(word).items():
            result[int(domain_id)] = word_weight * sum(score_list)
        s = sum(result)
        if s > 0:
            result = result / sum(result)
        return result

    def get_domains_tfidf(self, word, word_weight=1.0):
        result = np.zeros((len(self.id_to_domain)))
        domains_with_duplicates = self.extract_domains_with_duplicates(word)
        for domain_id, score_list in domains_with_duplicates.items():
            tf = sum(score_list) / math.log10(self.number_of_words_per_domain[domain_id])
            idf = math.log10(len(self.id_to_domain) / len(domains_with_duplicates))
            result[int(domain_id)] = word_weight * tf * idf
        return result

    def get_domains_max_hierarchy(self, word, word_weight=1.0):
        result = self.get_domains_max(word, word_weight)
        for domain_id, score in enumerate(result):
            if int(domain_id) in self.hierarchy:
                for super_domain in self.hierarchy[int(domain_id)]:
                    result[super_domain] += result[int(domain_id)]
        return result

    def print_domains(self, domains):
        domains_extracted = {self.id_to_domain[domain_id]: score for domain_id, score in enumerate(domains)}
        domains_extracted_ordered = {k: v for k, v in sorted(domains_extracted.items(), key=lambda item: item[1])}
        for d in domains_extracted_ordered:
            if domains_extracted_ordered[d]:
                print(f"{d}\t{domains_extracted_ordered[d]}")
