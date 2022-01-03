import re
import simplemma
langdata = simplemma.load_data('en', 'it', 'de', 'es', 'fr', 'nl', 'ru')

WORD = re.compile(r'[a-zA-Z][a-zA-Z]+')

def regTokenize(text):
    words = [simplemma.lemmatize(word, langdata) for word in WORD.findall(text)]
    return words

class LemmaTokenizer:
    def __call__(self, doc):
        return regTokenize(doc)

