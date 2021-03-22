import nltk.data
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import wordnet
import re


class TextUtils:
    def __init__(self):
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def split_abstract_to_sentences(self, abstract):
        parts = abstract.split(';')
        sentences = []
        for part in parts:
            sentences.extend(self.sent_tokenizer.tokenize(part))
        return sentences

    @staticmethod
    def flatten_list_of_lists(lst):
        return [item for sublist in lst for item in sublist]
        
    def word_tokenize_abstract(self, abstract):
        sentences = self.split_abstract_to_sentences(abstract)
        words = []
        for s in sentences:
            words.extend(word_tokenize(s))
        return words

    @staticmethod
    def decide_on_word(word):
        w = word.lower().strip(string.punctuation)
        if TextUtils.represents_number(w):
            return "<NUMBER>"
        return w.lower()

    @staticmethod
    def represents_number(word_to_check):
        possible_numbers = re.findall(r'[\d\.]+', word_to_check)
        if len(possible_numbers) > 0 and len(possible_numbers[0]) == len(word_to_check):
            return True
        # syns = wordnet.synsets(word_to_check)
        # for s in syns:
        #     if s.definition().startswith("the cardinal number"):
        #         return True
        # if "-" in word_to_check:
        #     word = word_to_check.split("-")[0]
        #     syns = wordnet.synsets(word)
        #     for s in syns:
        #         if s.definition().startswith("the cardinal number"):
        #             return True
        return False

    @staticmethod
    def filter_word_list(word_list):
        # decide_on_word removes punctuation, lowercases the words and replaces numbers with a specific token.
        return [w for w in map(TextUtils.decide_on_word, word_list) if len(w) > 0]

    def word_tokenize(self, text):
        return TextUtils.filter_word_list(self.word_tokenize_abstract(text))
