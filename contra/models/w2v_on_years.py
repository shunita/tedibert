import os
import pickle

from gensim.models import Word2Vec, KeyedVectors
from contra.constants import SAVE_PATH, FULL_PUMBED_2019_PATH
from contra import config
from gensim.models import Word2Vec


class W2VOnYears:
    def __init__(self, hparams):
        super(W2VOnYears, self).__init__()
        self.hparams = hparams
        self.start_year = hparams.start_year
        self.end_year = hparams.end_year

        self.embedding_size = 300
        self.window = 10
        self.min_count = 1
        self.iterations = hparams.max_epochs

        self.w2v_model = Word2Vec(min_count=self.min_count, size=self.embedding_size,
                                  workers=3, window=self.window, sg=0, iter=self.iterations)
        self.sentence = []

    def load_sentences(self):
        for year in range(self.start_year, self.end_year + 1):
            year_sentences_path = os.path.join(FULL_PUMBED_2019_PATH, f'{year}_sentences.pickle')
            sentences = pickle.load(open(year_sentences_path, 'rb'))
            self.sentences.extend(sentences)

    def fit(self):
        if len(self.sentences) == 0:
            self.load_sentences()
        self.w2v_model.train(self.sentences)

    def save(self):
        self.w2v_model.wv.save(os.path.join(SAVE_PATH, f"word2vec_{self.start_year}_{self.end_year}.wordvectors"))

    def load_keyed_vectors(self, path):
        wv = KeyedVectors.load(path, mmap='r')
        return wv


if __name__ == '__main__':
    hparams = config.parser.parse_args(['--name', 'W2VYears',
                                        '--start_year', '2018',
                                        '--end_year', '2018',
                                        '--by_sentence',
                                        '--max_epochs', '15'])
    wv = W2VOnYears(hparams)
    wv.fit()
    wv.save()
