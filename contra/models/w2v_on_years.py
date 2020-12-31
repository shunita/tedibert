import os
import pickle
import time
import sys
sys.path.append('/home/shunita/fairemb/')

from collections import defaultdict
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="pandas")
import torch
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec
from nltk.tokenize import word_tokenize
from contra.constants import SAVE_PATH, FULL_PUMBED_2019_PATH
from contra.utils.pubmed_utils import read_year 
from contra import config
import wandb

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''
    def __init__(self, monitor_loss=True):
        self.epoch = 0
        self.monitor_loss = monitor_loss
        self.loss = 0
        self.start = time.time()
    
    def on_epoch_begin(self, model):
        print(f"Epoch #{self.epoch} start")

    def on_epoch_end(self, model):
        self.epoch += 1
        print(f"Epoch #{self.epoch} end in {(time.time()-self.start)/self.epoch} seconds per epoch.")
        if self.monitor_loss:
            latest_loss = model.get_latest_training_loss()
            loss_delta = latest_loss-self.loss
            self.loss = latest_loss
        else:
            loss_delta = -1
        wandb.log({'epoch': self.epoch, 
                   'time per epoch': (time.time()-self.start)/self.epoch, 
                   'loss': loss_delta})


class EmbeddingOnYears:
    def __init__(self, hparams):
        self.hparams = hparams
        self.start_year = hparams.start_year
        self.end_year = hparams.end_year

        self.embedding_size = 300
        self.window = 10
        self.min_count = 1
        self.iterations = hparams.max_epochs
        
        self.model_name = hparams.emb_algorithm
        self.data = []
        self.model = None
        monitor_loss = (self.model_name == 'w2v')
        self.epoch_logger = EpochLogger(monitor_loss)
        self.idf = defaultdict(int)

    def count_words_in_abstract(self, abstract):
        tokenized = word_tokenize(abstract)
        for word in set(tokenized):
            self.idf[word] += 1

    def populate_idf_dict(self):
        for year in range(self.start_year, self.end_year + 1):
            df = read_year(year)
            df['title_and_abstract'].apply(self.count_words_in_abstract)
        output_path = os.path.join(FULL_PUMBED_2019_PATH, f'idf_dict{self.start_year}_{self.end_year}.pickle')
        pickle.dump(self.idf, open(output_path, 'wb'))

    def load_data_for_w2v(self):
        for year in range(self.start_year, self.end_year + 1):
            year_sentences_path = os.path.join(FULL_PUMBED_2019_PATH, f'{year}_sentences.pickle')
            year_sentences_tokenized_path = os.path.join(FULL_PUMBED_2019_PATH, f'{year}_sentences_tokenized.pickle')
            if os.path.exists(year_sentences_tokenized_path):
                sentences1 = pickle.load(open(year_sentences_tokenized_path, 'rb'))
            else:
                sentences = pickle.load(open(year_sentences_path, 'rb'))
                sentences1 = []
                print(f'word-tokenizing {year} sentences')
                for sent in tqdm(sentences):
                    sentences1.append(word_tokenize(sent))
                pickle.dump(sentences1, open(year_sentences_tokenized_path, 'wb'))
            self.data.extend(sentences1)
        print(f'loaded {len(self.data)} sentences.')
        
    def load_data_for_doc2vec(self):
        for year in range(self.start_year, self.end_year + 1):
            year_path = os.path.join(FULL_PUMBED_2019_PATH, f'pubmed_{year}.csv')
            year_tokenized_path = os.path.join(FULL_PUMBED_2019_PATH, f'pubmed_{year}_tokenized_abstracts.pickle')
            if os.path.exists(year_tokenized_path):
                abstracts = pickle.load(open(year_tokenized_path, 'rb'))
            else:
                df = read_year(year)
                abstracts = df['title_and_abstract'].progress_apply(word_tokenize)
                pickle.dump(abstracts, open(year_tokenized_path, 'wb'))
            docs = [TaggedDocument(doc, [i]) for i, doc in abstracts.iteritems()]    
            self.data.extend(docs)
        
    def load_data(self):
        if self.model_name=='w2v':
            self.load_data_for_w2v()
        elif self.model_name == 'doc2vec':
            self.load_data_for_doc2vec()
        else:
            print(f"unsupported model name {self.model_name}")
            sys.exit()

    def fit(self):
        if len(self.data) == 0:
            self.load_data()
        if self.model_name=='w2v':
            self.model = Word2Vec(self.data, min_count=self.min_count, size=self.embedding_size,
                                  compute_loss=True, window=self.window, sg=0, iter=self.iterations, 
                                  workers=3, callbacks=[self.epoch_logger]) 
        elif self.model_name == 'doc2vec':
            self.model = Doc2Vec(self.data, min_count=self.min_count, vector_size=self.embedding_size,
                                 window=self.window, epochs=self.iterations,
                                 workers=3, callbacks=[self.epoch_logger])
        else:
            print(f"unsupported model name {self.model_name}")
            sys.exit()

    def save(self):
        if self.model_name == 'w2v':
            self.model.wv.save(os.path.join(SAVE_PATH, f"word2vec_{self.start_year}_{self.end_year}.wordvectors"))
        elif self.model_name == 'doc2vec':
            self.model.delete_temporary_training_data(keep_doctags_vectors=False, keep_inference=True)
            self.model.save(os.path.join(SAVE_PATH, f"doc2vec_{self.start_year}_{self.end_year}.vectors"))

    
            

class PretrainedW2V:
    
    def __init__(self, idf_path, vectors_path, ndocs, model_name='w2v'):
        self.model_name = model_name
        self.ndocs = ndocs
        self.idf_map = pickle.load(open(idf_path, 'rb'))
        if self.model_name == 'w2v':
            self.wv = KeyedVectors.load(vectors_path, mmap='r')
        #if self.model_name == 'doc2vec':
        #    self.model = Doc2Vec.load(path)

    def embed(self, tokenized_text):
        tf = defaultdict(int)
        for word in tokenized_text:
            tf[word] += 1
        weights = []
        vectors = []
        for word in tf:
            if word in self.wv:
                idf = self.idf_map[word]
                if idf == 0:
                    # This word did not appear in any documents, so we know nothing about it - we don't add it to the
                    # averaged vector
                    continue
                vectors.append(self.wv.get_vector(word))
                weights.append(tf[word]*np.log(self.ndocs/self.idf_map[word]))
        return np.average(np.stack(vectors), axis=0, weights=np.array(weights))
    
    def embed_batch(self, texts, device=None):
        return torch.as_tensor(np.stack([self.embed(text) for text in texts]), dtype=torch.float32, device=device)


class PretrainedOldNewW2V:
    def __init__(self, old_w2v, new_w2v):
        self.old_w2v = old_w2v
        self.new_w2v = new_w2v
        
    def embed_batch(self, tokenized_texts, is_news, device=None):
        vectors = []
        for text, is_new in zip(tokenized_texts, is_news):
            if is_new:
                vector = self.new_w2v.embed(text)
            else:
                vector = self.old_w2v.embed(text)
            vectors.append(vector)
        return torch.as_tensor(np.stack(vectors), dtype=torch.float32, device=device)


def read_w2v_model(year1: int, year2: int) -> PretrainedW2V:
    '''Read the pretrained embeddings of a year range.'''
    year_to_ndocs = pd.read_csv(os.path.join(DATA_PATH, 'year_to_ndocs.csv'), index_col=0,
                                dtype={'year': int, 'ndocs': int}).to_dict(orient='dict')['ndocs']
    vectors_path = os.path.join(SAVE_PATH, f"word2vec_{year1}_{year2}.wordvectors")
    idf_path = os.path.join(SAVE_PATH, f'idf_dict{year1}_{year2}.pickle')
    num_docs_in_range = sum([year_to_ndocs[year] for year in range(year1, year2+1)])
    w2v = PretrainedW2V(idf_path, vectors_path, ndocs=num_docs_in_range)
    return w2v


if __name__ == '__main__':
    hparams = config.parser.parse_args(['--name', 'W2VYearsIDF',
                                        '--emb_algorithm', 'doc2vec',
                                        '--start_year', '2010',
                                        '--end_year', '2013',
                                        '--max_epochs', '15'])
    wandb.init(project="Contra")
    wandb.run.name = f'{hparams.name}_{hparams.start_year}_{hparams.end_year}'
    wandb.config.update(hparams)
    wv = EmbeddingOnYears(hparams)
    wv.populate_idf_dict()
    #wv.fit()
    #wv.save()
