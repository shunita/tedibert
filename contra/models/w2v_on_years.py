import os
import pickle
import time
import sys
sys.path.append('/home/shunita/fairemb/')

from collections import defaultdict
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="pandas")
import pandas as pd
import torch
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from nltk.tokenize import word_tokenize
from contra.constants import SAVE_PATH, FULL_PUMBED_2019_PATH, FULL_PUMBED_2020_PATH, DATA_PATH, DEFAULT_PUBMED_VERSION
from contra.utils.pubmed_utils import read_year, read_year_to_ndocs, process_year_range_into_sentences, \
    params_to_description, load_aact_data, process_aact_year_range_to_sentences, clean_abstracts, \
    df_to_tokenized_sentence_list
from contra.utils import text_utils as tu
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

        self.embedding_size = 2000
        self.window = 10
        self.min_count = 1
        self.iterations = hparams.max_epochs
        
        self.abstract_weighting_mode = hparams.abstract_weighting_mode
        self.pubmed_version = hparams.pubmed_version
        self.pubmed_folder = {2019: FULL_PUMBED_2019_PATH, 2020: FULL_PUMBED_2020_PATH}[self.pubmed_version]
        
        self.model_name = hparams.emb_algorithm
        self.df = None
        self.data = []
        self.model = None
        monitor_loss = (self.model_name == 'w2v')
        self.epoch_logger = EpochLogger(monitor_loss)

        if self.abstract_weighting_mode not in ('normal', 'subsample'):
            sys.exit()
        
        self.only_aact_data = hparams.only_aact_data
        self.desc = params_to_description(self.abstract_weighting_mode, self.only_aact_data, self.pubmed_version)

        self.idf = defaultdict(int)
        self.idf_filename = f'idf_dict{self.start_year}_{self.end_year}{self.desc}.pickle'
        self.text_utils = tu.TextUtils()

    def count_words_in_abstract(self, abstract):
        tokenized = self.text_utils.word_tokenize(abstract)
        for word in set(tokenized):
            self.idf[word] += 1

    def populate_idf_dict(self):
        output_path = os.path.join(self.pubmed_folder, self.idf_filename)
        if os.path.exists(output_path):
            print(f"IDF path already exists: {output_path}")
            return
        print("Generating idf dict")
        if self.only_aact_data:
            if self.df is None:
                self.load_data()
            self.df['title_and_abstract'].apply(self.count_words_in_abstract)
        else:
            for year in range(self.start_year, self.end_year + 1):
                df = read_year(year, version=self.pubmed_version, subsample=(self.abstract_weighting_mode=='subsample'))
                df['title_and_abstract'].apply(self.count_words_in_abstract)
        print(f"Saving IDF dict to path: {output_path}")
        pickle.dump(self.idf, open(output_path, 'wb'))

    def load_data_for_w2v(self):
        if self.only_aact_data:
            # self.df = load_aact_data(self.pubmed_version, (self.start_year, self.end_year), sample=False)
            self.df = pd.read_csv(os.path.join(DATA_PATH, 'pubmed2020_assigned.csv'), index_col=0)
            self.df = self.df[self.df['assignment'] == 0]  # only train samples, like in BOW
            self.df = clean_abstracts(self.df)
            self.data = df_to_tokenized_sentence_list(self.df)
        else:
            for year in range(self.start_year, self.end_year + 1):
                year_sentences_path = os.path.join(self.pubmed_folder, f'{year}{self.desc}_sentences.pickle')
                year_sentences_tokenized_path = os.path.join(self.pubmed_folder, f'{year}{self.desc}_sentences.pickle')
                if os.path.exists(year_sentences_tokenized_path):
                    print(f"Reading tokenized sentences for year {year}")
                    sentences1 = pickle.load(open(year_sentences_tokenized_path, 'rb'))
                else:
                    if not os.path.exists(year_sentences_path):
                        print(f"Splitting year to sentences, mode: {self.abstract_weighting_mode}")
                        process_year_range_into_sentences(year, year,
                                                          abstract_weighting_mode=self.abstract_weighting_mode,
                                                          pubmed_version=self.pubmed_version)
                    sentences = pickle.load(open(year_sentences_path, 'rb'))
                    print(f"Tokenizing sentences of {year}, mode: {self.abstract_weighting_mode}")
                    sentences1 = []
                    for sent in tqdm(sentences):
                        sentences1.append(word_tokenize(sent))
                    pickle.dump(sentences1, open(year_sentences_tokenized_path, 'wb'))
                self.data.extend(sentences1)
        print(f'loaded {len(self.data)} sentences.')

    def load_data(self):
        if self.model_name == 'w2v':
            self.load_data_for_w2v()
        else:
            print(f"unsupported model name {self.model_name}")
            sys.exit()

    def fit(self):
        if len(self.data) == 0:
            self.load_data()
        if self.model_name == 'w2v':
            self.model = Word2Vec(self.data, min_count=self.min_count, size=self.embedding_size,
                                  compute_loss=True, window=self.window, sg=0, negative=5, iter=self.iterations,
                                  workers=3, callbacks=[self.epoch_logger])
        else:
            print(f"unsupported model name {self.model_name}")
            sys.exit()

    def save(self):
        if self.model_name == 'w2v':
            self.model.wv.save(os.path.join(SAVE_PATH, f"word2vec_{self.start_year}_{self.end_year}{self.desc}.wordvectors"))
    

class PretrainedW2V:
    def __init__(self, idf_path, vectors_path, ndocs, model_name='w2v'):
        self.model_name = model_name
        self.ndocs = ndocs
        self.idf_map = pickle.load(open(idf_path, 'rb'))
        if self.model_name == 'w2v':
            self.wv = KeyedVectors.load(vectors_path, mmap='r')
            self.emb_size = 300

    def embed(self, tokenized_text, stats=False, agg=True):
        not_found = 0
        tf = defaultdict(int)
        for word in tokenized_text:
            tf[word] += 1
        weights = []
        vectors = []
        for word in tf:
            if word in self.wv:
                idf = self.idf_map[word]
                if idf > 0:
                    vec = self.wv.get_vector(word)
                else:  # idf == 0
                    # try to find the word in lowercase
                    idf = self.idf_map[word.lower()]
                    if idf > 0:
                        vec = self.wv.get_vector(word.lower())
                    else:
                        # This word did not appear in any documents, even in lowercase. 
                        # We know nothing about it - so we don't add it to the averaged vector.
                        not_found += 1
                        continue
                vectors.append(vec)
                weights.append(tf[word]*np.log(self.ndocs/idf))
        if len(vectors) == 0:
            if stats:
                return np.zeros(self.emb_size), not_found, len(tokenized_text)
            else:
                return np.zeros(self.emb_size)
        ret = np.average(np.stack(vectors), axis=0, weights=np.array(weights))
        if stats:
            return ret, not_found, len(tokenized_text)
        return ret
    
    def embed_batch(self, tokenized_texts, device=None):
        return torch.as_tensor(np.stack([self.embed(text) for text in tokenized_texts]), dtype=torch.float32, device=device)


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


def read_w2v_model(year1: int, year2: int,
                   abstract_weighting_mode='normal',
                   pubmed_version=DEFAULT_PUBMED_VERSION,
                   only_aact_data=True) -> PretrainedW2V:
    '''Read the pretrained embeddings of a year range.'''
    year_to_ndocs = read_year_to_ndocs(version=pubmed_version)
    desc = params_to_description(abstract_weighting_mode, only_aact_data, pubmed_version)
    vectors_path = os.path.join(SAVE_PATH, f"word2vec_{year1}_{year2}{desc}.wordvectors")
    idf_path = os.path.join(FULL_PUMBED_2019_PATH, f'idf_dict{year1}_{year2}{desc}.pickle')
    num_docs_in_range = sum([year_to_ndocs[year] for year in range(year1, year2+1)])
    w2v = PretrainedW2V(idf_path, vectors_path, ndocs=num_docs_in_range)
    print(f"Read w2v vectors from {vectors_path} and idf map from {idf_path}.")
    return w2v


if __name__ == '__main__':
    hparams = config.parser.parse_args(['--name', 'W2VYears+IDF_aact2010-2018',
                                        '--emb_algorithm', 'w2v',
                                        '--abstract_weighting_mode', 'normal', #subsample
                                        '--start_year', '2010',
                                        '--end_year', '2018',
                                        '--pubmed_version', '2020',
                                        '--max_epochs', '45',
                                        '--only_aact_data'])
    wandb.init(project="Contra")
    wandb.run.name = f'{hparams.name}'
    wandb.config.update(hparams)
    wv = EmbeddingOnYears(hparams)
    wv.populate_idf_dict()
    wv.fit()
    wv.save()
