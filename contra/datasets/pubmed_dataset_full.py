import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from contra.utils import text_utils as tu
from contra.utils.pubmed_utils import read_year, process_year_range_into_sentences, pubmed_version_to_folder
import pickle


class PubMedFullModule(pl.LightningDataModule):

    def __init__(self, start_year=2018, end_year=2018, test_size=0.2, by_sentence=True, abstract_weighting_mode='normal', pubmed_version=2019):
        super().__init__()
        self.start_year = start_year
        self.end_year = end_year
        self.test_size = test_size
        self.relevant_abstracts = None
        self.year_to_indexes = {}
        self.year_to_pmids = {}
        self.by_sentence = by_sentence
        if self.by_sentence:
            self.text_utils = tu.TextUtils()
            self.sentences = {}
        self.abstract_weighting_mode = abstract_weighting_mode
        self.pubmed_version = pubmed_version
        self.pubmed_folder = pubmed_version_to_folder(self.pubmed_version)
        if self.abstract_weighting_mode == 'normal':
            self.desc = ''
        if self.abstract_weighting_mode == 'subsample':
            self.desc='sample'
        else:
            print(f"Unsupported option for abstract_weighting_mode = {self.abstract_weighting_mode}")
            sys.exit()

    def prepare_data(self):
        """happens only on one GPU."""
        if self.by_sentence:
            process_year_range_into_sentences(self.start_year, self.end_year, self.pubmed_version, self.abstract_weighting_mode)


    def setup(self, stage=None):
        """happens on all GPUs."""
        if self.by_sentence:
            self.sentences = []
            for year in range(self.start_year, self.end_year + 1):
                year_sentences_path = os.path.join(self.pubmed_folder, f'{year}{self.desc}_sentences.pickle')
                sentences = pickle.load(open(year_sentences_path, 'rb'))
                self.sentences.extend(sentences)
            print(f'len(sentences) = {len(self.sentences)}')
            train_sentences, val_sentences = train_test_split(self.sentences, test_size=self.test_size)
            self.train = PubMedFullDataset(train_sentences, self.start_year, self.end_year,
                                           by_sentence=True)
            self.val = PubMedFullDataset(val_sentences, self.start_year, self.end_year,
                                         by_sentence=True)
        else:
            current_index = 0
            for year in range(self.start_year, self.end_year+1):
                relevant = read_year(year)
                self.year_to_indexes[year] = (current_index, current_index+len(relevant))
                self.year_to_pmids[year] = relevant.index.tolist()
                current_index += len(relevant)
            self.relevant_abstracts = current_index
            train_indices, val_indices = train_test_split(range(self.relevant_abstracts), test_size=self.test_size)
            self.train = PubMedFullDataset(train_indices, self.start_year, self.end_year, self.pubmed_version,
                                           year_to_indexes=self.year_to_indexes, year_to_pmids=self.year_to_pmids)
            self.val = PubMedFullDataset(val_indices, self.start_year, self.end_year, self.pubmed_version,
                                         year_to_indexes=self.year_to_indexes, year_to_pmids=self.year_to_pmids)

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=150, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=150, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=150, num_workers=8)


class PubMedFullDataset(Dataset):
    def __init__(self, indexes_or_sentences, start_year, end_year, pubmed_version,
                 year_to_indexes=None, year_to_pmids=None,
                 by_sentence=False):
        if by_sentence:
            self.sentences = indexes_or_sentences
        else:
            self.indexes = indexes_or_sentences
        self.year_to_indexes = year_to_indexes
        self.year_to_pmids = year_to_pmids
        self.start_year = start_year
        self.end_year = end_year
        self.by_sentence = by_sentence
        self.pubmed_folder = pubmed_version_to_folder(pubmed_version)

    def index_to_filename(self, index):
        for year, interval in self.year_to_indexes.items():
            if interval[0] <= index < interval[1]:
                pmid = self.year_to_pmids[year][index-interval[0]]
                return os.path.join(self.pubmed_folder, str(year), f'{pmid}.csv')
        raise IndexError

    def __len__(self):
        if self.by_sentence:
            return len(self.sentences)
        return len(self.indexes)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        if self.by_sentence:
            return {'text': self.sentences[index]}
        # Not by_sentence: (based on every abstract being in a different file)
        fname = self.index_to_filename(index)
        df = pd.read_csv(fname, index_col=0)
        row = df.iloc[0]
        text = '; '.join([str(row['title']), str(row['abstract'])])
        return {'text': text}

