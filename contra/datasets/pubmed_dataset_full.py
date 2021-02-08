import os
import sys
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from contra.utils import text_utils as tu
from contra.utils.pubmed_utils import read_year, process_year_range_into_sentences, process_aact_year_range_to_sentences
from contra.utils.pubmed_utils import params_to_description, pubmed_version_to_folder
import pickle


class PubMedFullModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.start_year = hparams.start_year
        self.end_year = hparams.end_year
        self.test_size = 1 - hparams.train_test_split
        self.only_aact = hparams.only_aact_data
        self.relevant_abstracts = None
        self.year_to_indexes = {}
        self.year_to_pmids = {}
        self.by_sentence = hparams.by_sentence
        if self.by_sentence:
            self.text_utils = tu.TextUtils()
            self.sentences = {}
        self.batch_size = hparams.batch_size
        self.abstract_weighting_mode = hparams.abstract_weighting_mode
        self.pubmed_version = hparams.pubmed_version
        self.pubmed_folder = pubmed_version_to_folder(self.pubmed_version)
        self.desc = params_to_description(self.abstract_weighting_mode,
                                          only_aact_data=self.only_aact,
                                          pubmed_version=self.pubmed_version)
        if self.abstract_weighting_mode not in ('normal', 'subsample'):
            raise Exception(f"Unsupported option for abstract_weighting_mode = {self.abstract_weighting_mode}")

    def prepare_data(self):
        """happens only on one GPU."""
        if self.by_sentence:
            if not self.only_aact:
                process_year_range_into_sentences(self.start_year, self.end_year, self.pubmed_version, self.abstract_weighting_mode)

    def setup(self, stage=None):
        """happens on all GPUs."""
        if self.by_sentence:
            if self.only_aact:
                self.sentences = process_aact_year_range_to_sentences(self.pubmed_version, (self.start_year, self.end_year))
            else:
                self.sentences = []
                for year in range(self.start_year, self.end_year + 1):
                    year_sentences_path = os.path.join(self.pubmed_folder, f'{year}{self.desc}_sentences.pickle')
                    sentences = pickle.load(open(year_sentences_path, 'rb'))
                    self.sentences.extend(sentences)
            print(f'len(sentences) = {len(self.sentences)}')
            train_sentences, val_sentences = train_test_split(self.sentences, test_size=self.test_size, random_state=1)
            self.train = PubMedFullDataset(train_sentences, self.start_year, self.end_year, self.pubmed_version,
                                           by_sentence=True)
            self.val = PubMedFullDataset(val_sentences, self.start_year, self.end_year, self.pubmed_version,
                                         by_sentence=True)
        else:
            if self.only_aact:
                raise Exception("Currently unsupported: only_aact_data=True and by_sentence=False")
            else:
                current_index = 0
                for year in range(self.start_year, self.end_year+1):
                    relevant = read_year(year)
                    self.year_to_indexes[year] = (current_index, current_index+len(relevant))
                    self.year_to_pmids[year] = relevant.index.tolist()
                    current_index += len(relevant)
                self.relevant_abstracts = current_index
                train_indices, val_indices = train_test_split(range(self.relevant_abstracts), test_size=self.test_size,
                                                              random_state=1)
                self.train = PubMedFullDataset(train_indices, self.start_year, self.end_year, self.pubmed_version,
                                               year_to_indexes=self.year_to_indexes, year_to_pmids=self.year_to_pmids)
                self.val = PubMedFullDataset(val_indices, self.start_year, self.end_year, self.pubmed_version,
                                             year_to_indexes=self.year_to_indexes, year_to_pmids=self.year_to_pmids)

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=8)


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
