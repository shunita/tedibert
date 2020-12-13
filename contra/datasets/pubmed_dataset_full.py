import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from contra.constants import FULL_PUMBED_2019_PATH

# helper function

def read_shard(path, start_date, end_date):
    # fields: 'title', 'abstract', 'labels', 'pub_types', 'date', 'file', 'mesh_headings', 'keywords'
    print(f'reading pubmed shard: {path}')
    df = pd.read_csv(path, index_col=0)
    df = df.dropna(subset=['date'], axis=0)
    df['date'] = df['date'].map(lambda dt: datetime.strptime(dt, '%Y-%m-%d'))
    relevant = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    print(f'finished reading shard. found {len(relevant)} records with matching dates.')
    return relevant


def read_year(path_or_year):
    path = path_or_year
    if type(path_or_year) == int: # it's a year
        path = os.path.join(FULL_PUMBED_2019_PATH, f'pubmed_{path_or_year}.csv')
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    df = pd.read_csv(path, index_col=0)
    df = df.dropna(subset=['date'], axis=0)
    return df


class PubMedFullModule(pl.LightningDataModule):

    def __init__(self, start_year=2018, end_year=2018, test_size=0.2, by_sentence=True):
        super().__init__()
        self.start_year = start_year
        self.end_year = end_year
        self.test_size = test_size
        self.relevant_abstracts = None
        self.year_to_indexes = {}
        self.year_to_pmids = {}
        self.by_sentence = by_sentence

    def prepare_data(self):
        # We remember which indexes belong to which year, so we know which folder contains them.
        current_index = 0
        for year in range(self.start_year, self.end_year+1):
            relevant = read_year(year)
            self.year_to_indexes[year] = (current_index, current_index+len(relevant))
            self.year_to_pmids[year] = relevant.index.tolist()
            current_index += len(relevant)
        self.relevant_abstracts = current_index

    def setup(self, stage=None):
        train_indices, val_indices = train_test_split(range(self.relevant_abstracts), test_size=self.test_size)
        self.train = PubMedFullDataset(train_indices, self.year_to_indexes, self.year_to_pmids, self.start_year, self.end_year)
        self.val = PubMedFullDataset(val_indices, self.year_to_indexes, self.year_to_pmids, self.start_year, self.end_year)

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=False, batch_size=16, num_workers=3)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=16, num_workers=3)

    def test_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=16, num_workers=3)


class PubMedFullDataset(Dataset):
    def __init__(self, indexes, year_to_indexes, year_to_pmids, start_year, end_year):
        self.indexes = indexes
        self.year_to_indexes = year_to_indexes
        self.year_to_pmids = year_to_pmids
        self.start_year = start_year
        self.end_year = end_year
        self.df = None
        self.current_df_name = None

    def index_to_filename(self, index):
        for year, interval in self.year_to_indexes.items():
            if interval[0] <= index < interval[1]:
                pmid = self.year_to_pmids[year][index-interval[0]]
                return os.path.join(FULL_PUMBED_2019_PATH, str(year), f'{pmid}.csv')
        raise IndexError

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        fname = self.index_to_filename(index)
        df = pd.read_csv(fname, index_col=0)
        row = df.iloc[0]
        text = '; '.join([str(row['title']), str(row['abstract'])])
        return {'text': text}
