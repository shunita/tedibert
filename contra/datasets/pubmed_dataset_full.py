import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from contra.constants import FULL_PUMBED_2018_PATH, PUBMED_SHARDS

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


class PubMedFullModule(pl.LightningDataModule):

    def __init__(self, start_year=2018, end_year=2018, test_size=0.2):
        super().__init__()
        self.start_year = datetime(start_year, 1, 1)
        self.end_year = datetime(end_year, 12, 31)
        self.test_size = test_size
        self.df = None
        self.relevant_abstracts = None
        self.shard_to_indexes = []

    def prepare_data(self):
        # Holding all shards in memory is too much: 50 shards*600 MB (per file).
        # Instead, we remember which indexes belong to which file, and read that file only when we have to.
        # TODO: for just one year, we can hold everything in memory (should be around 1.5 GB).
        # That's why we have to use Shuffle=False in all the dataloaders.
        current_index = 0
        for i in tqdm(range(PUBMED_SHARDS)):
            relevant = read_shard(os.path.join(FULL_PUMBED_2018_PATH, f'pubmed_v2_shard_{i}.csv'), 
                                  self.start_year, 
                                  self.end_year)
            self.shard_to_indexes.append((current_index, current_index+len(relevant)))
            current_index += len(relevant)
        self.relevant_abstracts = current_index

    def setup(self, stage=None):
        train_indices, val_indices = train_test_split(range(self.relevant_abstracts), test_size=self.test_size)
        self.train = PubMedFullDataset(sorted(train_indices), self.shard_to_indexes, self.start_year, self.end_year)
        self.val = PubMedFullDataset(sorted(val_indices), self.shard_to_indexes, self.start_year, self.end_year)

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=False, batch_size=128, num_workers=32)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=128, num_workers=32)

    def test_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=128, num_workers=32)


class PubMedFullDataset(Dataset):
    def __init__(self, indexes, shard_to_index, start_year, end_year):
        self.indexes = indexes
        self.shard_to_index = shard_to_index
        self.start_year = start_year
        self.end_year = end_year
        self.df = None
        self.current_df_name = None

    def index_to_filename(self, index):
        for shard, interval in enumerate(self.shard_to_index):
            if interval[0] <= index < interval[1]:
                return interval[0], os.path.join(FULL_PUMBED_2018_PATH, f'pubmed_v2_shard_{shard}.csv')
        raise IndexError

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        first_index_in_shard, shard_fname = self.index_to_filename(index)
        if self.current_df_name is None or self.current_df_name != shard_fname:
            df = read_shard(shard_fname, self.start_year, self.end_year)
            self.current_df_name = shard_fname
        row = self.df.iloc[index - first_index_in_shard]
        text = '; '.join([row['title'], row['abstract']])
        return {'text': text}
