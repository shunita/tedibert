import os
import pandas as pd
from datetime import datetime
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from contra.constants import DATA_PATH


class PubMedModule(pl.LightningDataModule):

    def __init__(self, min_num_participants=1, pivot_datetime=datetime(2016, 1, 1), train_test_split=0.8):
        super().__init__()
        self.min_num_participants = min_num_participants
        self.pivot_datetime = pivot_datetime
        self.train_test_split = train_test_split

    def prepare_data(self):
        self.df = pd.read_csv(os.path.join(DATA_PATH, 'abstracts_and_population_tokenized_for_cui2vec.csv'),
                              usecols=['Unnamed: 0', 'female', 'male', 'title', 'abstract'], index_col='Unnamed: 0')
        date_df = pd.read_csv(os.path.join(DATA_PATH, 'abstracts_population_date_topics.csv'),
                              usecols=['Unnamed: 0', 'date'], index_col='Unnamed: 0')
        self.df = self.df.join(date_df)
        self.df = self.df[(~pd.isna(self.df['date'])) & (~pd.isna(self.df['female'])) & (~pd.isna(self.df['male']))]

        self.df['date'] = self.df['date'].map(lambda dt: datetime.strptime(dt, '%m/%d/%Y'))
        self.df['num_participants'] = self.df.apply(lambda row: row['female'] + row['male'], axis=1)
        self.df = self.df[self.df['num_participants'] >= self.min_num_participants]

    def setup(self, stage=None):
        new_df = self.df[self.df['date'] >= self.pivot_datetime]
        old_df = self.df[self.df['date'] < self.pivot_datetime]
        old_train_df, old_val_df = train_test_split(old_df, test_size=0.2)
        new_train_df, new_val_df = train_test_split(new_df, test_size=0.2)
        train_df = pd.concat([new_train_df, old_train_df])
        val_df = pd.concat([new_val_df, old_val_df])

        self.train = PubMedDataset(train_df, self.pivot_datetime)
        self.val = PubMedDataset(val_df, self.pivot_datetime)
        print("Loaded {} train samples and {} validation samples".format(len(train_df), len(val_df)))

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=128, num_workers=32)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=True, batch_size=128, num_workers=32)

    def test_dataloader(self):
        return DataLoader(self.val, shuffle=True, batch_size=128, num_workers=32)


class PubMedDataset(Dataset):
    def __init__(self, df, pivot_datetime):
        self.df = df
        self.pivot_datetime = pivot_datetime

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        row = self.df.iloc[index]
        text = '; '.join([row['title'], row['abstract']])
        is_new = torch.as_tensor(row['date'] >= self.pivot_datetime)
        female_ratio = torch.as_tensor(row['female'] / row['num_participants'])

        return {'text': text, 'is_new': is_new, 'female_ratio': female_ratio}
