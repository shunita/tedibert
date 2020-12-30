import os
import pandas as pd
from datetime import datetime
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from contra.constants import DATA_PATH


class PubMedModule(pl.LightningDataModule):

    def __init__(self, min_num_participants=1, first_year_range=(2010,2013), second_year_range=(2018,2018), train_test_split=0.8):
        super().__init__()
        self.min_num_participants = min_num_participants
        self.first_time_range = (datetime(first_year_range[0], 1, 1), datetime(first_year_range[1], 12, 31))
        self.second_time_range = (datetime(second_year_range[0], 1, 1), datetime(second_year_range[1], 12, 31))
        self.train_test_split = train_test_split

    def prepare_data(self):
        self.df = pd.read_csv(os.path.join(DATA_PATH, 'pubmed2019_abstracts_with_participants.csv'), index_col=0)
        self.df = self.df.dropna(subset=['date', 'male', 'female'], axis=0)
        self.df['title'] = self.df['title'].fillna('')
        self.df['title'] = self.df['title'].apply(lambda x: x.strip('[]')) 
        self.df['date'] = self.df['date'].map(lambda dt: datetime.strptime(dt, '%Y-%m-%d'))
        self.df['num_participants'] = self.df['female'] + self.df['male']
        self.df = self.df[self.df['num_participants'] >= self.min_num_participants]

    def setup(self, stage=None):
        old_df = self.df[(self.df['date'] >= self.first_time_range[0]) & (self.df['date'] <= self.first_time_range[1])]
        new_df = self.df[(self.df['date'] >= self.second_time_range[0]) & (self.df['date'] <= self.second_time_range[1])]
        
        old_train_df, old_val_df = train_test_split(old_df, test_size=0.2)
        new_train_df, new_val_df = train_test_split(new_df, test_size=0.2)
        train_df = pd.concat([new_train_df, old_train_df])
        val_df = pd.concat([new_val_df, old_val_df])

        self.train = PubMedDataset(train_df, self.first_time_range, self.second_time_range)
        self.val = PubMedDataset(val_df, self.first_time_range, self.second_time_range)
        print("Loaded {} train samples and {} validation samples".format(len(train_df), len(val_df)))

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=128, num_workers=32)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=128, num_workers=32)

    def test_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=128, num_workers=32)


class PubMedDataset(Dataset):
    def __init__(self, df, first_range, second_range):
        self.df = df
        self.first_range = first_range
        self.second_range = second_range

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        row = self.df.iloc[index]
        text = '; '.join([row['title'], row['abstract']])
        # at this point we don't have abstracts from outside the two ranges.
        is_new = torch.as_tensor(row['date'] >= self.second_range[0])
        female_ratio = torch.as_tensor(row['female'] / row['num_participants'])

        return {'text': text, 'is_new': is_new, 'female_ratio': female_ratio}
