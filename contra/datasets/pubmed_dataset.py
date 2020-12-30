import os
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

from contra.common.utils import get_bert_model, mean_pooling
from contra.constants import DATA_PATH
from transformers import AutoTokenizer, AutoModel


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
        self.test = CUIDataset(frac=0.001, sample_type=1)
        print("Loaded {} train samples and {} validation samples".format(len(train_df), len(val_df)))

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=32, num_workers=32)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=32, num_workers=32)

    def test_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=32, num_workers=32)


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

class CUIDataset(Dataset):
    def __init__(self, bert="google/bert_uncased_L-2_H-128_A-2", top_percentile=0.01, semtypes=None, frac=1., sample_type=0):
        """
        This Dataset handles the CUI pairs and their similarity
        Args:
            bert: the bert path to use
            top_percentile: from what percentile of apperance to filter
            semtypes: str or List[str] representing the wanted semtypes. If None then uses all
            frac: a number in the range [0,1], representing the fraction of pairs to sample.
            sample_type: in order to subsample relevant pairs, there are different methods for subsampling:
                            0 - take all - if selected, ignores `frac`
                            1 - random sample
                            2 - top similar
                            3 - uniform distribution of similarities
        """
        self.top_percentile = top_percentile
        self.semtypes = semtypes
        self.frac = frac
        self.sample_type = sample_type

        self.tokenizer = AutoTokenizer.from_pretrained(bert)
        self.bert_model = AutoModel.from_pretrained(bert)

        df = pd.read_csv(os.path.join(DATA_PATH, 'cui_table_for_cui2vec_with_abstract_counts.csv'))
        if self.semtypes is not None:
            if not isinstance(self.semtypes, list):
                self.semtypes = [self.semtypes]
            df = df[df['semtypes'].map(lambda sems: np.any([sem in self.semtypes for sem in eval(sems)]))]
        df = df[df['abstracts'] >= df['abstracts'].quantile(1 - self.top_percentile)]

        CUI_names = df['name'].tolist()
        inputs = self.tokenizer(CUI_names, padding=True, return_tensors="pt")
        outputs = self.bert_model(**inputs)
        CUI_embeddings = mean_pooling(outputs, inputs['attention_mask']).detach().numpy()
        similarity = cosine_similarity(CUI_embeddings, CUI_embeddings).flatten()
        pairs = list(product(CUI_names, CUI_names))

        self.similarity_df = pd.DataFrame()
        self.similarity_df[['CUI1', 'CUI2']] = pairs
        self.similarity_df['similarity'] = similarity

        if self.sample_type == 1:
            self.similarity_df = self.similarity_df.sample(frac=self.frac)
        elif self.sample_type == 2:
            num_samples = int(self.frac*len(similarity))
            self.similarity_df = self.similarity_df.nlargest(num_samples, 'similarity')
        elif self.sample_type == 3:
            step_size = int(1/self.frac)
            self.similarity_df = self.similarity_df.sort_values('similarity', ascending=False).iloc[::step_size]

    def __len__(self):
        return len(self.similarity_df)

    def __getitem__(self, index):
        row = self.similarity_df.iloc[index]
        return {'CUI1': row['CUI1'], 'CUI2': row['CUI1'], 'true_similarity': row['similarity']}
