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
from contra.constants import DATA_PATH, SAVE_PATH
from transformers import AutoTokenizer, AutoModel
from contra.models.w2v_on_years import read_w2v_model
from contra.utils.text_utils import TextUtils


class PubMedModule(pl.LightningDataModule):

    def __init__(self, min_num_participants=1,
                 first_year_range=(2010,2013),
                 second_year_range=(2018,2018),
                 train_test_split=0.8,
                 test_year_range=None,
                 test_fname=None):
        super().__init__()
        self.min_num_participants = min_num_participants
        self.first_time_range = (datetime(first_year_range[0], 1, 1), datetime(first_year_range[1], 12, 31))
        self.second_time_range = (datetime(second_year_range[0], 1, 1), datetime(second_year_range[1], 12, 31))
        self.test_year_range = test_year_range
        self.test_fname = test_fname
        self.train_test_split = train_test_split
        self.df = None
        self.train, self.test, self.val = None, None, None

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
        
        if self.test_fname is not None:
            fname = os.path.join(SAVE_PATH, self.test_fname)
        elif self.test_year_range is not None and self.test_year_range[0] is not None:
            testyear0, testyear1 = self.test_year_range
            fname = os.path.join(SAVE_PATH, 'test_similarities_CUI_names_{testyear0}_{testyear1}.csv')
        else:
            print("both test year and test file name were not provided.")
            sys.exit()
        self.test = CUIDataset(bert=None, w2v_years=self.test_year_range, frac=0.001, sample_type=1, top_percentile=0.5, semtypes=['dsyn'], 
                               read_from_file=fname)
        print(f'Loaded {len(train_df)} train samples and {len(val_df)} validation samples.\nLoaded {len(self.test)} cui pairs for test.')

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=32, num_workers=32)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=32, num_workers=32)

    def test_dataloader(self):
        return DataLoader(self.test, shuffle=False, batch_size=32, num_workers=32)


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
    def __init__(self, bert="google/bert_uncased_L-2_H-128_A-2", w2v_years=(2018, 2018), read_w2v_params={},
                 top_percentile=0.01, semtypes=None, frac=1., sample_type=0, 
                 filter_by_models=[],
                 read_from_file=None):
        """
        This Dataset handles the CUI pairs and their similarity
        Args:
            bert: the bert path to use. If None, will use w2v.
            w2v_years: (start_year, end_year) - the range of years on which a w2v model was trained.
            top_percentile: from what percentile of apperance to filter
            semtypes: str or List[str] representing the wanted semtypes. If None then uses all
            frac: a number in the range [0,1], representing the fraction of pairs to sample.
            sample_type: in order to subsample relevant pairs, there are different methods for subsampling:
                            0 - take all - if selected, ignores `frac`
                            1 - random sample
                            2 - top similar
                            3 - uniform distribution of similarities
        """
        if read_from_file is not None:
            self.similarity_df = pd.read_csv(read_from_file, index_col=0)
            print(f"read {len(self.similarity_df)} CUI pairs from {read_from_file}.")
            return
        
        self.top_percentile = top_percentile
        self.semtypes = semtypes
        self.frac = frac
        self.sample_type = sample_type
        if bert is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(bert)
            self.bert_model = AutoModel.from_pretrained(bert)
        elif w2v_years is not None:
            self.tokenizer = TextUtils()
            self.w2v_model = read_w2v_model(w2v_years[0], w2v_years[1], **read_w2v_params)
        else:
            print("CUIDataset got no model to work with: both bert and w2v_years are None.")
            sys.exit()

        df = pd.read_csv(os.path.join(DATA_PATH, 'cui_table_for_cui2vec_with_abstract_counts.csv'))
        all_cuis = len(df)
        if self.semtypes is not None:
            if not isinstance(self.semtypes, list):
                self.semtypes = [self.semtypes]
            df = df[df['semtypes'].map(lambda sems: np.any([sem in self.semtypes for sem in eval(sems)]))]
        semtype_match = len(df)
        df = df[df['abstracts'] >= df['abstracts'].quantile(1 - self.top_percentile)]
        percentile_match = len(df)
        print(f"read {all_cuis} CUIs.\nFiltered to {semtype_match} by semptypes: {self.semtypes}.\nKept {self.top_percentile} with the most abstract appearances: {percentile_match} CUIs.")

        CUI_names = df['name'].tolist()
        if bert is not None:
            inputs = self.tokenizer(CUI_names, padding=True, return_tensors="pt")
            outputs = self.bert_model(**inputs)
            CUI_embeddings = mean_pooling(outputs, inputs['attention_mask']).detach().numpy()
        else:  # w2v_years is not None
            tokenized_names = [self.tokenizer.word_tokenize_abstract(name) for name in CUI_names]
            # Filter the CUIs according to the models we want to compare.
            # CUIs in the final test set should have embeddings in all compared models.
            for (first_year, last_year) in filter_by_models:
                print(f"Filtering CUIs by {first_year}_{last_year} model")
                w2v_model = read_w2v_model(first_year, last_year)
                embs = w2v_model.embed_batch(tokenized_names)
                got_emb = torch.count_nonzero(embs, dim=1) > 0
                before = len(CUI_names)
                CUI_names = [name for i, name in enumerate(CUI_names) if got_emb[i]]
                print(f"Keeping {len(CUI_names)}/{before} CUIs.")
            
            print(f"Filtering CUIs by main model: {first_year}_{last_year} model")
            CUI_embeddings = self.w2v_model.embed_batch(tokenized_names) 
            got_embedding = torch.count_nonzero(CUI_embeddings, dim=1) > 0
            CUI_embeddings = CUI_embeddings[got_embedding]
            before = len(CUI_names)
            CUI_names = [name for i, name in enumerate(CUI_names) if got_embedding[i]]
            print(f"Keeping {len(CUI_names)}/{before} CUIs.")

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
        self.similarity_df.to_csv(os.path.join(SAVE_PATH, f"test_similarities_CUI_names_{w2v_years[0]}_{w2v_years[1]}.csv"))

    def __len__(self):
        return len(self.similarity_df)

    def __getitem__(self, index):
        row = self.similarity_df.iloc[index]
        return {'CUI1': row['CUI1'], 'CUI2': row['CUI2'], 'true_similarity': row['similarity']}
