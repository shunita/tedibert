import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

from contra.common.utils import mean_pooling
from contra.constants import DATA_PATH, SAVE_PATH
from transformers import AutoTokenizer, AutoModel
from contra.models.w2v_on_years import read_w2v_model
from contra.utils.text_utils import TextUtils
from contra.utils.pubmed_utils import split_abstracts_to_sentences_df, load_aact_data


class PubMedModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.min_num_participants = hparams.min_num_participants
        self.first_time_range = (datetime(hparams.first_start_year, 1, 1), datetime(hparams.first_end_year, 12, 31))
        self.second_time_range = (datetime(hparams.second_start_year, 1, 1), datetime(hparams.second_end_year, 12, 31))
        self.test_start_year = hparams.test_start_year
        self.test_end_year = hparams.test_end_year
        self.test_fname = hparams.test_pairs_file
        if self.test_fname is not None:
            self.test_fname = os.path.join(SAVE_PATH, self.test_fname)
        self.train_test_split = hparams.train_test_split
        self.pubmed_version = hparams.pubmed_version
        self.emb_algorithm = hparams.emb_algorithm
        self.batch_size = hparams.batch_size
        self.df = None
        self.train, self.test, self.val = None, None, None

    def prepare_data(self):
        # This is a df containing a row for each abstract.
        df = load_aact_data(self.pubmed_version, year_range=None)
        df = df.dropna(subset=['date', 'male', 'female'], axis=0)
        df['date'] = df['date'].map(lambda dt: datetime.strptime(dt, '%Y-%m-%d'))
        df['num_participants'] = df['female'] + df['male']
        df = df[df['num_participants'] >= self.min_num_participants]
        self.df = df

    def setup(self, stage=None):
        old_df = self.df[(self.df['date'] >= self.first_time_range[0]) & (self.df['date'] <= self.first_time_range[1])]
        new_df = self.df[(self.df['date'] >= self.second_time_range[0]) & (self.df['date'] <= self.second_time_range[1])]

        # We split to train and test while we still have the whole abstract.
        old_train_df, old_val_df = train_test_split(old_df, test_size=0.2)
        new_train_df, new_val_df = train_test_split(new_df, test_size=0.2)
        train_df = pd.concat([new_train_df, old_train_df])
        val_df = pd.concat([new_val_df, old_val_df])
        # Transform the Dataframes to have a row for each sentence, and the details of the abstract it came from.
        keep_fields = ['date', 'year', 'female', 'male', 'num_participants']
        train_df = split_abstracts_to_sentences_df(train_df, keep=keep_fields)
        val_df = split_abstracts_to_sentences_df(val_df, keep=keep_fields)

        self.train = PubMedDataset(train_df, self.first_time_range, self.second_time_range)
        self.val = PubMedDataset(val_df, self.first_time_range, self.second_time_range)

        if self.emb_algorithm == 'w2v':
            self.test = CUIDataset(bert=None, test_start_year=self.test_start_year, test_end_year=self.test_end_year,
                                   frac=0.001, sample_type=1, top_percentile=0.5, semtypes=['dsyn'],
                                   read_from_file=self.test_fname)
        elif self.emb_algorithm == 'bert':
            bert_path = f'bert_tiny_uncased_{self.test_start_year}_{self.test_end_year}_v{self.pubmed_version}_epoch39'
            #bert_path = f'bert_base_cased_{self.test_start_year}_{self.test_end_year}_v{self.pubmed_version}_epoch39'
            self.test = CUIDataset(bert=os.path.join(SAVE_PATH, bert_path),
                                   bert_tokenizer=self.hparams.bert_tokenizer,
                                   test_start_year=self.test_start_year, test_end_year=self.test_end_year,
                                   frac=0.001, sample_type=1, top_percentile=0.5, semtypes=['dsyn'], 
                                   read_from_file=self.test_fname)
        else:
            print(f'Unsupported emb_algorithm: {self.emb_algorithm}')
        print(f'Loaded {len(train_df)} train samples and {len(val_df)} validation samples.\nLoaded {len(self.test)} cui pairs for test.')

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test, shuffle=False, batch_size=self.batch_size, num_workers=8)


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
        text = row['text']
        # at this point we don't have data from outside the two ranges.
        is_new = torch.as_tensor(row['date'] >= self.second_range[0])
        female_ratio = torch.as_tensor(row['female'] / row['num_participants'])
        return {'text': text, 'is_new': is_new, 'female_ratio': female_ratio}


class CUIDataset(Dataset):
    def __init__(self, bert='google/bert_uncased_L-2_H-128_A-2', bert_tokenizer=None,
                 test_start_year=2018, test_end_year=2018,
                 read_w2v_params={},
                 top_percentile=0.01, semtypes=None, frac=1., sample_type=0, 
                 filter_by_models=(),
                 read_from_file=None):
        """
        This Dataset handles the CUI pairs and their similarity
        Args:
            bert: the bert path to use. If None, will use w2v.
            test_years: (start_year, end_year) - the range of years on which the w2v or bert model was trained.
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
            self.emb_algorithm = 'bert'
            tokenizer_name = bert
            if bert_tokenizer is not None:
                tokenizer_name = bert_tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.bert_model = AutoModel.from_pretrained(bert)
            if bert == 'google/bert_uncased_L-2_H-128_A-2':
                self.model_desc = 'tinybert'
            else:
                self.model_desc = 'bert'
        elif test_start_year is not None and test_end_year is not None:
            self.emb_algorithm = 'w2v'
            self.tokenizer = TextUtils()
            self.w2v_model = read_w2v_model(test_start_year, test_end_year, **read_w2v_params)
            self.model_desc = 'w2v'
        else:
            print("CUIDataset got no model to work with: both bert and w2v_years are None.")
            sys.exit()

        # Note: this table was generated based on pubmed 2018.
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
            CUI_embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask']).detach().numpy()
        else:  # test_years is not None
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
            
            print(f"Filtering CUIs by main model: {test_start_year}_{test_end_year} model")
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

        self.similarity_df.to_csv(
            os.path.join(SAVE_PATH,
                         f'test_similarities_CUI_names_{self.model_desc}_{test_start_year}_{test_end_year}.csv'))

    def __len__(self):
        return len(self.similarity_df)

    def __getitem__(self, index):
        row = self.similarity_df.iloc[index]
        return {'CUI1': row['CUI1'], 'CUI2': row['CUI2'], 'true_similarity': row['similarity']}
