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

from contra.experimental.exp_utils import fill_binary_year_label
from contra.models.w2v_on_years import read_w2v_model
from contra.utils.text_utils import TextUtils
from contra.utils.pubmed_utils import split_abstracts_to_sentences_df, load_aact_data, clean_abstracts


class PubMedModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.serve_type = hparams.serve_type
        self.min_num_participants = hparams.min_num_participants
        self.first_time_range = (datetime(hparams.first_start_year, 1, 1), datetime(hparams.first_end_year, 12, 31))
        self.second_time_range = (datetime(hparams.second_start_year, 1, 1), datetime(hparams.second_end_year, 12, 31))
        self.test_start_year = hparams.test_start_year
        self.test_end_year = hparams.test_end_year
        self.test_fname = hparams.test_pairs_file
        if self.test_fname is not None:
            self.test_fname = os.path.join(SAVE_PATH, self.test_fname)
        self.test_size = hparams.test_size
        self.pubmed_version = hparams.pubmed_version
        self.emb_algorithm = hparams.emb_algorithm
        self.batch_size = hparams.batch_size
        self.reassign = False
        self.df = None
        self.train_df, self.val_df = None, None
        self.train, self.test, self.val = None, None, None

    def prepare_data(self):
        if self.train_df is not None:
            return
        if self.reassign:
            # This is a df containing a row for each abstract.
            df = load_aact_data(self.pubmed_version, year_range=None, sample=self.hparams.debug)
            df = df.dropna(subset=['date', 'male', 'female'], axis=0)
            df['date'] = df['date'].map(lambda dt: datetime.strptime(dt, '%Y-%m-%d'))
            df['num_participants'] = df['female'] + df['male']
            df = df[df['num_participants'] >= self.min_num_participants]
            self.df = df

            old_df = self.df[(self.df['date'] >= self.first_time_range[0]) & (self.df['date'] <= self.first_time_range[1])]
            new_df = self.df[
                (self.df['date'] >= self.second_time_range[0]) & (self.df['date'] <= self.second_time_range[1])]
            tu = TextUtils()
            df['sentences'] = df['title_and_abstract'].apply(tu.split_abstract_to_sentences)
            # We split to train and test while we still have the whole abstract.
            old_train_df, old_val_df = train_test_split(old_df, test_size=self.test_size)
            new_train_df, new_val_df = train_test_split(new_df, test_size=self.test_size)
            train_df = pd.concat([new_train_df, old_train_df])
            val_df = pd.concat([new_val_df, old_val_df])
        else:
            df = pd.read_csv(os.path.join(DATA_PATH, 'pubmed2020_assigned.csv'), index_col=0)
            tu = TextUtils()
            df['sentences'] = df['title_and_abstract'].apply(tu.split_abstract_to_sentences)
            if self.hparams.debug:
                df = df.sample(1000)
            train_df = df[df['assignment'] == 0].copy()
            val_df = df[df['assignment'] == 1].copy()
            print(f"Read from pre-assigned train, test file. Read: {len(train_df)} train, {len(val_df)} test.")

        train_df = clean_abstracts(train_df)
        val_df = clean_abstracts(val_df)

        if self.serve_type == 0:  # Full abstract
            self.train_df = train_df.rename({'title_and_abstract': 'text'}, axis=1)
            self.val_df = val_df.rename({'title_and_abstract': 'text'}, axis=1)
        elif self.serve_type > 1:  # single sentence or three sentences
            # Transform the Dataframes to have a row for each sentence, and the details of the abstract it came from.
            keep_fields = ['date', 'year', 'female', 'male', 'num_participants', 'title_and_abstract']
            split_params = {'text_field': 'title_and_abstract',
                            'keep': keep_fields,
                            'overlap': self.hparams.overlap_sentences}
            print(f"Before splitting to (3?) sentences, train: {len(train_df)} val: {len(val_df)}")
            self.train_df = split_abstracts_to_sentences_df(train_df, **split_params)
            self.val_df = split_abstracts_to_sentences_df(val_df, **split_params)
        print(f'Serve type: {self.serve_type}, overlap: {self.hparams.overlap_sentences}')
        print(f'Loaded {len(self.train_df)} train samples and {len(self.val_df)} validation samples.')

    def setup(self, stage=None):
        self.train = PubMedDataset(self.train_df, self.first_time_range, self.second_time_range)
        self.val = PubMedDataset(self.val_df, self.first_time_range, self.second_time_range)
        if self.emb_algorithm == 'w2v':
            self.test = CUIDataset(bert=None, test_start_year=self.test_start_year, test_end_year=self.test_end_year,
                                   frac=0.001, sample_type=1, top_percentile=0.5, semtypes=['dsyn'],
                                   read_from_file=self.test_fname)
        elif self.emb_algorithm == 'bert':
            bert_path = f'bert_tiny_uncased_{self.test_start_year}_{self.test_end_year}_v{self.pubmed_version}_epoch39'
            #bert_path = f'bert_tiny_uncased_2020_2020_v{self.pubmed_version}_epoch39'
            #bert_path = f'bert_base_cased_{self.test_start_year}_{self.test_end_year}_v{self.pubmed_version}_epoch39'
            self.test = CUIDataset(bert=os.path.join(SAVE_PATH, bert_path),
                                   bert_tokenizer=self.hparams.bert_tokenizer,
                                   test_start_year=self.test_start_year, test_end_year=self.test_end_year,
                                   frac=0.001, sample_type=1, top_percentile=0.5, semtypes=['dsyn'], 
                                   read_from_file=self.test_fname)
        else:
            print(f'Unsupported emb_algorithm: {self.emb_algorithm}')
        print(f'Loaded {len(self.test)} cui pairs for test.')

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
        self.df = fill_binary_year_label(self.df, self.first_range, self.second_range)
        self.tu = TextUtils()

    def __len__(self):
        return len(self.df)

    def create_text_field(self, sent_list):
        sent_list_filtered_by_words = [' '.join(self.tu.word_tokenize(sent)) for sent in sent_list]
        return '<BREAK>'.join(sent_list_filtered_by_words)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        row = self.df.iloc[index]
        # text = row['text']
        text = self.create_text_field(row['sentences'])
        female_ratio = torch.as_tensor(row['female'] / row['num_participants'])
        return {'text': text, 'is_new': row['label'], 'female_ratio': female_ratio}


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
