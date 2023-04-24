import os
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from ast import literal_eval
import pickle

CLINICAL_NOTES_PICKLES_PATH = os.path.expanduser('~/mimic3/clinical-notes-diagnosis-dl-nlp/code/data')

def clean_nans(string_rep_of_seq):
    return string_rep_of_seq.replace('nan, ', '').replace(', nan', '').replace('nan', '')


def clean_ethnicity_field(ethnicity_str):
    s = ethnicity_str.split(" - ")[0]
    if s == 'HISPANIC OR LATINO':
        return 'HISPANIC/LATINO'
    if s.startswith('BLACK'):
        return 'BLACK'
    if s in ['UNABLE TO OBTAIN', 'PATIENT DECLINED TO ANSWER', 'UNKNOWN/NOT SPECIFIED']:
        return 'UNKNOWN'
    return s


def get_labels(topX, use_icd9_cat):
    prefix = 'ICD9CAT_' if use_icd9_cat else 'ICD9CODES_'
    suffix = 'TOP10.p' if topX == 10 else 'TOP50.p'
    pickle_path = os.path.join(CLINICAL_NOTES_PICKLES_PATH, prefix + suffix)
    labels = pickle.load(open(pickle_path, 'rb'))
    return labels


def get_focused(row):
    if len(row['diag_sents']) > 0:
        return row['diag_sents']
    return row['sentences']


class ClinicalNotesModule(pl.LightningDataModule):
    def __init__(self,
                 data_file,
                 topX=10,  # or 50
                 use_icd9_cat=False,  # code or category
                 batch_size=16,
                 CV_fold=None,
                 focus_on_diags=False):
        super().__init__()
        self.batch_size = batch_size
        print(f"Reading data from {data_file}")
        # print("*********Sampling for faster debugging.. Remove this for actual results!*********")
        self.data = pd.read_csv(data_file, index_col=0,
                                # nrows=1000
                                )
        self.labels = get_labels(topX, use_icd9_cat)
        # self.labels = ['4280', '4019', '42731', '41401', '5849', '25000', '2724', '51881', '5990', '53081']
        # only keep records that have at least one positive label
        before = len(self.data)
        self.data = self.data[self.data[self.labels].sum(axis=1) > 0]
        print(f"removed {before-len(self.data)} records without any positive label.")
        self.CV_fold = CV_fold
        self.focus_on_diags = focus_on_diags
        # focus on sentences that contain the word "diagnos".
        if self.focus_on_diags:
            print("Focus on diags is True - removing sentences that don't have 'diagnos' in them")
            self.data['diag_sents'] = self.data['sentences'].apply(lambda x: [s for s in x if "diagnos" in s.lower()])
            self.data['sentences'] = self.data.apply(get_focused, axis=1)

    def prepare_data(self):
        assignment_field = 'ASSIGNMENT'
        if self.CV_fold is not None:
            assignment_field = f'ASSIGNMENT_{self.CV_fold}'
        self.train_df = self.data[self.data[assignment_field] == 'train']
        self.val_df = self.data[self.data[assignment_field] == 'val']
        self.test_df = self.data[self.data[assignment_field] == 'test']
        print(f'Divided to {len(self.train_df)} train, {len(self.val_df)} val, {len(self.test_df)} test.')
        # TODO: upsample?

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, shuffle=False, batch_size=self.batch_size, num_workers=4)

    def setup(self, stage=None):
        self.train = NotesAndDiagsDataset(self.train_df, self.labels)
        self.val = NotesAndDiagsDataset(self.val_df, self.labels)
        self.test = NotesAndDiagsDataset(self.test_df, self.labels)


class NotesAndDiagsDataset(Dataset):
    def __init__(self, df, label_fields):
        self.df = df
        self.label_fields = label_fields

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        row = self.df.iloc[index]
        ret = {'sentences': row['sentences'],
               'hadm_id': row['HADM_ID'],
               'labels': torch.Tensor(row[self.label_fields].values.astype(float)),
               'gender': row['GENDER']}
        return ret