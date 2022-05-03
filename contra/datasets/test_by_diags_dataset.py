import os
from collections import defaultdict

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader

from contra.utils.code_mapper import read_emb, CodeMapper
from contra.utils.pubmed_utils import clean_abstracts
from contra.utils.text_utils import TextUtils
from contra.constants import DATA_PATH, READMIT_TEST_PATH, LOS_TEST_PATH_V4
from ast import literal_eval
import numpy as np


def calculate_idf(list_of_lists):
    counter = defaultdict(int)
    N = len(list_of_lists)
    for l in list_of_lists:
        for item in set(l):
            counter[item] += 1
    # for unknown terms, we don't want to affect their weight.
    res = defaultdict(lambda: 1)
    for code in counter:
        res[code] = np.log(N/counter[code])
    return res


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


class DiagsModuleBase(pl.LightningDataModule):
    def __init__(self,
                 diag_dict,
                 procedures_dict,
                 data_file,
                 batch_size=16,
                 upsample_female_file=None):
        super().__init__()
        self.batch_size = batch_size
        self.upsample_female = upsample_female_file
        print(f"Reading data from {data_file}")
        self.data = pd.read_csv(data_file, index_col=0)
        self.diag_dict = diag_dict
        self.procedure_dict = procedures_dict
        self.removed_diags = set()
        self.total_diags = set()
        self.removed_procs = set()
        self.total_procs = set()
        self.diag_idf = defaultdict(lambda: 1)
        self.categorical_features = []

    def filter_code_list(self, icd9_codes_list_as_text, lookup_dict, name='diags', remove_if_half_missing=False):
        icd9_codes_list_as_text = clean_nans(icd9_codes_list_as_text)
        if pd.isna(icd9_codes_list_as_text):
            return '[]'
        try:
            codes = literal_eval(icd9_codes_list_as_text)
        except:
            print(f"could not eval: {icd9_codes_list_as_text}")
            return '[]'
        new_codes = []
        for code in codes:
            if name == 'diags':
                self.total_diags.add(code)
            elif name == 'procs':
                self.total_procs.add(code)

            if code in lookup_dict:
                new_codes.append(code)
            else:
                if name == 'diags':
                    self.removed_diags.add(code)
                elif name == 'procs':
                    self.removed_procs.add(code)
        return str(new_codes)

    def prepare_data(self):
        # Assumes that all NAs in important fields were dropped.
        self.handle_categorical_features()
        self.data = self.data[self.data.AGE >= 18]  # don't include newborns and children
        self.data.DIAGS = self.data.DIAGS.apply(lambda x: self.filter_code_list(x, self.diag_dict, 'diags', remove_if_half_missing=False))
        records_before = len(self.data)
        self.data = self.data[self.data.DIAGS != '[]']
        print(f'After removal of unknown/unembeddable ICD9s and then empty records, {len(self.data)}/{records_before} records remain.')

        if 'PREV_DIAGS' in self.data.columns:
            print("Processing PREV_DIAGS")
            self.data.PREV_DIAGS = self.data.PREV_DIAGS.apply(clean_nans)
            self.data.PREV_DIAGS = self.data.PREV_DIAGS.apply(
                lambda x: self.filter_code_list(x, self.diag_dict, 'diags', remove_if_half_missing=False))
            self.data['NUM_PREV_DIAGS'] = self.data.PREV_DIAGS.apply(literal_eval).apply(len)
            # self.data = self.data[self.data.PREV_DIAGS != '[]']
            # TODO: what if we didn't remove cases without prev diags?
            # print(f"Working with PREV_DIAGS. Kept {len(self.data)} records who had PREV_DIAGS.")

        if 'DRUG' in self.data.columns:
            self.data['DRUG'] = self.data.DRUG.fillna('{}')

        if 'PROCEDURES' in self.data.columns:
            self.data['PROCEDURES'] = self.data.PROCEDURES.fillna('[]')
            self.data.PROCEDURES = self.data.PROCEDURES.apply(clean_nans)
            self.data.PROCEDURES = self.data.PROCEDURES.apply(
                lambda x: self.filter_code_list(x, self.procedure_dict, 'procs', remove_if_half_missing=False))

        print(f'Unknown diag codes: {len(self.removed_diags)}/{len(self.total_diags)}')
        print(f'Unknown procedure codes: {len(self.removed_procs)}/{len(self.total_procs)}')

        self.train_df = self.data[self.data['ASSIGNMENT'] == 'train']
        self.val_df = self.data[self.data['ASSIGNMENT'] == 'test']
        print(f'Divided to {len(self.train_df)} train, {len(self.val_df)} test.')
        if self.upsample_female is not None:
            print(f"before upsampling female patients, train contains {self.train_df.GENDER_F.sum()} female patients.")
            if os.path.exists(self.upsample_female):
                sample = pd.read_csv(self.upsample_female, index_col=0)
            else:  # resample
                train_fem = self.train_df[self.train_df.GENDER_F == 1]
                train_male = self.train_df[self.train_df.GENDER_M == 1]
                num_fem_subjects = train_fem.SUBJECT_ID.nunique()
                num_male_subjects = train_male.SUBJECT_ID.nunique()
                subjects_to_sample = num_male_subjects - num_fem_subjects
                sampled_subjects = np.random.choice(train_fem.SUBJECT_ID.unique(), size=subjects_to_sample, replace=False)
                sample = train_fem[train_fem.SUBJECT_ID.isin(sampled_subjects)]
                sample.to_csv(self.upsample_female)
            self.train_df = pd.concat([self.train_df, sample]).reset_index()  # The original index will still be under 'index'
            print(f'After upsampling female patients: {len(self.train_df)} train, {len(self.val_df)} test. Train contains {self.train_df.GENDER_F.sum()} female patients.')
        self.diag_idf = calculate_idf(self.train_df.DIAGS.apply(literal_eval).values)
        self.calculate_stats()

    def calculate_stats(self):
        pass

    def handle_categorical_features(self):
        # TODO: more refined handling of features like RELIGION and ETHNICITY?
        self.data = pd.get_dummies(self.data, columns=self.categorical_features)

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=4)


class LOSbyDiagsModule(DiagsModuleBase):
    def __init__(self, diag_dict, procedure_dict,
                 data_file=LOS_TEST_PATH_V4,
                 batch_size=16, classification=False, upsample_female_file=None):
        super(LOSbyDiagsModule, self).__init__(diag_dict, procedure_dict, data_file, batch_size, upsample_female_file)
        self.data = self.data.rename({'ICD9_CODE': 'DIAGS'}, axis=1)
        self.data = self.data.dropna(subset=['DIAGS', 'LOS'])
        self.data = self.data[self.data.DIAGS != '[nan]']
        self.classification = classification
        # only for regression - remove the top 5% because they are outliers.
        if not self.classification:
            # remove top 5%
            longest_LOS = self.data.LOS.quantile(0.95)
            before = len(self.data)
            self.data = self.data[self.data.LOS <= longest_LOS]
            print(f"Removed records with LOS above {longest_LOS}, remaining records: {len(self.data)/before}")
        self.categorical_features = ['GENDER']   +\
                                    ['ADMISSION_TYPE', 'ADMISSION_LOCATION', 'INSURANCE', 'RELIGION',
                                     'ETHNICITY', 'MARITAL_STATUS']

    def calculate_stats(self):
        c = self.train_df['LOS'].mean()
        ypred = np.ones(len(self.val_df)) * c
        ret = mean_squared_error(self.val_df['LOS'], ypred, squared=False)
        print(f"Test RMSE of predicting the train mean ({c}): {ret}")
        return ret

    def setup(self, stage=None):
        self.train = LOSbyDiagsDataset(self.train_df, self.diag_idf, self.categorical_features)
        self.val = LOSbyDiagsDataset(self.val_df, self.diag_idf, self.categorical_features)


class ReadmissionbyDiagsModule(DiagsModuleBase):
    def __init__(self, diag_dict,
                 procedure_dict,
                 data_file=READMIT_TEST_PATH,
                 batch_size=16,
                 upsample_female_file=None):
        super(ReadmissionbyDiagsModule, self).__init__(diag_dict, procedure_dict, data_file, batch_size, upsample_female_file)
        self.data = self.data.dropna(subset=['DIAGS', 'READMISSION'])
        self.data = self.data[self.data['READMISSION'] != 2]  # Remove patients who died in the hospital
        # print("train: female: {}")
        self.categorical_features = ['GENDER'] + \
                                    ['ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION',
                                     'INSURANCE', 'RELIGION', 'ETHNICITY', 'MARITAL_STATUS'] #+\
                                    # ['Glascow coma scale eye opening first', 'Glascow coma scale eye opening last',
                                    #  'Glascow coma scale motor response first', 'Glascow coma scale motor response last',
                                    #  'Glascow coma scale verbal response first', 'Glascow coma scale verbal response last'
                                    #  ]
        # can add FIRST/LAST CAREUNIT/WARDID, LOS (in the ICU)

    # def prepare_data(self):
    #     # filter the procedures field
    #     self.data.PROCEDURES = self.data.PROCEDURES.apply(lambda x: self.filter_code_list(x, self.procedures_dict))
    #     # filter diags field, age, and split to train and test is implemented in the Base class.
    #     super(ReadmissionbyDiagsModule, self).prepare_data()

    def setup(self, stage=None):
        self.train = ReadmitbyDiagsDataset(self.train_df, self.diag_idf, self.categorical_features)
        self.val = ReadmitbyDiagsDataset(self.val_df, self.diag_idf, self.categorical_features)


class DiagsDatasetBase(Dataset):
    def __init__(self, df, diag_idf, categorical_feature_prefixes, non_cat_features):
        self.df = df
        self.diag_idf = diag_idf
        self.categorical_features = [c for c in self.df.columns
                                     if self.categorical_column(c, categorical_feature_prefixes)]
        self.non_cat_features = non_cat_features
        print(
            f"DiagsDataset: Additional features dimension: {len(self.categorical_features) + len(self.non_cat_features)}")

    def __len__(self):
        return len(self.df)

    def categorical_column(self, col_name, prefixes):
        for prefix in prefixes:
            if col_name.startswith(prefix):
                return True
        return False

    def __getitem__(self, index):
        raise('Not Implemented: getitem in DiagsDatasetBase')


class LOSbyDiagsDataset(DiagsDatasetBase):
    def __init__(self, df, diag_idf, categorical_feature_prefixes, classification=False):
        super(LOSbyDiagsDataset, self).__init__(
            df, diag_idf, categorical_feature_prefixes,
            non_cat_features=['AGE'] + \
                             ['NUM_PREV_ADMIS', 'DAYS_SINCE_LAST_ADMIS', 'NUM_PREV_PROCS', 'NUM_PREV_DIAGS']
        )
        self.classification = classification

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        row = self.df.iloc[index]
        label = row['LOS']
        if self.classification:
            label = (label > 5)
        ret = {'diags': row['DIAGS'],
               'los': label,
               'diag_idfs': str([self.diag_idf[d] for d in literal_eval(row['DIAGS'])]),
               'sample_id': row.name}
        if 'PREV_DIAGS' in row:
            ret['prev_diags'] = row['PREV_DIAGS']
        # handle categoricals
        ret['additionals'] = torch.Tensor(row[self.categorical_features+self.non_cat_features].values.astype(float))
        return ret


class ReadmitbyDiagsDataset(DiagsDatasetBase):
    def __init__(self, df, diag_idf, categorical_feature_prefixes):
        non_cat_features = ['AGE'] + \
                           ['NUM_PREV_ADMIS', 'DAYS_SINCE_LAST_ADMIS', 'NUM_PREV_PROCS', 'NUM_PREV_DIAGS']  # +\
                           # ['Glascow coma scale total avg', 'Glascow coma scale total first',
                           #  'Glascow coma scale total last', 'Capillary refill rate avg',
                           #  'Capillary refill rate first', 'Capillary refill rate last',
                           #  'Diastolic blood pressure avg', 'Diastolic blood pressure first',
                           #  'Diastolic blood pressure last',
                           #  'Fraction inspired oxygen avg',
                           #  'Fraction inspired oxygen first', 'Fraction inspired oxygen last',
                           #  'Glucose avg',
                           #  'Glucose first', 'Glucose last', 'Heart Rate avg', 'Heart Rate first',
                           #  'Heart Rate last', 'Height avg', 'Height first', 'Height last',
                           #  'Mean blood pressure avg', 'Mean blood pressure first',
                           #  'Mean blood pressure last',
                           #  'Oxygen saturation avg',
                           #  'Oxygen saturation first', 'Oxygen saturation last',
                           #  'Respiratory rate avg', 'Respiratory rate first',
                           #  'Respiratory rate last',
                           #  'Systolic blood pressure avg',
                           #  'Systolic blood pressure first', 'Systolic blood pressure last',
                           #  'Temperature avg', 'Temperature first', 'Temperature last',
                           #  'Weight avg', 'Weight first', 'Weight last',
                           #  'pH avg', 'pH first', 'pH last',
                           #  ]
        super(ReadmitbyDiagsDataset, self).__init__(
            df, diag_idf, categorical_feature_prefixes,
            non_cat_features=non_cat_features)
        self.df = df
        self.diag_idf = diag_idf
        self.categorical_features = [c for c in self.df.columns
                                     if self.categorical_column(c, categorical_feature_prefixes)]

        self.df = self.df.fillna(0)
        # self.non_cat_features = non_cat_features
        print(
            f"ReadmitbyDiagsDataset: Additional features dimension: {len(self.categorical_features) + len(self.non_cat_features)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        row = self.df.iloc[index]
        ret = {'diags': row['DIAGS'],
               'diag_idfs': str([self.diag_idf[d] for d in literal_eval(row['DIAGS'])]),
               'drugs': row['DRUG'],
               'prev_diags': row['PREV_DIAGS'],
               'procedures': row['PROCEDURES'],
               'readmitted': row['READMISSION'],
               'sample_id': row.name,
               'additionals': torch.Tensor(row[self.categorical_features + self.non_cat_features].values.astype(float))}
        return ret


class DiagsEmbModuleBase(DiagsModuleBase):
    def __init__(self,
                 emb_files,
                 data_file,
                 batch_size=16):
        super(DiagsEmbModuleBase, self).__init__(None, None, data_file, batch_size)
        self.embs = [read_emb(emb_file) for emb_file in emb_files]
        self.code_mapper = CodeMapper()

    def check_cui_in_embs(self, cui):
        return all([cui in emb for emb in self.embs])

    def filter_code_list(self, icd9_codes_list_as_text, lookup_dict=None, name='diags', remove_if_half_missing=True):
        icd9_codes_list_as_text = clean_nans(icd9_codes_list_as_text)
        codes = literal_eval(icd9_codes_list_as_text)
        new_codes = []
        for code in codes:
            if name == 'diags':
                self.total_diags.add(code)
            elif name == 'procs':
                self.total_procs.add(code)

            cuis = self.code_mapper[code]
            possible_embs_for_single_diag = []
            for cui in cuis:
                if self.check_cui_in_embs(cui):
                    possible_embs_for_single_diag.append(cui)
            if len(possible_embs_for_single_diag) == 0:
                if name == 'diags':
                    self.removed_diags.add(code)
                elif name == 'procs':
                    self.removed_procs.add(code)
            elif len(possible_embs_for_single_diag) == 1:
                new_codes.append(possible_embs_for_single_diag[0])
            else:
                counts = [self.code_mapper.get_cui_appearances(cui) for cui in possible_embs_for_single_diag]
                # take the cui that is most common in the dataset
                new_codes.append(possible_embs_for_single_diag[np.argmax(counts)])
        # if the mapping made us lose half of the diagnoses,
        if remove_if_half_missing and len(set(new_codes)) < 0.5*len(set(codes)):
            return '[]'
        return str(new_codes)

    # def prepare_data(self):
    #     # Assumes that all NAs in important fields were dropped.
    #     self.data = self.data[self.data.AGE >= 18]  # don't include newborns, but also not children
    #     self.data['DIAGS'] = self.data.DIAGS.apply(self.filter_code_list)
    #     before = len(self.data)
    #     self.data = self.data.dropna(subset=['DIAGS'], axis=0)
    #     print(f"Kept {len(self.data)}/{before} rows after mapping to embeddable CUIs.")
    #     print(f'Unknown codes: {len(self.removed_diags)}')
    #     if 'DRUG' in self.data.columns:
    #         self.data['DRUG'] = self.data.DRUG.fillna('{}')
    #     if 'PROCEDURES' in self.data.columns:
    #         self.data['PROCEDURES'] = self.data.DRUG.fillna('[]')
    #     if 'PREV_DIAGS' in self.data.columns:
    #         self.data['PREV_DIAGS'] = self.data['PREV_DIAGS'].apply(self.filter_code_list)
    #     self.train_df = self.data[self.data['ASSIGNMENT'] == 'train']
    #     self.val_df = self.data[self.data['ASSIGNMENT'] == 'test']
    #     print(f'Divided to {len(self.train_df)} train, {len(self.val_df)} test.')
    #     self.diag_idf = calculate_idf(self.train_df.DIAGS.apply(literal_eval).values)
    #     print(f"calculated diag idf: {list(self.diag_idf.items())[:5]}")


class LOSbyEmbDiagsModule(DiagsEmbModuleBase):
    def __init__(self, emb_files,
                 data_file=LOS_TEST_PATH_V4,
                 batch_size=16, classification=False):
        super(LOSbyEmbDiagsModule, self).__init__(emb_files, data_file, batch_size)
        self.data = self.data.rename({'ICD9_CODE': 'DIAGS'}, axis=1)
        self.data = self.data.dropna(subset=['DIAGS', 'LOS'])
        self.data = self.data[self.data.DIAGS != '[nan]']

        self.classification = classification
        if not self.classification:
            longest_LOS = self.data.LOS.quantile(0.95)
            before = len(self.data)
            self.data = self.data[self.data.LOS <= longest_LOS]
            print(f"Removed records with LOS above {longest_LOS}, remaining records: {len(self.data) / before}")
        self.categorical_features = ['GENDER', 'ETHNICITY',
                                     #'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'INSURANCE', 'RELIGION', 'MARITAL_STATUS'
                                     ]

    def calculate_stats(self):
        c = self.train_df['LOS'].mean()
        ypred = np.ones(len(self.val_df)) * c
        ret = mean_squared_error(self.val_df['LOS'], ypred, squared=False)
        print(f"Test RMSE of predicting the train mean ({c}): {ret}")
        return ret

    def setup(self, stage=None):
        self.train = LOSbyDiagsDataset(self.train_df, self.diag_idf, self.categorical_features, self.classification)
        self.val = LOSbyDiagsDataset(self.val_df, self.diag_idf, self.categorical_features, self.classification)


class ReadmissionbyEmbDiagsModule(DiagsEmbModuleBase):
    def __init__(self, emb_files,
                 data_file=READMIT_TEST_PATH,
                 batch_size=16):
        super(ReadmissionbyEmbDiagsModule, self).__init__(emb_files, data_file, batch_size)
        self.data = self.data.dropna(subset=['DIAGS', 'READMISSION'])
        self.data = self.data[self.data['READMISSION'] != 2]  # Remove patients who died in the hospital
        self.categorical_features = ['GENDER',
                                     'ETHNICITY',
                                     # 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION',
                                     # 'INSURANCE', 'RELIGION', 'MARITAL_STATUS'
                                     ]
        # can add FIRST/LAST CAREUNIT/WARDID, LOS (in the ICU)

    def setup(self, stage=None):
        self.train = ReadmitbyDiagsDataset(self.train_df, self.diag_idf, self.categorical_features)
        self.val = ReadmitbyDiagsDataset(self.val_df, self.diag_idf, self.categorical_features)
