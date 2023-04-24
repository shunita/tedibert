import sys

sys.path.append('/home/shunita/fairemb/')
import os
import pandas as pd
import numpy as np
import scipy.stats as st
import random
import torch
from torch import nn
import pytz
import wandb
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from datetime import datetime
from ast import literal_eval
from collections import defaultdict
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer, AutoModel
from contra.constants import SAVE_PATH, LOG_PATH, DATA_PATH, LOS_TEST_PATH_V4
from contra.models.Transformer1D import Encoder1DLayer
from contra.datasets.test_by_diags_dataset import ReadmissionbyDiagsModule, ReadmissionbyEmbDiagsModule, clean_nans
from contra.tests.bert_on_diags_base import BertOnDiagsBase, EmbOnDiagsBase
from contra.constants import READMIT_TEST_PATH
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from contra.utils.delong_auc import delong_roc_test
from contra.tests.descs_and_models import DESCS_AND_MODELS, cui_embeddings

MIMIC3_CUSTOM_TASKS_PATH = '/home/shunita/mimic3/custom_tasks/data'
MIMIC_PATH = "/home/shunita/mimic3/physionet.org/files/mimiciii/1.4/"

DESCS_AND_MODELS = DESCS_AND_MODELS

# LR = 0.87 * 1e-5
LR = 1e-5
# LR = 1e-3
# used for most experiments
BATCH_SIZE = 64

# used with bert base models
# BATCH_SIZE = 16

# used with clinical BERT
# BATCH_SIZE = 2

RUN_MODEL_INDEX = 5
USE_EMB = True
# UPSAMPLE_FEMALE = 'data/readmission_by_diags_female_sample.csv'
# UPSAMPLE_FEMALE = 'data/readmission_by_diags_sampled_medgan.csv'
# UPSAMPLE_FEMALE = 'data/readmission_by_diags_with_smote.csv'
UPSAMPLE_FEMALE = None
DOWNSAMPLE_MALE = True
MAX_EPOCHS = 4  # 4
CROSS_VAL = None  # number (10) or None
# ADDITIONALS = 3 # age + gender
# ADDITIONALS = 0
ADDITIONALS = 1 # age

# setting random seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Note: edited /home/shunita/miniconda3/envs/fairemb1/lib/python3.6/site-packages/pytorch_lightning/utilities/data.py to get rid of noisy warning


def desc_to_name(desc):
    return f'{desc}_cls_diags_drugs_lstm_2L'

def print_metrics(ytrue, ypred):
    auc = roc_auc_score(ytrue, ypred)
    acc = accuracy_score(ytrue, ypred.round())
    print(f"AUC: {auc}, Accuracy: {acc}")
    return auc, acc


def print_aucs_for_readmission(result_df, name=None):
    test_df = pd.read_csv(READMIT_TEST_PATH, index_col=0)
    test_df = test_df.merge(result_df, left_index=True, right_on='sample_id')
    desc = 'pred_prob'
    print("all records:")
    all_auc, all_acc = print_metrics(test_df['READMISSION'], test_df[desc])
    fem_subset = test_df[test_df.GENDER == 'F']
    print("female records:")
    fem_auc, fem_acc = print_metrics(fem_subset['READMISSION'], fem_subset[desc])
    male_subset = test_df[test_df.GENDER == 'M']
    print("male records:")
    male_auc, male_acc = print_metrics(male_subset['READMISSION'], male_subset[desc])
    if name is not None:
        out_path = os.path.join(SAVE_PATH, "cross_validation", f"{name}_metrics.csv")
        if not os.path.exists(out_path):
            f = open(out_path, "w")
            f.write("all_auc,all_acc,fem_auc,fem_acc,male_auc,male_acc\n")
        else:
            f = open(out_path, "a")
        f.write(f"{all_auc},{all_acc},{fem_auc},{fem_acc},{male_auc},{male_acc}\n")
    return {'all_auc': all_auc, 'all_acc': all_acc, 'fem_auc': fem_auc, 'fem_acc': fem_acc, 'male_auc': male_auc, 'male_acc': male_acc}


def print_aucs_for_los(result_df):
    test_df = pd.read_csv(LOS_TEST_PATH_V4, index_col=0)
    test_df['LOS5'] = test_df['LOS'] > 5

    print(f"{sum(test_df['LOS5'])/len(test_df)} positive (>5 days)")
    test_df = test_df.merge(result_df, left_index=True, right_on='sample_id')
    desc = 'pred_prob'
    print("all records:")
    print_metrics(test_df['LOS5'], test_df[desc])

    fem_subset = test_df[test_df.GENDER == 'F']
    print(f"female records: {sum(fem_subset['LOS5']) / len(fem_subset)} positive (>5 days)")
    print_metrics(fem_subset['LOS5'], fem_subset[desc])

    male_subset = test_df[test_df.GENDER == 'M']
    print(f"male records: {sum(male_subset['LOS5']) / len(male_subset)} positive (>5 days)")
    print_metrics(male_subset['LOS5'], male_subset[desc])


def delong_on_df(df, true_field, pred1_field, pred2_field):
    df1 = df[(df[pred1_field] != 0) & (df[pred1_field] != 1) & (df[pred2_field] != 0) & (df[pred2_field] != 1)]
    print(f"dropped {len(df)-len(df1)} rows because of extreme values")
    return 10**delong_roc_test(df1[true_field], df1[pred1_field], df1[pred2_field])


def join_results(result_files_and_descs, output_fname, descs_for_auc_comparison=[]):
    test_df = pd.read_csv(READMIT_TEST_PATH, index_col=0)
    for desc, fname in result_files_and_descs:
        df = pd.read_csv(fname, index_col=0)
        df = df.rename({'pred_prob': desc}, axis=1)
        if 'sample_id' not in test_df.columns:
            test_df = test_df.merge(df, left_index=True, right_on='sample_id')
        else:
            test_df = test_df.merge(df, on='sample_id')
        test_df[f'{desc}_BCE_loss'] = -(test_df['READMISSION']*np.log(test_df[desc]) +
                                        (1 - test_df['READMISSION'])*np.log(1-test_df[desc]))
        print(f"{desc}\n~~~~~~~~~~~~~~~~")
        print_metrics(test_df['READMISSION'], test_df[desc])
        fem_subset = test_df[test_df.GENDER == 'F']
        print("Female metrics:")
        print_metrics(fem_subset['READMISSION'], fem_subset[desc])
        male_subset = test_df[test_df.GENDER == 'M']
        print("Male metrics:")
        print_metrics(male_subset['READMISSION'], male_subset[desc])
    if descs_for_auc_comparison is None and len(result_files_and_descs) == 2:
        descs_for_auc_comparison = [x[0] for x in result_files_and_descs]
    if len(descs_for_auc_comparison) == 2:
        desc1, desc2 = descs_for_auc_comparison
        print(f"comparing {desc1}, {desc2}:")
        print(f"All records: {delong_on_df(test_df, 'READMISSION', desc1, desc2)}")
        print(f"Female records ({len(test_df[test_df.GENDER == 'F'])}): {delong_on_df(test_df[test_df.GENDER == 'F'], 'READMISSION', desc1, desc2)}")
        print(f"Male records({len(test_df[test_df.GENDER == 'M'])}) :{delong_on_df(test_df[test_df.GENDER == 'M'], 'READMISSION', desc1, desc2)}")
    if output_fname is not None:
        test_df.to_csv(output_fname)


def analyze_results_by_field(res_file_or_df, compared_models=['female100', 'neutral100'],
                             field='ETHNICITY_CLEAN',
                             values=['BLACK/AFRICAN AMERICAN', 'WHITE', 'ASIAN', 'HISPANIC/LATINO']):
    if type(res_file_or_df) == str:
        res = pd.read_csv(res_file_or_df, index_col=0)
    else:
        res = res_file_or_df
    if values == []:
        values = set(res[field].values)
    for v in values:
        subset = res[res[field] == v]
        print(f"stats for {v}\n~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"All records ({len(subset)}):")
        if subset['READMISSION'].mean() == 0 or subset['READMISSION'].mean() == 1:
            print(f"all values are {subset['READMISSION'].iloc[0]}")
            continue
        for model in compared_models:
            print(f"AUC {model}")
            print_metrics(subset['READMISSION'], subset[model])
        desc1, desc2 = compared_models
        print(f"diff pvalue: {delong_on_df(subset, 'READMISSION', desc1, desc2)}")

        f_subset = subset[subset.GENDER == 'F']
        m_subset = subset[subset.GENDER == 'M']

        print(f"Female records ({len(f_subset)}):")
        if f_subset['READMISSION'].mean() == 0 or f_subset['READMISSION'].mean() == 1:
            print(f"all values are {f_subset['READMISSION'].iloc[0]}")
        else:
            for model in compared_models:
                print(f"AUC {model}")
                print_metrics(f_subset['READMISSION'], f_subset[model])
            print(f"diff pvalue {delong_on_df(f_subset, 'READMISSION', desc1, desc2)}")

        print(f"Male records ({len(m_subset)}):")
        if m_subset['READMISSION'].mean() == 0 or m_subset['READMISSION'].mean() == 1:
            print(f"all values are {f_subset['READMISSION'].iloc[0]}")
        else:
            for model in compared_models:
                print(f"AUC {model}")
                print_metrics(m_subset['READMISSION'], m_subset[model])
            print(f"diff pvalue {delong_on_df(m_subset, 'READMISSION', desc1, desc2)}")


def analyze_results_by_numeric_field(res_file_or_df, field, values, compared_models=['female100', 'neutral100']):
    if type(res_file_or_df) == str:
        res = pd.read_csv(res_file_or_df, index_col=0)
    else:
        res = res_file_or_df
    for i in range(len(values)+1):
        if i == 0:
            subset = res[res[field] == values[i]]
            s = "0 diags"
        elif i == len(values):
            subset = res[res[field] > values[i-1]]
            s = f">{values[i-1]} diags"
        else:
            subset = res[(values[i-1] < res[field]) & (res[field] <= values[i])]
            s = f"[{values[i-1]},{values[i]})"
        print(f"\nstats for {s}\n~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"All records ({len(subset)}):")
        for model in compared_models:
            print(f"AUC {model}")
            print_metrics(subset['READMISSION'], subset[model])
        desc1, desc2 = compared_models
        print(f"diff pvalue {delong_on_df(subset, 'READMISSION', desc1, desc2)}")

        f_subset = subset[subset.GENDER == 'F']
        m_subset = subset[subset.GENDER == 'M']

        print(f"Female records ({len(f_subset)}):")
        for model in compared_models:
            print(f"AUC {model}")
            print_metrics(f_subset['READMISSION'], f_subset[model])
        print(f"diff pvalue {delong_on_df(f_subset, 'READMISSION', desc1, desc2)}")

        print(f"Male records ({len(m_subset)}):")
        for model in compared_models:
            print(f"AUC {model}")
            print_metrics(m_subset['READMISSION'], m_subset[model])
        print(f"diff pvalue {delong_on_df(m_subset, 'READMISSION', desc1, desc2)}")


def analyze_results_by_prev_diags(res_file, compared_models=['female100', 'neutral100']):
    res = pd.read_csv(res_file, index_col=0)
    res['PREV_DIAGS'] = res['PREV_DIAGS'].apply(clean_nans).apply(literal_eval)
    res['num_prev_diags'] = res['PREV_DIAGS'].apply(len)
    values = [0, 20, 40]
    analyze_results_by_numeric_field(res, 'num_prev_diags', values, compared_models)


def analyze_results_by_charlson_index(res_file, compared_models=['female100', 'neutral100']):
    res = pd.read_csv(res_file, index_col=0)
    res['DIAGS'] = res['DIAGS'].apply(literal_eval)
    res['PREV_DIAGS'] = res['PREV_DIAGS'].apply(clean_nans).apply(literal_eval)
    res['DIAGS_set'] = res['DIAGS'].apply(set)
    res['all_diags'] = res.apply(lambda row: row['DIAGS_set'].union(row['DIAGS']), axis=1)

    def find_common_code(list1, list2):
        for code in list1:
            if code in list2:
                return True
        return False

    MI_codes = ['41000', '41001', '41002', '41010', '41011', '41012', '41020', '41021', '41022', '41030', '41031', '41032', '41040', '41041', '41042', '41050', '41051', '41052', '41080', '41081', '41082', '41090', '41091', '41092', '4110', '412']
    res['MI'] = res['all_diags'].apply(lambda x: find_common_code(x, MI_codes))
    CHF_codes = ['39891', '4280', '4281', '42820', '42821', '42822', '42823', '42830', '42831', '42832', '42833', '42840', '42841', '42842', '42843', '4289']
    res['CHF'] = res['all_diags'].apply(lambda x: find_common_code(x, CHF_codes))
    PVD_codes = ['44389', '4439', '74760', '74769', '9972']
    res['PVD'] = res['all_diags'].apply(lambda x: find_common_code(x, PVD_codes))
    CVA_or_TIA_codes = ['43884', '43885', '43889', '4389', '436', '4371', '4378', '4379', '4380', '43810', '43811', '43812', '43813', '43814', '43819', '43820', '43821', '43822', '43830', '43831', '43832', '43840', '43841', '43842', '43850', '43851', '43852', '43853', '4386', '4387', '43881', '43882', '43883', '74781', '67400', '67401', '67402', '67403', '99702', '67404', 'V1254', '38802']
    res['CVA_or_TIA'] = res['all_diags'].apply(lambda x: find_common_code(x, CVA_or_TIA_codes))
    dementia_codes = ['2900', '29010', '29011', '29012', '29013', '29020', '29021', '2903', '29040', '29041', '29042', '29043', '2912', '29282', '29410', '29411', '29420', '29421', '33119', '33182']
    res['dementia'] = res['all_diags'].apply(lambda x: find_common_code(x, dementia_codes))
    COPD_codes = ['49120', '49121', '49122']
    res['COPD'] = res['all_diags'].apply(lambda x: find_common_code(x, COPD_codes))
    connective_tissue_codes = ['7108', '7109']
    res['connective_tissue'] = res['all_diags'].apply(lambda x: find_common_code(x, connective_tissue_codes))
    peptic_ulcer_codes = ['53300', '53301', '53310', '53311', '53320', '53321', '53330', '53331', '53340', '53341', '53350', '53351', '53360', '53361', '53370', '53371', '53390', '53391', 'V1271']
    res['peptic_ulcer'] = res['all_diags'].apply(lambda x: find_common_code(x, peptic_ulcer_codes))
    liver_non_mild = ['5723', '5712', '5715', '5716']
    res['liver_non_mild'] = res['all_diags'].apply(lambda x: find_common_code(x, liver_non_mild))
    liver_mild = ['9162', '1305', '700', '701', '7022', '7023', '7032', '7033', '7041', '7042', '7043', '7044', '7049', '7051', '7052', '7053', '7054', '7059', '706', '7070', '7071', '709', '7271', '5711', '57140', '57141', '57142', '57149', '5731', '5732', '5733', 'V0260', 'V0261', 'V0262', 'V0269']
    res['liver_mild'] = res['all_diags'].apply(lambda x: find_common_code(x, liver_mild))
    DM_mild = ['25000', '25001', '25002', '25003', '24900', '24901']
    res['DM_mild'] = res['all_diags'].apply(lambda x: find_common_code(x, DM_mild))
    DM_complicated = ['24910', '24911', '24920', '24921', '24930', '24931', '24940', '24941', '24950', '24951', '24960', '24961', '24970', '24971', '24980', '24981', '24990', '24991']
    res['DM_complicated'] = res['all_diags'].apply(lambda x: find_common_code(x, DM_complicated))
    hemiplegia = ['34200', '34201', '34202', '34210', '34211', '34212', '34280', '34281', '34282', '34290', '34291', '34292', '3431', '3434', '43820', '43821', '43822']
    res['hemiplegia'] = res['all_diags'].apply(lambda x: find_common_code(x, hemiplegia))
    CKD_codes = ['28521', '40300', '40301', '40310', '40311', '40390', '40391', '40400', '40401', '40402', '40403', '40410', '40411', '40412', '40413', '40490', '40491', '40492', '40493', '5851', '5852', '5853', '5854', '5855', '5859']
    res['CKD'] = res['all_diags'].apply(lambda x: find_common_code(x, CKD_codes))
    # Tumor should be separated to localized or metastatic.
    tumor_codes = ['20260', '20261', '20262', '20263', '20264', '20265', '20266', '20267', '20268', '20020', '20021', '20022', '20023', '20024', '20025', '20026', '20027', '20028', '27788', '20900', '20901', '20902', '20903', '20910', '20911', '20912', '20913', '20914', '20915', '20916', '20917', '20920', '20921', '20922', '20923', '20924', '20925', '20926', '20927', '20929', '20940', '20941', '20942', '20943', '20950', '20951', '20952', '20953', '20954', '20955', '20956', '20957', '20960', '20961', '20962', '20963', '20964', '20965', '20966', '20967', '20969', '20970', '20971', '20972', '20973', '20974', '20979', '36564', '65410', '65411', '65412', '65413', '65414', '79589', '7310', '72702', 'V1091']
    res['tumor'] = res['all_diags'].apply(lambda x: find_common_code(x, tumor_codes))
    leukemia = ['20310', '20311', '20312', '20400', '20401', '20402', '20410', '20411', '20412', '20420', '20421', '20422', '20480', '20481', '20482', '20490', '20491', '20492', '20500', '20501', '20502', '20510', '20511', '20512', '20520', '20521', '20522', '20580', '20581', '20582', '20590', '20591', '20592', '20600', '20601', '20602', '20610', '20611', '20612', '20620', '20621', '20622', '20680', '20681', '20682', '20690', '20691', '20692', '20700', '20701', '20702', '20720', '20721', '20722', '20780', '20781', '20782', '20800', '20801', '20802', '20810', '20811', '20812', '20820', '20821', '20822', '20880', '20881', '20882', '20890', '20891', '20892', 'V1060', 'V1061', 'V1062', 'V1063', 'V1069']
    res['leukemia'] = res['all_diags'].apply(lambda x: find_common_code(x, leukemia))
    lymphoma = ['20076', '20200', '20201', '20202', '20203', '20204', '20205', '20206', '20207', '20208', '20270', '20271', '20272', '20273', '20274', '20275', '20276', '20277', '20278', '20280', '20281', '20282', '20283', '20284', '20285', '20286', '20287', '20288', '20020', '20021', '20022', '20023', '20024', '20025', '20026', '20027', '20028', '20030', '20031', '20032', '20033', '20034', '20035', '20036', '20037', '20038', '20040', '20041', '20042', '20043', '20044', '20045', '20046', '20047', '20048', '20050', '20051', '20052', '20053', '20054', '20055', '20056', '20057', '20058', '20060', '20061', '20062', '20063', '20064', '20065', '20066', '20067', '20068', '20070', '20071', '20072', '20073', '20074', '20075', '20077', '20078']
    res['lymphoma'] = res['all_diags'].apply(lambda x: find_common_code(x, lymphoma))
    AIDS = ['42', '7953', '79571']
    res['AIDS'] = res['all_diags'].apply(lambda x: find_common_code(x, AIDS))


    def calculate_charlson_comorbidity_index(row):
        # expecting a list of all diags: previous and current
        cci = 0
        age = row['AGE']
        if age >= 50:
            cci += 1
        if age >= 60:
            cci += 1
        if age >= 70:
            cci += 1
        if age >= 80:
            cci += 1
        for field in ['MI', 'CHF', 'PVD', 'CVA_or_TIA', 'dementia', 'COPD', 'connective_tissue', 'peptic_ulcer']:
            if row[field]:
                cci += 1
        if row['liver_non_mild']:
            cci += 3
        elif row['liver_mild']:
            cci += 1

        if row['DM_complicated']:
            cci += 2
        elif row['DM_mild']:
            cci += 1

        for field in ['hemiplegia', 'CKD', 'tumor', 'leukemia', 'lymphoma']:
            if row[field]:
                cci += 2

        if row['AIDS']:
            cci += 6
        return cci

    res['cci'] = res.apply(calculate_charlson_comorbidity_index, axis=1)
    analyze_results_by_numeric_field(res, 'cci', values=[0, 2, 4, 6], compared_models=compared_models)
    # analyze_results_by_field(res, values=[], field='cci')


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Classifier, self).__init__()
        # self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        # self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim//2)
        # self.linear3 = torch.nn.Linear(hidden_dim//2, 1)

        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, 1)

        # self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim//2)
        # self.linear3 = torch.nn.Linear(hidden_dim//2, hidden_dim//4)
        # self.linear4 = torch.nn.Linear(hidden_dim//4, 1)
        self.activation = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        out1 = self.activation(self.linear1(x))
        return self.linear2(out1)

        # out2 = self.activation(self.linear2(out1))
        # return self.linear3(out2)

        # out3 = self.activation(self.linear3(out2))
        # return self.linear4(out3)


class BertOnDiagsWithClassifier(BertOnDiagsBase):
    def __init__(self, bert_model, diag_to_title, procedure_to_title, lr, name, use_procedures=False, use_drugs=False,
                 use_lstm=False, additionals=0, label_field='readmitted', frozen_emb_file=None):
        super(BertOnDiagsWithClassifier, self).__init__(bert_model, diag_to_title, procedure_to_title, lr, name, use_lstm, frozen_emb_file)
        self.label_field = label_field
        if self.label_field == 'readmitted':
            self.print_aucs = print_aucs_for_readmission
        elif self.label_field == 'los':
            self.print_aucs = print_aucs_for_los

        # self.emb_size = self.bert_model.get_input_embeddings().embedding_dim
        print(f"embedding dim: {self.emb_size}")
        self.additionals = additionals
        print(f"Using {self.additionals} additionals.")
        cls_input_size = self.emb_size * 2 + additionals  # diags and previous diags
        self.use_procedures = use_procedures
        if self.use_procedures:
            cls_input_size += self.emb_size
        self.use_drugs = use_drugs
        if self.use_drugs:
            cls_input_size += self.emb_size

        self.classifier = Classifier(cls_input_size, 100)

        # this is for the elaborate forward_transformer
        self.activation = nn.ReLU()

        # self.heads_num = int(self.emb_size / 64)
        # self.sentence_transformer_encoder = Encoder1DLayer(d_model=self.emb_size, n_head=self.heads_num)
        # # Embedding for abstract-level CLS token:
        # self.cls = nn.Embedding(1, self.emb_size)
        # self.classifier = nn.Linear(self.emb_size, 1)
        #
        # if self.use_procedures:
        #     self.sentence_transformer_encoder2 = Encoder1DLayer(d_model=self.emb_size, n_head=self.heads_num)
        #     # Embedding for abstract-level CLS token:
        #     self.cls2 = nn.Embedding(1, self.emb_size)
        #     self.classifier2 = nn.Linear(self.emb_size, 1)
        #
        # if self.use_drugs:
        #     self.sentence_transformer_encoder3 = Encoder1DLayer(d_model=self.emb_size, n_head=self.heads_num)
        #     # Embedding for abstract-level CLS token:
        #     self.cls3 = nn.Embedding(1, self.emb_size)
        #     self.classifier3 = nn.Linear(self.emb_size, 1)
        #
        # final_classifier_size = 1 + sum([self.use_procedures, self.use_drugs])
        # if final_classifier_size > 1:
        #     self.final_classifier = nn.Linear(final_classifier_size, 1)

        # TODO: more layers? concat instead of mean? the most diags per person is 39
        # self.classifier = Classifier(emb_size, 20)
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward_generic(self, sequence, lookup_dict, sent_transformer, cls_instance, classifier_instance):
        # TODO: use diag_idfs from the batch somehow
        indexes, texts, max_len = self.code_list_to_text_list([list(literal_eval(str_repr)) for str_repr in sequence], lookup_dict)
        inputs = self.bert_tokenizer.batch_encode_plus(texts, padding=True, truncation=True,
                                                       max_length=70,
                                                       add_special_tokens=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # each title is embedded - we take the CLS token embedding
        sent_embedding = self.bert_model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0]
        sample_embedding, mask = [], []
        for start, end in indexes:
            if (start, end) == (-1, -1):
                sample_embedding.append(torch.zeros(max_len+1, self.emb_size, device=self.device))
                cur_mask = torch.ones(max_len+1, device=self.device).bool()
                cur_mask[:] = False
                mask.append(cur_mask)
                continue
            # embedding of CLS token.
            cls = cls_instance(torch.LongTensor([0]).to(self.device))
            cur_sent_embedding = sent_embedding[start:end]
            padding = torch.zeros(max_len - len(cur_sent_embedding), self.emb_size, device=self.device)
            sample_embedding.append(torch.cat([cls, cur_sent_embedding, padding], dim=0))

            cur_mask = torch.ones(max_len + 1, device=self.device).bool()
            cur_mask[len(cur_sent_embedding) + 1:] = False
            mask.append(cur_mask)
        sample_embedding = torch.stack(sample_embedding)
        mask = torch.stack(mask)
        # at this point sample_embedding holds a row for each sentence, and inside the embedded words + embedded 0 (padding).
        # the mask holds True for real words or False for padding.
        # only take the output for the cls - which represents the entire abstract.
        aggregated = sent_transformer(sample_embedding, slf_attn_mask=mask.unsqueeze(1))[0][:, 0, :]
        y_pred = classifier_instance(aggregated)
        return y_pred

    # def forward(self, x):
    #     indexes, texts, max_len = self.code_list_to_text_list([literal_eval(str_repr) for str_repr in x])
    #     inputs = self.bert_tokenizer.batch_encode_plus(texts, padding=True, truncation=True,
    #                                                    max_length=70,
    #                                                    add_special_tokens=True, return_tensors="pt")
    #     inputs = {k: v.to(self.device) for k, v in inputs.items()}
    #     # each title is embedded - we take the CLS token embedding
    #     sent_embedding = self.bert_model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0]
    #     sample_embedding, mask = [], []
    #     for start, end in indexes:
    #         # embedding of CLS token.
    #         cls = self.cls(torch.LongTensor([0]).to(self.device))
    #         cur_sent_embedding = sent_embedding[start:end]
    #         padding = torch.zeros(max_len - len(cur_sent_embedding), self.emb_size, device=self.device)
    #         sample_embedding.append(torch.cat([cls, cur_sent_embedding, padding], dim=0))
    #
    #         cur_mask = torch.ones(max_len + 1, device=self.device).bool()
    #         cur_mask[len(cur_sent_embedding) + 1:] = False
    #         mask.append(cur_mask)
    #     sample_embedding = torch.stack(sample_embedding)
    #     mask = torch.stack(mask)
    #     # at this point sample_embedding holds a row for each sentence, and inside the embedded words + embedded 0 (padding).
    #     # the mask holds True for real words or False for padding.
    #     # only take the output for the cls - which represents the entire abstract.
    #     aggregated = self.sentence_transformer_encoder(sample_embedding, slf_attn_mask=mask.unsqueeze(1))[0][:, 0, :]
    #
    #     y_pred = self.classifier(aggregated)
    #
    #     return y_pred

    def forward(self, batch, name):
        agg = 'lstm' if self.use_lstm else 'sum'
        sample_diag_embeddings = self.embed_diags(batch['diags'], agg)
        sample_prev_diag_embeddings = self.embed_diags(batch['prev_diags'], agg)
        sample_embeddings = torch.cat([sample_prev_diag_embeddings, sample_diag_embeddings], dim=1)
        if self.use_procedures:
            sample_proc_embeddings = self.embed_procs(batch['procedures'], agg)
            sample_embeddings = torch.cat([sample_embeddings, sample_proc_embeddings], dim=1)
        if self.use_drugs:
            sample_drug_embeddings = self.embed_drugs(batch['drugs'], agg)
            sample_embeddings = torch.cat([sample_embeddings, sample_drug_embeddings], dim=1)

        if self.additionals > 0:
            sample_embeddings = torch.cat([sample_embeddings, batch['additionals']], dim=1)

        ypred = self.classifier(sample_embeddings)
        return ypred

    def y_pred_to_probabilities(self, y_pred):
        return torch.sigmoid(y_pred)

    def forward_transformer(self, batch, name):
        ytrue = batch['readmitted'].to(torch.float32)
        ypred_by_diags_logit = self.forward_generic(batch['diags'],
                                                    self.diag_to_title,
                                                    self.sentence_transformer_encoder,
                                                    self.cls,
                                                    self.classifier)
        diag_losses = self.loss_func(ypred_by_diags_logit.squeeze(1), ytrue)
        self.log(f'classification/{name}_diags_BCE_loss', diag_losses.mean(), batch_size=BATCH_SIZE)

        inputs_to_final_classifier = [self.activation(ypred_by_diags_logit)]
        if self.use_procedures:
            ypred_by_procedures_logit = self.forward_generic(batch['procedures'],
                                                             self.procedure_to_title,
                                                             self.sentence_transformer_encoder2,
                                                             self.cls2,
                                                             self.classifier2)
            inputs_to_final_classifier.append(self.activation(ypred_by_procedures_logit))
            proc_losses = self.loss_func(ypred_by_procedures_logit.squeeze(1), ytrue)
            self.log(f'classification/{name}_procs_BCE_loss', proc_losses.mean(), batch_size=BATCH_SIZE)
        if self.use_drugs:
            ypred_by_drugs_logit = self.forward_generic(batch['drugs'], None, self.sentence_transformer_encoder3,
                                                        self.cls3, self.classifier3)
            inputs_to_final_classifier.append(self.activation(ypred_by_drugs_logit))
            drug_losses = self.loss_func(ypred_by_drugs_logit.squeeze(1), ytrue)
            self.log(f'classification/{name}_drugs_BCE_loss', drug_losses.mean(), batch_size=BATCH_SIZE)
        if len(inputs_to_final_classifier) > 1:
            # print(f"shape of ypred_by_diags: {ypred_by_diags_logit.shape}, ypred_by_procs: {ypred_by_procedures_logit.shape}")
            # ypred_logit = self.final_classifier(torch.cat(inputs_to_final_classifier, dim=1))
            ypred_logit = self.final_classifier(torch.cat(inputs_to_final_classifier, dim=1))
            # print(f"shape of final ypred: {ypred_logit.shape}")
        else:
            ypred_logit = ypred_by_diags_logit
        return ypred_logit

    def step(self, batch, name):
        ypred_logit = self.forward(batch, name).squeeze(1)
        losses = self.loss_func(ypred_logit, batch[self.label_field].to(torch.float32))  # calculates loss per sample
        loss = losses.mean()
        self.log(f'classification/{name}_BCE_loss', loss)
        if name == 'val':
            ypred = self.y_pred_to_probabilities(self.forward(batch, 'mid_val'))
            return {'loss': loss, 'sample_id': batch['sample_id'], 'pred_prob': ypred}
        return {'loss': loss}

    def validation_epoch_end(self, outputs) -> None:
        sample_ids = np.concatenate([batch['sample_id'].cpu().numpy() for batch in outputs])
        pred_prob = np.concatenate([batch['pred_prob'].cpu().numpy().squeeze() for batch in outputs])
        df = pd.DataFrame.from_dict({'sample_id': sample_ids, 'pred_prob': pred_prob}, orient='columns')
        try:
            res = self.print_aucs(df)
            self.log(f'classification/AUC_all', res['all_auc'])
            self.log(f'classification/AUC_female', res['fem_auc'])
            self.log(f'classification/AUC_male', res['male_auc'])
        except:
            print("Could not calculate AUCs, probably only one class in batch.")

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        ypred = self.y_pred_to_probabilities(self.forward(batch, 'test'))
        return batch['sample_id'], ypred

    def test_epoch_end(self, outputs) -> None:
        records = []
        for batch in outputs:
            batch_size = len(batch[0])
            # tuple returned by test_step
            for i in range(batch_size):
                records.append({
                    'sample_id': batch[0][i].cpu().numpy(),
                    'pred_prob': batch[1][i].cpu().numpy().squeeze(),
                    })
        df = pd.DataFrame.from_records(records)
        df.to_csv(os.path.join(SAVE_PATH, f'readmission_test_{self.name}.csv'))
        df = pd.read_csv(os.path.join(SAVE_PATH, f'readmission_test_{self.name}.csv'), index_col=0)
        if CROSS_VAL is not None:
            self.print_aucs(df, self.name)
        else:
            self.print_aucs(df)


    def configure_optimizers(self):
        grouped_parameters = [
            #{'params': self.bert_model.pooler.dense.parameters()},
            # {'params': self.cls.parameters()},
            # {'params': self.sentence_transformer_encoder.parameters()},
            {'params': self.classifier.parameters()}
        ]
        if self.use_lstm:
            grouped_parameters.append({'params': self.lstm.parameters()})
        # if self.use_procedures:
        #     grouped_parameters.extend([
        #         {'params': self.cls2.parameters()},
        #         {'params': self.sentence_transformer_encoder2.parameters()},
        #         {'params': self.classifier2.parameters()}
        #     ])
        # if self.use_drugs:
        #     grouped_parameters.extend([
        #         {'params': self.cls3.parameters()},
        #         {'params': self.sentence_transformer_encoder3.parameters()},
        #         {'params': self.classifier3.parameters()}
        #     ])
        optimizer = torch.optim.Adam(grouped_parameters, lr=self.learning_rate)
        return [optimizer]


class EmbOnDiagsWithClassifier(EmbOnDiagsBase):
    def __init__(self, emb_path, lr, name, use_procedures=False, use_lstm=False, additionals=0,
                 label_field='readmitted', agg_prev_diags=None, agg_diags=None, use_diags=True):
        # Can't use the drugs data because they are not CUIs
        super(EmbOnDiagsWithClassifier, self).__init__(emb_path, lr, name, use_lstm)
        self.label_field = label_field
        if self.label_field == 'readmitted':
            self.print_aucs = print_aucs_for_readmission
        elif self.label_field == 'los':
            self.print_aucs = print_aucs_for_los
        default_agg = 'lstm' if self.use_lstm else 'sum'
        self.agg_prev_diags = agg_prev_diags
        if self.agg_prev_diags is None:
            self.agg_prev_diags = default_agg
        self.agg_diags = agg_diags
        if self.agg_diags is None:
            self.agg_diags = default_agg
        self.default_agg = default_agg

        self.use_diags = use_diags
        cls_input_size = 0
        if self.use_diags:
            self.emb_size = list(self.emb.values())[0].shape[0]
            cls_input_size += 2*self.emb_size  # diags + prev_diags
        self.use_procedures = use_procedures
        if self.use_procedures:
            cls_input_size += self.emb_size
        self.additionals = additionals
        cls_input_size += self.additionals

        # self.heads_num = max(1, int(self.emb_size / 64))
        # self.sentence_transformer_encoder = Encoder1DLayer(d_model=self.emb_size, n_head=self.heads_num)
        # # Embedding for patient-level CLS token:
        # self.cls = nn.Embedding(1, self.emb_size)
        # self.classifier = nn.Linear(self.emb_size, 1)

        # TODO: more layers? concat instead of mean? the most diags per person is 39
        print(f"cls input size: {cls_input_size}")
        # self.classifier = Classifier(cls_input_size, 100)
        self.classifier = Classifier(cls_input_size, 100)
        # self.classifier = nn.Linear(cls_input_size, 1)
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward_transformer(self, x):
        max_len = 0
        sample_embeddings = []
        for admission_str in x:
            admission = literal_eval(admission_str)
            max_len = max(max_len, len(admission))
            embs = torch.Tensor([self.emb[code] for code in admission]).to(self.device)
            sample_embeddings.append(embs)
        padded_samples, mask = [], []
        for embs in sample_embeddings:
            # embedding of CLS token.
            cls = self.cls(torch.LongTensor([0]).to(self.device))
            padding = torch.zeros(max_len - len(embs), self.emb_size, device=self.device)
            padded_samples.append(torch.cat([cls, embs, padding], dim=0))

            cur_mask = torch.ones(max_len + 1, device=self.device).bool()
            cur_mask[len(embs) + 1:] = False
            mask.append(cur_mask)
        padded_samples = torch.stack(padded_samples)
        mask = torch.stack(mask)
        # at this point sample_embedding holds a row for each sentence, and inside the embedded words + embedded 0 (padding).
        # the mask holds True for real words or False for padding.
        # only take the output for the cls - which represents the entire abstract.
        aggregated = self.sentence_transformer_encoder(padded_samples, slf_attn_mask=mask.unsqueeze(1))[0][:, 0, :]
        y_pred = self.classifier(aggregated)
        return y_pred

    def forward(self, batch):
        # agg = 'lstm' if self.use_lstm else 'sum'
        # sample_diag_embeddings = self.embed_diags(batch['diags'], agg, batch['diag_idfs'])
        if self.use_diags:
            sample_diag_embeddings = self.embed_codes(batch['diags'], self.agg_diags)  # no idf weighting
            sample_prev_diag_embeddings = self.embed_codes(batch['prev_diags'], self.agg_prev_diags)
            sample_embeddings = torch.cat([sample_prev_diag_embeddings, sample_diag_embeddings], dim=1)
        if self.use_procedures:
            sample_proc_embeddings = self.embed_codes(batch['procedures'], self.default_agg)
            sample_embeddings = torch.cat([sample_embeddings, sample_proc_embeddings], dim=1)
        if self.additionals > 0:
            if self.use_diags:
                sample_embeddings = torch.cat([sample_embeddings, batch['additionals']], dim=1)
            else:  # only additionals
                sample_embeddings = batch['additionals']
        ypred = self.classifier(sample_embeddings)
        return ypred

    def y_pred_to_probabilities(self, y_pred):
        return torch.sigmoid(y_pred)

    def step(self, batch, name):
        ypred_logit = self.forward(batch).squeeze(1)
        losses = self.loss_func(ypred_logit, batch[self.label_field].to(torch.float32))  # calculates loss per sample
        loss = losses.mean()
        self.log(f'classification/{name}_BCE_loss', loss, batch_size=BATCH_SIZE)
        return {'loss': loss}

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        return batch['sample_id'], self.y_pred_to_probabilities(self.forward(batch)).squeeze(1)

    def test_epoch_end(self, outputs):
        records = []
        for batch in outputs:
            batch_size = len(batch[0])
            # tuple returned by test_step
            for i in range(batch_size):
                records.append({
                    'sample_id': batch[0][i].cpu().numpy(),
                    'pred_prob': batch[1][i].cpu().numpy(),
                    })
        df = pd.DataFrame.from_records(records)
        df.to_csv(os.path.join(SAVE_PATH, f'readmission_test_{self.name}.csv'))
        df = pd.read_csv(os.path.join(SAVE_PATH, f'readmission_test_{self.name}.csv'), index_col=0)
        return self.print_aucs(df)

    def configure_optimizers(self):
        grouped_parameters = [
            #{'params': self.bert_model.pooler.dense.parameters()},
            #{'params': self.cls.parameters()},
            #{'params': self.sentence_transformer_encoder.parameters()},
            {'params': self.classifier.parameters()}
        ]
        if self.use_lstm:
            grouped_parameters.append({'params': self.lstm.parameters()})
        optimizer = torch.optim.Adam(grouped_parameters, lr=self.learning_rate)
        return [optimizer]

    def get_weights(self, feature_names):
        feature_weights = self.classifier.weight # shape: (1, in_features)
        names_and_weights = list(zip(feature_names, feature_weights.detach().transpose(0,1)))
        names_and_weights = [(name, w.item()) for name, w in names_and_weights]
        df = pd.DataFrame(names_and_weights, columns=['feat_name', 'weight'])

        #names_and_weights = sorted(names_and_weights, key=lambda x: np.abs(x[1]), reverse=True)
        return df




def find_best_threshold_for_model(res_df, model_fields, label_field='READMISSION'):
    fem = res_df[res_df.GENDER == 'F']
    male = res_df[res_df.GENDER == 'M']
    for model_field in model_fields:
        print(model_field)
        print("_________________")

        # find a single best threshold of the model
        fpr, tpr, thresholds = roc_curve(res_df[label_field], res_df[model_field])
        gmeans = (tpr * (1 - fpr)) ** 0.5
        ix = np.argmax(gmeans)
        thresh = thresholds[ix]
        print('Best Threshold={}, G-Mean={:.3f}'.format(thresholds[ix], gmeans[ix]))
        for desc, res in [("female", fem), ("male", male), ("all", res_df)]:
            print(desc)
            # calculate tpr and fpr for the found threshold
            ypred = res[model_field] > thresh
            ytrue = res[label_field] == 1 # convert to boolean
            tpr_thresh = np.sum(ypred & ytrue)/np.sum(ytrue)
            fpr_thresh = 1 - np.sum((~ypred) & (~ytrue))/np.sum(~ytrue)
            print('thresh = {}, tpr: {} fpr: {}'.format(thresh, tpr_thresh, fpr_thresh))


def calc_avg_results(fname_template):
    # list of: {'all_auc', 'all_acc', 'fem_auc', 'fem_acc', 'male_auc', 'male_acc'}
    results_list = []
    print(fname_template)
    #TODO: what does the confidence interval look like?
    for i in range(CROSS_VAL):
        res = pd.read_csv(fname_template.format(i)).to_dict(orient='records')[0]
        results_list.append(res)
    for population in ['all', 'fem', 'male']:
        aucs = [res[f'{population}_auc'] for res in results_list]
        ci = st.t.interval(0.95, len(aucs) - 1, loc=np.mean(aucs), scale=st.sem(aucs))
        print(f"population: {population}, mean auc: {np.mean(aucs)}, CI: {ci}")


if __name__ == '__main__':
    desc, model_path = DESCS_AND_MODELS[RUN_MODEL_INDEX]
    if UPSAMPLE_FEMALE is not None:
        desc = desc + "_random_upsample"
    elif "smote" in READMIT_TEST_PATH:
        desc = desc + "_smote"
    elif "medgan" in READMIT_TEST_PATH:
        desc = desc + "_medgan"
    elif DOWNSAMPLE_MALE:
        desc = desc + "_downsample"

    desc = desc_to_name(desc)
    if not USE_EMB:
        desc = 'only_feats'
    frozen_emb = None
    if RUN_MODEL_INDEX == 39:  # null it out uses frozen embeddings
        frozen_emb = os.path.expanduser('~/fairemb/exp_results/nullspace_projection/BERT_tiny_medical_diags_and_drugs_debiased.tsv')


    # cui_embeddings = [7, 8, 9, 10, 11, 12, 13, 14, 15]
    if RUN_MODEL_INDEX in cui_embeddings:
        embs_with_missing_diags = [DESCS_AND_MODELS[i][1] for i in cui_embeddings]
        dm = ReadmissionbyEmbDiagsModule(embs_with_missing_diags, batch_size=BATCH_SIZE)
        # without measurements: 48 additionals
        # with measurements: 90 additionals
        # with measurements but removed some features: 78
        model = EmbOnDiagsWithClassifier(model_path, lr=LR, name=desc, use_procedures=False, use_lstm=False,
                                         additionals=90, use_diags=USE_EMB)
        logger = WandbLogger(name=desc, save_dir=LOG_PATH,
                             version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                             project='FairEmbedding_test',
                             config={'lr': LR, 'batch_size': BATCH_SIZE}
                             )
        trainer = pl.Trainer(gpus=1,
                             max_epochs=MAX_EPOCHS,
                             logger=logger,
                             log_every_n_steps=20,
                             accumulate_grad_batches=1,
                             # num_sanity_val_steps=2,
                             )
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)


    else:
        # Here the keys are str
        diag_dict = pd.read_csv(os.path.join(MIMIC_PATH, 'D_ICD_DIAGNOSES.csv'), index_col=0)
        diag_dict = diag_dict.set_index('ICD9_CODE')['LONG_TITLE'].to_dict()  # icd9 code to description
        # Here the keys are int
        proc_dict = pd.read_csv(os.path.join(MIMIC_PATH, 'D_ICD_PROCEDURES.csv'), index_col=0)
        proc_dict = proc_dict.set_index('ICD9_CODE')['LONG_TITLE'].to_dict()

        if CROSS_VAL is not None:
            results = []
            for i in range(CROSS_VAL):
                model = BertOnDiagsWithClassifier(model_path, diag_dict, proc_dict, lr=LR, name=desc + f"_CV{i}",
                                                  use_procedures=False, use_drugs=True,
                                                  # use_procedures=False, use_drugs=False,
                                                  use_lstm=True,
                                                  additionals=ADDITIONALS,
                                                  # additionals=3, # age and gender only
                                                  # additionals=110
                                                  frozen_emb_file=frozen_emb
                                                  )
                dm = ReadmissionbyDiagsModule(diag_dict, proc_dict, batch_size=BATCH_SIZE,
                                              upsample_female_file=UPSAMPLE_FEMALE, downsample_male=DOWNSAMPLE_MALE, CV_fold=i)
                logger = WandbLogger(name=desc+f"_CV{i}", save_dir=LOG_PATH,
                                     version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                                     project='FairEmbedding_test',
                                     config={'lr': LR, 'batch_size': BATCH_SIZE}
                                     )
                trainer = pl.Trainer(gpus=1,
                                     max_epochs=MAX_EPOCHS,
                                     logger=logger,
                                     log_every_n_steps=20,
                                     accumulate_grad_batches=1,
                                     # num_sanity_val_steps=2,
                                     )
                trainer.fit(model, datamodule=dm)
                # TODO: trainer.test returns empty dict
                res = trainer.test(model, datamodule=dm)
                results.append(res)
            calc_avg_results(os.path.join(SAVE_PATH, "cross_validation", desc+"_CV{}_metrics.csv"))

        else:
            dm = ReadmissionbyDiagsModule(diag_dict, proc_dict, batch_size=BATCH_SIZE, downsample_male=DOWNSAMPLE_MALE,
                                          upsample_female_file=UPSAMPLE_FEMALE)
            model = BertOnDiagsWithClassifier(model_path, diag_dict, proc_dict, lr=LR, name=desc,
                                              use_procedures=False, use_drugs=True,
                                              # use_procedures=False, use_drugs=False,
                                              use_lstm=True,
                                              additionals=ADDITIONALS,  # age and gender only
                                              # additionals=110
                                              frozen_emb_file=frozen_emb
                                              )
            logger = WandbLogger(name=desc, save_dir=LOG_PATH,
                                 version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                                 project='FairEmbedding_test',
                                 config={'lr': LR, 'batch_size': BATCH_SIZE}
                                 )
            trainer = pl.Trainer(gpus=1,
                                 max_epochs=MAX_EPOCHS,
                                 logger=logger,
                                 log_every_n_steps=20,
                                 accumulate_grad_batches=1,
                                 # num_sanity_val_steps=2,
                                 )
            trainer.fit(model, datamodule=dm)
            trainer.test(model, datamodule=dm)

    # feature importance analysis
    # feature_names = (40 * ['prev_emb']) + (40 * ['emb']) + dm.train.categorical_features + dm.train.non_cat_features
    # names_and_weights = model.get_weights(feature_names)
    # names_and_weights.to_csv(os.path.join(SAVE_PATH, f'{desc}_readmission_logreg_feat_importance.csv'))

    # to_show = [DESCS_AND_MODELS[i] for i in (1, 3, 5, 6)]
    # join_results([(desc, os.path.join(SAVE_PATH, f'readmission_test_{desc_to_name(desc)}.csv')) for desc, path in to_show],
    #              output_fname=os.path.join(SAVE_PATH, 'readmission_test_with_attrs_with2020.csv'))

