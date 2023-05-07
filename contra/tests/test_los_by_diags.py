import pytorch_lightning as pl
# pl.trainer.seed_everything(42)
import sys
sys.path.append('/home/shunita/fairemb/')
import os
import pandas as pd
import numpy as np
import scipy.stats as st
import torch
from torch import nn
import pytz
import random
# import pytorch_lightning as pl
import matplotlib.pyplot as plt
from datetime import datetime
from ast import literal_eval
from collections import defaultdict
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer, AutoModel
from contra.constants import SAVE_PATH, LOG_PATH, DATA_PATH
from contra.datasets.test_by_diags_dataset import LOSbyDiagsModule, LOSbyEmbDiagsModule
from contra.tests.bert_on_diags_base import BertOnDiagsBase, EmbOnDiagsBase
from contra.tests.test_readmission_by_diags import EmbOnDiagsWithClassifier
from contra.constants import LOS_TEST_PATH_V4
from sklearn.metrics import mean_squared_error
from contra.utils.diebold_mariano import dm_test
from contra.tests.descs_and_models import DESCS_AND_MODELS, cui_embeddings

MIMIC3_CUSTOM_TASKS_PATH = '/home/shunita/mimic3/custom_tasks/data'
MIMIC_PATH = "/home/shunita/mimic3/physionet.org/files/mimiciii/1.4/"


# LR = 5e-4
# BATCH_SIZE = 163

# params used in ACL ARR, october 15 2021
LR = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 1

# params used for bert-base runs
# LR = 1e-4
# BATCH_SIZE = 4

RUN_MODEL_INDEX = 5
# UPSAMPLE_FEMALE = 'data/los_by_diags_female_sample.csv'
UPSAMPLE_FEMALE = None
DOWNSAMPLE_MALE = False
USE_EMB = True
CROSS_VAL = None  # number of folds (10) or None
# 3 - age+gender (used in upsample). 0 - no additional features (used in main figure). 1- only age.
ADDITIONALS = 7
TEST_CHECKPOINT = '/home/shunita/fairemb/saved_models/LOS/exp_15/epoch=0-step=304.ckpt'
# TEST_CHECKPOINT = None



# setting random seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


########### utilities ###############

def dm_test_on_df(df, true_field, pred1_field, pred2_field):
    return dm_test(df[true_field], df[pred1_field], df[pred2_field])


def join_results(result_files_and_descs, output_fname, descs_for_comparison=[]):
    test_df = pd.read_csv(LOS_TEST_PATH_V4, index_col=0)
    if 'source' in test_df.columns:
        test_df = test_df[test_df['source'] == 'orig']
    for desc, fname in result_files_and_descs:
        df = pd.read_csv(fname, index_col=0)
        df = df.rename({'pred_LOS': desc}, axis=1)
        if 'sample_id' not in test_df.columns:
            test_df = test_df.merge(df, left_index=True, right_on='sample_id')
        else:
            test_df = test_df.merge(df, on='sample_id')
        test_df[f'{desc}_loss'] = np.abs(test_df[desc]-test_df['LOS'])
    if descs_for_comparison is None and len(result_files_and_descs) == 2:
        descs_for_comparison = [x[0] for x in result_files_and_descs]
    if len(descs_for_comparison) == 2:
        desc1, desc2 = descs_for_comparison
        print(f"comparing {desc1}, {desc2}:")

        print(f"All records: {dm_test_on_df(test_df, 'LOS', desc1, desc2)}")
        print(f"Female records ({len(test_df[test_df.GENDER == 'F'])}): {dm_test_on_df(test_df[test_df.GENDER == 'F'], 'LOS', desc1, desc2)}")
        print(f"Male records({len(test_df[test_df.GENDER == 'M'])}) :{dm_test_on_df(test_df[test_df.GENDER == 'M'], 'LOS', desc1, desc2)}")
    test_df.to_csv(output_fname)


def abs_error(ytrue, ypred):
    return np.mean(np.abs(ytrue-ypred))


def print_results(result_df, name=None, log_file=None, row_desc=''):
    test_df = pd.read_csv(LOS_TEST_PATH_V4, index_col=0)
    test_df = test_df.merge(result_df, left_index=True, right_on='sample_id')
    if 'source' in test_df.columns:
        test_df = test_df[test_df['source'] == 'orig']
    fem_subset = test_df[test_df.GENDER == 'F']
    male_subset = test_df[test_df.GENDER == 'M']
    all_RMSE = mean_squared_error(test_df['LOS'], test_df['pred_LOS'], squared=False)
    print(f"\nRMSE all records: {all_RMSE}")
    fem_RMSE = mean_squared_error(fem_subset['LOS'], fem_subset['pred_LOS'], squared=False)
    print(f"RMSE female records: {fem_RMSE}")
    male_RMSE = mean_squared_error(male_subset['LOS'], male_subset['pred_LOS'], squared=False)
    print(f"RMSE male records: {male_RMSE}")
    all_MAE = abs_error(test_df['LOS'], test_df['pred_LOS'])
    print(f"\nMAE loss on all records: {all_MAE}")
    fem_MAE = abs_error(fem_subset['LOS'], fem_subset['pred_LOS'])
    print(f"MAE loss on female records: {fem_MAE}")
    male_MAE = abs_error(male_subset['LOS'], male_subset['pred_LOS'])
    print(f"MAE loss on male records: {male_MAE}")
    if name is not None or log_file is not None:
        if name is not None:
            out_path = os.path.join(SAVE_PATH, "cross_validation", f"{name}_metrics.csv")
        else:
            out_path = log_file
        if not os.path.exists(out_path):
            f = open(out_path, "w")
            f.write("row,all_RMSE,fem_RMSE,male_RMSE,all_MAE,fem_MAE,male_MAE\n")
        else:
            f = open(out_path, "a")
        f.write(f"{row_desc},{all_RMSE},{fem_RMSE},{male_RMSE},{all_MAE},{fem_MAE},{male_MAE}\n")
    return {'all_MAE': all_MAE, 'fem_MAE': fem_MAE, 'male_MAE': male_MAE,
            'all_RMSE': all_RMSE, 'fem_RMSE': fem_RMSE, 'male_RMSE': male_RMSE}


def calc_avg_results(fname_template):
    #{'all_MAE': all_MAE, 'fem_MAE': fem_MAE, 'male_MAE': male_MAE,
    #'all_RMSE': all_RMSE, 'fem_RMSE': fem_RMSE, 'male_RMSE': male_RMSE}
    results_list = []
    print(fname_template)
    for i in range(CROSS_VAL):
        res = pd.read_csv(fname_template.format(i)).to_dict(orient='records')[0]
        results_list.append(res)
    for population in ['all', 'fem', 'male']:
        maes = [res[f'{population}_MAE'] for res in results_list]
        ci = st.t.interval(0.95, len(maes) - 1, loc=np.mean(maes), scale=st.sem(maes))
        print(f"population: {population}, mean MAE: {np.mean(maes)}, CI: {ci}")


def union(list_of_lists):
    s = []
    for s1 in list_of_lists:
        s.extend(s1)
    return list(set(s))


def analyze_results_by_disease(combined_res_file, model_fields=[], output_file=None, make_boxplot=False):
    diag_dict_short = pd.read_csv(os.path.join(MIMIC_PATH, 'D_ICD_DIAGNOSES.csv'), index_col=0).set_index('ICD9_CODE')[
        'SHORT_TITLE'].to_dict()
    diag_dict_long = pd.read_csv(os.path.join(MIMIC_PATH, 'D_ICD_DIAGNOSES.csv'), index_col=0).set_index('ICD9_CODE')[
        'LONG_TITLE'].to_dict()
    df = pd.read_csv(combined_res_file, index_col=0)
    df['PREV_DIAGS'] = df['PREV_DIAGS'].apply(literal_eval)
    if len(model_fields) == 0:
        model_fields = [c for c in df.columns if c.endswith('loss')]
    all_diags = union(df.PREV_DIAGS.values)
    records = []
    values = defaultdict(dict)
    for diag in all_diags:
        subset = df[df.PREV_DIAGS.apply(lambda x: diag in x)]
        record = {'ICD9': diag,
                  'short_title': diag_dict_short[diag] if diag in diag_dict_short else 'NA',
                  'long_title': diag_dict_long[diag] if diag in diag_dict_long else 'NA',
                  'count_M': len(subset[subset.GENDER == 'M']),
                  'count_F': len(subset[subset.GENDER == 'F']),
                  'count_patients': len(subset)}
        for field_name in model_fields:
            values[field_name][diag] = subset[field_name].values
            record[f'{field_name}_avg'] = subset[field_name].mean()
        records.append(record)
    res = pd.DataFrame.from_records(records)
    if make_boxplot:
            make_boxplot_from_values(values, ['BERT10-18_40eps_loss', 'GAN20_loss'],
                                     ['4019', '4280', '5849', '42731', '51881', '29181', '49390'],
                                     ['Hypertension essential', 'CHF', 'Acute Kidney Failure', 'Atrial Fibrillation',
                                      'Acute resp. fail.', 'Alcohol Withdrawl', 'Asthma'])
    if output_file is not None:
        res.to_csv(output_file)
    else:
        return res


def make_boxplot_from_values(values_dict, models_to_show, diags_to_show, diag_names):
    ticks = [diag_names[i] for i, d in enumerate(diags_to_show)]
    colors = ['#D7191C', '#2C7BB6']  # colors are from http://colorbrewer2.org/
    offsets = [-0.4, 0.4]

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure()

    for i, model_name in enumerate(models_to_show):
        print(f"1: {diags_to_show}, 2:{model_name}, 3: {list(values_dict[model_name].keys())[:3]}")
        data = [values_dict[model_name][d] for d in diags_to_show]
        bp = plt.boxplot(data, positions=np.array(range(len(data))) * 2.0 + offsets[i], sym='', widths=0.6)
        #bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b))) * 2.0 + 0.4, sym='', widths=0.6)
        set_box_color(bp, colors[i])
        #set_box_color(bpr, '#2C7BB6')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c=colors[0], label=models_to_show[0])
    plt.plot([], c=colors[1], label=models_to_show[1])
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks) * 2)
    plt.ylim(0, 8)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, 'los_test_2L_boxcompare.png'))


class Regressor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Regressor, self).__init__()
        # self.linear1 = torch.nn.Linear(input_dim, 1)

        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, 1)

        # self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        # self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim//2)
        # self.linear3 = torch.nn.Linear(hidden_dim//2, 1)

        self.activation = torch.nn.ReLU(inplace=False)
        print("Regressor model:")
        print(self)

    def forward(self, x):
        # 1 layer
        # return self.linear1(x)
        # 2 layers
        out1 = self.activation(self.linear1(x))
        return self.linear2(out1)
        # 3 layers
        # out1 = self.activation(self.linear1(x))
        # out2 = self.activation(self.linear2(out1))
        # return self.linear3(out2)


class BertOnDiagsWithRegression(BertOnDiagsBase):
    def __init__(self, bert_model, diag_to_title, proc_to_title, lr, name, use_lstm=False, use_diags=True,
                 additionals=0, frozen_emb_file=None, save_dir=None):
        super(BertOnDiagsWithRegression, self).__init__(bert_model, diag_to_title, proc_to_title, lr, name, use_lstm, frozen_emb_file)
        self.save_dir = save_dir
        self.additionals = additionals
        print(f"BertOnDiagsWithRegression: using {self.additionals} additional features")
        # emb_size = self.bert_model.get_input_embeddings().embedding_dim
        self.use_diags = use_diags
        input_dim = 0
        if self.use_diags:
            input_dim += 2 * self.emb_size  # primary diag + prev_diags
        input_dim += additionals
        # self.regression_model = Regressor(emb_size, emb_size//2)
        hidden = 128  # this is the emb size of tiny bert but can be increased when using bert base (768?).
        self.regression_model = Regressor(input_dim, hidden)
        # self.regression_model = torch.nn.Linear(input_dim, hidden_dim)
        self.loss_func = torch.nn.MSELoss()

    def forward(self, batch):
        # test here what happens if we only take the first diagnosis
        agg = 'lstm' if self.use_lstm else 'sum'
        if self.use_diags:
            sample_prev_diag_embeddings = self.embed_diags(batch['prev_diags'], agg=agg)
            sample_primary_diag_embeddings = self.embed_diags(batch['diags'], agg='first')
            sample_embeddings = torch.cat([sample_prev_diag_embeddings, sample_primary_diag_embeddings], dim=1)
        if self.additionals > 0:
            if self.use_diags:
                sample_embeddings = torch.cat([sample_embeddings, batch['additionals']], dim=1)
            else:  # only additionals
                sample_embeddings = batch['additionals']
        ypred = self.regression_model(sample_embeddings)
        return ypred

    def step(self, batch, name):
        ypred = self.forward(batch).squeeze(1)
        loss = self.loss_func(ypred, batch['los'].to(torch.float32))
        self.log(f'regression/{name}_MSE_loss', loss)
        if name == 'val':
            return {'loss': loss, 'sample_id': batch['sample_id'], 'pred_LOS': ypred}
        return {'loss': loss}

    def validation_epoch_end(self, outputs) -> None:
        sample_ids = np.concatenate([batch['sample_id'].cpu().numpy() for batch in outputs])
        pred_prob = np.concatenate([batch['pred_LOS'].cpu().numpy().squeeze() for batch in outputs])
        df = pd.DataFrame.from_dict({'sample_id': sample_ids, 'pred_LOS': pred_prob}, orient='columns')
        print_results(df, log_file=os.path.join(self.save_dir, 'metrics.csv'), row_desc=f'val_ep{self.current_epoch}')

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        return batch['sample_id'], self.forward(batch).squeeze(1)

    def test_epoch_end(self, outputs) -> None:
        records = []
        for batch in outputs:
            batch_size = len(batch[0])
            # tuple returned by test_step
            for i in range(batch_size):
                records.append({
                    'sample_id': batch[0][i].cpu().numpy(),
                    'pred_LOS': batch[1][i].cpu().numpy(),
                    })
        df = pd.DataFrame.from_records(records)
        sp = self.save_dir if self.save_dir is not None else SAVE_PATH
        df.to_csv(os.path.join(sp, f'los_test_{self.name}_ep{self.current_epoch}.csv'))
        df = pd.read_csv(os.path.join(sp, f'los_test_{self.name}_ep{self.current_epoch}.csv'), index_col=0)
        if CROSS_VAL is not None:
            print_results(df, name=self.name)
        else:
            print_results(df, log_file=os.path.join(self.save_dir, 'metrics.csv'), row_desc=f'test_ep_{self.current_epoch}')

    def configure_optimizers(self):
        grouped_parameters = [
            #{'params': self.bert_model.pooler.dense.parameters()},
            {'params': self.regression_model.parameters()}
        ]
        if self.use_lstm:
            grouped_parameters.append({'params': self.lstm.parameters()})
        optimizer = torch.optim.Adam(grouped_parameters, lr=self.learning_rate)
        return [optimizer]


class EmbOnDiagsWithRegression(EmbOnDiagsBase):

    def __init__(self, emb_path, lr, name, use_lstm, additionals=0, save_dir=None):
        super(EmbOnDiagsWithRegression, self).__init__(emb_path, lr, name, use_lstm)
        self.save_dir = save_dir
        emb_size = list(self.emb.values())[0].shape[0]
        self.additionals = additionals
        input_dim = 2*emb_size+additionals
        self.regression_model = Regressor(input_dim, emb_size)
        # self.regression_model = torch.nn.Linear(input_dim, 1)
        self.loss_func = torch.nn.MSELoss()

    def forward(self, batch):
        # can use tf-idf by passing weights to embed_diags
        # sample_embeddings = self.embed_diags(batch['diags'], batch['diag_idfs'])
        agg = 'lstm' if self.use_lstm else 'sum'
        sample_prev_diag_embeddings = self.embed_codes(batch['prev_diags'], agg=agg, weights=None)
        sample_primary_diag_embeddings = self.embed_codes(batch['diags'], agg='first', weights=None)
        sample_embeddings = torch.cat([sample_prev_diag_embeddings, sample_primary_diag_embeddings], dim=1)
        if self.additionals > 0:
            sample_embeddings = torch.cat([sample_embeddings, batch['additionals']], dim=1)
        ypred = self.regression_model(sample_embeddings)
        return ypred

    def step(self, batch, name):
        ypred = self.forward(batch).squeeze(1)
        loss = self.loss_func(ypred, batch['los'].to(torch.float32))
        self.log(f'regression/{name}_MSE_loss', loss)
        return {'loss': loss}

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        return batch['sample_id'], self.forward(batch).squeeze(1)

    def test_epoch_end(self, outputs) -> None:
        records = []
        for batch in outputs:
            batch_size = len(batch[0])
            # tuple returned by test_step
            for i in range(batch_size):
                records.append({
                    'sample_id': batch[0][i].cpu().numpy(),
                    'pred_LOS': batch[1][i].cpu().numpy(),
                    })
        df = pd.DataFrame.from_records(records)
        sp = self.save_dir if self.save_dir is not None else SAVE_PATH
        df.to_csv(os.path.join(sp, f'los_test_{self.name}.csv'))
        df = pd.read_csv(os.path.join(sp, f'los_test_{self.name}.csv'), index_col=0)
        print_results(df, log_file=os.path.join(self.save_dir, 'metrics.csv'))

    def configure_optimizers(self):
        grouped_parameters = [
            {'params': self.regression_model.parameters()}
        ]
        if self.use_lstm:
            grouped_parameters.append({'params': self.lstm.parameters()})
        optimizer = torch.optim.Adam(grouped_parameters, lr=self.learning_rate)
        return [optimizer]


if __name__ == '__main__':
    desc, model_path = DESCS_AND_MODELS[RUN_MODEL_INDEX]
    if UPSAMPLE_FEMALE is not None:
        desc = desc + "_random_upsample"
    elif "smote" in LOS_TEST_PATH_V4:
        desc = desc + "_smote"
    elif "medgan" in LOS_TEST_PATH_V4:
        desc = desc + "_medgan"
    elif DOWNSAMPLE_MALE:
        desc = desc + "_downsample"

    frozen_emb = None
    if RUN_MODEL_INDEX == 39:  # null it out uses frozen embeddings
        frozen_emb = os.path.expanduser(
            '~/fairemb/exp_results/nullspace_projection/BERT_tiny_medical_diags_and_drugs_debiased.tsv')

    if TEST_CHECKPOINT is not None:
        save_dir, filename = os.path.split(TEST_CHECKPOINT)
        ep = filename.split("-")[0]
    else:
        next_exp_index = max([int(x.split('_')[-1]) for x in os.listdir(os.path.join(SAVE_PATH, 'LOS'))]) + 1
        save_dir = os.path.join(SAVE_PATH, 'LOS', f'exp_{next_exp_index}')
        os.makedirs(save_dir)
    print(f'results will be written to: {save_dir}')


    ckpt_callback = ModelCheckpoint(dirpath=save_dir, every_n_epochs=1, save_top_k=-1)

    # cui_embeddings = [9, 10, 11, 12]
    # cui_embeddings = [7, 8, 9, 10, 11]
    if RUN_MODEL_INDEX in cui_embeddings:
        embs_with_missing_diags = [DESCS_AND_MODELS[i][1] for i in cui_embeddings]
        dm = LOSbyEmbDiagsModule(embs_with_missing_diags, batch_size=BATCH_SIZE)
        model = EmbOnDiagsWithRegression(model_path, lr=LR, name=desc, use_lstm=True, additionals=48,
                                         save_dir=save_dir)
        # additionals = 93 with all categorical features + non categorical
        # additionals = 48 with GENDER, ETHNICITY +  non categoricals
        # dm = LOSbyEmbDiagsModule(embs_with_missing_diags, batch_size=BATCH_SIZE, classification=True)
        # model = EmbOnDiagsWithClassifier(model_path, lr=LR, name=desc, use_lstm=True, additionals=48,
        #                                  label_field='los', agg_prev_diags=None, agg_diags='first')
        logger = WandbLogger(name=f'{desc}_los', save_dir=LOG_PATH,
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
                             callbacks=[ckpt_callback]
                             )
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)
    else:
        diag_dict = pd.read_csv(os.path.join(MIMIC_PATH, 'D_ICD_DIAGNOSES.csv'), index_col=0)
        diag_dict = diag_dict.set_index('ICD9_CODE')['LONG_TITLE'].to_dict()  # icd9 code to description
        proc_dict = pd.read_csv(os.path.join(MIMIC_PATH, 'D_ICD_PROCEDURES.csv'), index_col=0)
        proc_dict = proc_dict.set_index('ICD9_CODE')['LONG_TITLE'].to_dict()

        if CROSS_VAL is not None:

            results = []
            for i in range(CROSS_VAL):
                model = BertOnDiagsWithRegression(model_path, diag_dict, proc_dict, lr=LR, name=f'{desc}_los_CV{i}', use_lstm=True,
                                                  use_diags=USE_EMB,
                                                  additionals=ADDITIONALS,  # submitted results
                                                  # additionals=93
                                                  # additionals=3
                                                  frozen_emb_file=frozen_emb,
                                                  save_dir=save_dir,
                                                  )
                logger = WandbLogger(name=f'{desc}_los_CV{i}', save_dir=LOG_PATH,
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
                                     callbacks=[ckpt_callback]
                                     )
                dm = LOSbyDiagsModule(diag_dict, proc_dict, batch_size=BATCH_SIZE, upsample_female_file=UPSAMPLE_FEMALE,
                                      downsample_male=DOWNSAMPLE_MALE, CV_fold=i)
                trainer.fit(model, datamodule=dm)
                res = trainer.test(model, datamodule=dm)
                results.append(res)
            # print(results)
            calc_avg_results(os.path.join(SAVE_PATH, "cross_validation", desc+"_los_CV{}_metrics.csv"))

        else:
            model = BertOnDiagsWithRegression(model_path, diag_dict, proc_dict, lr=LR, name=f'{desc}_los',
                                              use_lstm=True,
                                              use_diags=USE_EMB,
                                              #additionals=0,  # submitted results
                                              # additionals=93
                                              additionals=ADDITIONALS,
                                              frozen_emb_file=frozen_emb,
                                              save_dir=save_dir,
                                              )
            logger = WandbLogger(name=f'{desc}_los', save_dir=LOG_PATH,
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
                                 callbacks=[ckpt_callback],
                                 )
            dm = LOSbyDiagsModule(diag_dict, proc_dict, batch_size=BATCH_SIZE, upsample_female_file=UPSAMPLE_FEMALE,
                                  downsample_male=DOWNSAMPLE_MALE)
            if TEST_CHECKPOINT is not None:
                # torch_model = torch.load(TEST_CHECKPOINT)
                # torch_model['state_dict']['bert_model.pooler.dense.bias']
                # torch_model['state_dict']['bert_model.pooler.dense.weight']
                # print(f"testing checkpoint: {TEST_CHECKPOINT}")
                # print('without passing the checkpoint path again to trainer.test')
                # trainer.test(model, datamodule=dm)
                # print('when passing the checkpoint path again to trainer.test')
                model.eval() # did not check if this works/ is needed
                trainer.test(model, datamodule=dm, ckpt_path=TEST_CHECKPOINT)
            else:
                trainer.fit(model, datamodule=dm)
                trainer.test(model, datamodule=dm)

    if TEST_CHECKPOINT is None:
        with open(os.path.join(save_dir, 'log.txt'), 'w') as f:
            f.write(f"Run description: {desc}\n")
            f.write(f"Used features: diags? {USE_EMB}\n")
            f.write(f"Additional features: using {ADDITIONALS}\navailable features: {dm.additional_features_list()}\n")
            f.write(f"Batch size: {BATCH_SIZE}\nlearning rate: {LR}\nepochs: {MAX_EPOCHS}\ncross_val? {CROSS_VAL}\n")

    # join_results([(desc, os.path.join(SAVE_PATH, f'los_test_{desc}.csv')) for desc, path in DESCS_AND_MODELS],
    #              output_fname=os.path.join(SAVE_PATH, 'los_test_2L_with_attrs.csv'))

