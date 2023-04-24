import os
import numpy as np

from contra.utils.delong_auc import delong_roc_test

np.random.seed(1234)
rng=np.random.RandomState(1234)
import pandas as pd
import scipy as sp
from sklearn.metrics import roc_auc_score
from contra.utils.diebold_mariano import dm_test
# from contra.tests.test_readmission_by_diags import delong_on_df


# LOS_PATH = r'C:\Users\shunita\OneDrive - Technion\kira\fairemb\exp_results\paper 2 los\age gender tiny bert'
# main_los_file = 'merged_with_upsample_raw_test_data.csv'
LOS_PATH = r'C:\Users\shunita\OneDrive - Technion\kira\fairemb\exp_results\paper 2 los\tiny bert no additionals'
main_los_file = 'merged_orig_and_upsample.csv'


# READMISSION_PATH = r'C:\Users\shunita\OneDrive - Technion\kira\fairemb\exp_results\paper 2 readmission\tiny bert no additionals'
# main_readmission_file = 'merged_upsample_with_original.csv'
READMISSION_PATH = r'C:\Users\shunita\OneDrive - Technion\kira\fairemb\exp_results\paper 2 readmission\age gender tiny bert'
main_readmission_file = 'stays_readmission_plus_measurements.csv'

def calculate_diebold_mariano():
    # main_los_file = 'merged_with_upsample_raw_test_data.csv'
    test_df = pd.read_csv(os.path.join(LOS_PATH, main_los_file), index_col=0).rename(
        lambda x: x.replace('GAN20', 'DERT').replace('BERT10-18_40eps', 'BERT10-18'), axis=1)
    los_files = {
        'BERT10-18_Medgan': 'los_test_BERT10-18_40eps_medgan_los.csv',
        'DERT_Medgan': 'los_test_GAN20_medgan_los.csv',
        'DERT_SMOTE': 'los_test_GAN20_smote_los.csv',
        'BERT10-18_SMOTE': 'los_test_BERT10-18_40eps_smote_los.csv',
        'nullitout': 'los_test_Medical_tiny_BERT_nullitout_los.csv'
    }
    # pairs = [('DERT' + x, 'BERT10-18' + x) for x in ['', '_upsample', '_Medgan', '_SMOTE']]

    models = ['DERT', 'BERT10-18', 'tinybert_non_medical', 'nullitout']
    pairs = [(models[i], models[j]) for i in range(len(models)) for j in range(len(models)) if i < j]

    for pair in pairs:
        print("\n")
        print(f"Compare {pair[0]} with {pair[1]}")
        for population in ['F', 'M', 'all']:
            print(population, end=': ')
            if population == 'all':
                subset = test_df
            else:
                subset = test_df[test_df['GENDER'] == population]
            cols = ['DERT', 'BERT10-18', 'tinybert_non_medical']
            m = subset[['LOS', 'GENDER', 'sample_id'] + cols + [x+'_loss' for x in cols]]
            for model in pair:
                if model not in subset.columns:
                    df1 = pd.read_csv(os.path.join(LOS_PATH, los_files[model]), index_col=0).rename(
                        {'pred_LOS': model}, axis=1)
                    m = m.merge(df1, on='sample_id')
                    m[model + "_loss"] = (m[model] - m['LOS']).abs()
                print(f"{model} MAE: {m[model + '_loss'].mean()}")
            print(dm_test(m['LOS'], m[pair[0]], m[pair[1]]))


def merge_los_files():
    # merge all the individual result files into one
    # main_los_file = 'merged_with_upsample_raw_test_data.csv'
    # main_los_file = 'merged_orig_and_upsample.csv'

    df = pd.read_csv(os.path.join(LOS_PATH, main_los_file), index_col=0)
    # df = df.rename(lambda x: x.replace('GAN20', 'DERT'), axis=1)
    # df = df.rename(lambda x: x.replace('BERT10-18_40eps', 'BERT10-18'), axis=1)
    # df = df.rename(lambda x: x.replace('tinybert_non_medical', 'tinybert_non_med'), axis=1)
    los_files = {
        'BERT10-18_Medgan': 'los_test_BERT10-18_40eps_medgan_los.csv',
        'DERT_Medgan': 'los_test_GAN20_medgan_los.csv',
        'DERT_SMOTE': 'los_test_GAN20_smote_los.csv',
        'BERT10-18_SMOTE': 'los_test_BERT10-18_40eps_smote_los.csv',
        'nullitout': 'los_test_Medical_tiny_BERT_nullitout_los.csv'
        }
    print(f"main file has {len(df)} rows")
    for name, file_path in los_files.items():
        df1 = pd.read_csv(os.path.join(LOS_PATH, file_path), index_col=0)
        print(f"merging: {name}, {len(df1)} rows")
        df1 = df1.rename({'pred_LOS': name}, axis=1)
        df = df.merge(df1, on='sample_id')
        df[f'{name}_loss'] = (df[f'{name}']-df['LOS']).abs()

    # df.to_csv(os.path.join(los_path, 'merged_with_upsample_raw_test_data_4upsampling_methods.csv'))
    return df


def calculate_mae_for_test_file(path):
    # los_path = r'C:\Users\shunita\OneDrive - Technion\kira\fairemb\exp_results\paper 2 los\age gender tiny bert'
    # main_los_file = 'merged_with_upsample_raw_test_data.csv'
    # los_path = r'C:\Users\shunita\OneDrive - Technion\kira\fairemb\exp_results\paper 2 los\tiny bert no additionals'
    df = pd.read_csv(os.path.join(LOS_PATH, main_los_file), index_col=0)
    df1 = pd.read_csv(os.path.join(LOS_PATH, path), index_col=0)
    print(f"main los file has {len(df)} rows, df1 has {len(df1)} rows.")
    df = df[['sample_id', 'LOS', 'GENDER']].merge(df1, on='sample_id')
    df['loss'] = (df['pred_LOS'] - df['LOS']).abs()
    print(f"MAE: {df['loss'].mean()}, female MAE: {df[df['GENDER'] == 'F']['loss'].mean()}, male MAE: {df[df['GENDER'] == 'M']['loss'].mean()}")


def bootstrap_MAE(df):
    # models = ['DERT_loss', 'BERT10-18_loss', 'DERT_upsample_loss', 'BERT10-18_upsample_loss',
    #           'DERT_Medgan_loss', 'BERT10-18_Medgan_loss', 'DERT_SMOTE_loss', 'BERT10-18_SMOTE_loss']
    models = ['DERT_loss', 'BERT10-18_loss', 'tinybert_non_med_loss', 'nullitout_loss']
    for model in models:
        for population in ['all', 'F', 'M']:
        # for population in ['F', 'M']:
            print(f"{model}, {population} MAE: mean, Negative error, Positive error")
            if population == 'all':
                subset = df
            else:
                subset = df[df.GENDER == population]
            data = subset[model].values
            res = sp.stats.bootstrap((data,), np.mean, n_resamples=2000, confidence_level=0.95)
            m = np.mean(data)
            bm = np.mean(res.bootstrap_distribution)
            print(f"mean: {m} vs. bootstrap mean: {bm}")
            print("{0:.4f}".format(bm), end=",")
            print("{0:.4f}".format(bm-res.confidence_interval.low), end=",")
            print("{0:.4f}".format(res.confidence_interval.high - bm))


def merge_readmission_files():
    # merge all the individual result files into one
    df = pd.read_csv(os.path.join(READMISSION_PATH, main_readmission_file), index_col=0)
    df = df.rename(lambda x: x.replace('GAN20', 'DERT').replace('BERT10-18_40eps', 'BERT10-18'), axis=1)
    readmission_files = {#'BERT10-18_Medgan': 'readmission_test_BERT10-18_40eps_cls_diags_drugs_lstm_2L_medgan.csv',
                         'BERT10-18_Medgan': 'readmission_test_BERT10-18_40eps_medgan_cls_diags_drugs_lstm_2L.csv',
                         'DERT_Medgan': 'readmission_test_GAN20_cls_diags_drugs_lstm_2L_medgan.csv',
                         'DERT_SMOTE': 'readmission_test_GAN20_cls_diags_drugs_lstm_2L_smote.csv',
                         #'BERT10-18_SMOTE': 'readmission_test_BERT10-18_40eps_cls_diags_drugs_lstm_2L_smote.csv',
                         'BERT10-18_SMOTE': 'readmission_test_BERT10-18_40eps_smote_cls_diags_drugs_lstm_2L.csv',
                         'Null it out': 'readmission_test_Medical_tiny_BERT_nullitout_cls_diags_drugs_lstm_2L.csv',
                         'BERT10-18_downsample': 'readmission_test_BERT10-18_40eps_downsample_cls_diags_drugs_lstm_2L.csv',
                         'DERT_downsample': 'readmission_test_GAN20_downsample_cls_diags_drugs_lstm_2L.csv',
                         'DERT': 'readmission_test_GAN20_cls_diags_drugs_lstm_2L.csv',
                         'tinybert_non_medical': 'readmission_test_tinybert_non_medical_cls_diags_drugs_lstm_2L.csv',
                         'BERT10-18': 'readmission_test_BERT10-18_40eps_cls_diags_drugs_lstm_2L.csv',
                        }
    print(f"main file has {len(df)} rows")
    for name, file_path in readmission_files.items():
        full_path = os.path.join(READMISSION_PATH, file_path)
        if not os.path.exists(full_path):
            print(f"{name} not found. skipping")
            continue
        df1 = pd.read_csv(full_path, index_col=0)
        print(f"read: {name} with {len(df1)} rows.")
        df1 = df1.rename({'pred_prob': name}, axis=1)
        if 'sample_id' in df.columns:
            df = df.merge(df1, on='sample_id')
        else:
            df = df.merge(df1, right_on='sample_id', left_index=True)
    # df.to_csv(os.path.join(readmission_path, 'readmission_merged_with_upsample_raw_test_data_4upsampling_methods.csv'))
    return df




def get_ci_auc(y_true, y_pred):
    from scipy.stats import sem
    n_bootstraps = 2000
    bootstrapped_scores = []

    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))

        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # 90% c.i.
    # confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    # confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]

    # 95% c.i.
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    return confidence_lower, confidence_upper, np.mean(sorted_scores)


def bootstrap_AUC(df):
    models = ['DERT', 'BERT10-18',
              'tinybert_non_medical',
              'Null it out',
              # 'DERT_upsample', 'BERT10-18_upsample',
              # 'DERT_Medgan',
              'BERT10-18_Medgan',
              # 'DERT_SMOTE',
              'BERT10-18_SMOTE',
              'BERT10-18_downsample',
              # 'DERT_downsample',
              ]
    for model in models:
        print(f"model: {model}")
        for population in ['all', 'F', 'M']:
        # for population in ['F', 'M']:
            if population == 'all':
                subset = df
            else:
                subset = df[df.GENDER == population]
            # print(f"{model}, {population} AUC: mean, CI low, CI high")
            label = subset['READMISSION'].values
            data = subset[model].values
            #res = sp.stats.bootstrap((label, data), roc_auc_score, n_resamples=2000, confidence_level=0.95)
            lower, upper, bm = get_ci_auc(label, data)
            m = roc_auc_score(label, data)
            # print(f"Mean {m} vs. bootstrap mean: {bm}")
            print("{0:.4f}".format(bm), end=",")
            print("{0:.4f}".format(bm - lower), end=",")
            print("{0:.4f}".format(upper - bm), end=',')
        print("\n", end="")


def delong_on_df(df, true_field, pred1_field, pred2_field):
    df1 = df[(df[pred1_field] != 0) & (df[pred1_field] != 1) & (df[pred2_field] != 0) & (df[pred2_field] != 1)]
    print(f"dropped {len(df)-len(df1)} rows because of extreme values")
    return 10**delong_roc_test(df1[true_field], df1[pred1_field], df1[pred2_field])


def calculate_delong(df):
    models = ['DERT', 'BERT10-18',
              'tinybert_non_medical',
              # 'DERT_upsample', 'BERT10-18_upsample',
              # 'DERT_Medgan', 'BERT10-18_Medgan',
              # 'DERT_SMOTE', 'BERT10-18_SMOTE',
              'BERT10-18_downsample', 'DERT_downsample',
              'Null it out']
    readmission_files = {'BERT10-18_Medgan': 'readmission_test_BERT10-18_40eps_cls_diags_drugs_lstm_2L_medgan.csv',
                         'DERT_Medgan': 'readmission_test_GAN20_cls_diags_drugs_lstm_2L_medgan.csv',
                         'DERT_SMOTE': 'readmission_test_GAN20_cls_diags_drugs_lstm_2L_smote.csv',
                         'BERT10-18_SMOTE': 'readmission_test_BERT10-18_40eps_cls_diags_drugs_lstm_2L_smote.csv',
                         'Null it out': 'readmission_test_Medical_tiny_BERT_nullitout_cls_diags_drugs_lstm_2L.csv',
                         'BERT10-18_downsample': 'readmission_test_BERT10-18_40eps_downsample_cls_diags_drugs_lstm_2L.csv',
                         'DERT_downsample': 'readmission_test_GAN20_downsample_cls_diags_drugs_lstm_2L.csv',
                        }
    #pairs = [('DERT' + x, 'BERT10-18' + x) for x in ['', '_upsample', '_Medgan', '_SMOTE']]
    pairs = [('DERT', 'BERT10-18'), ('DERT', 'tinybert_non_medical'), ('DERT', 'Null it out'),
             ('BERT10-18', 'tinybert_non_medical'), ('BERT10-18', 'Null it out'), ('tinybert_non_medical', 'Null it out')]
    for pair in pairs:
        print(f"Compare {pair[0]} with {pair[1]}")
        for population in ['F', 'M', 'all']:
            print(population, end=': \n')
            if population == 'all':
                subset = df
            else:
                subset = df[df['GENDER'] == population]
            if pair[0] in subset.columns and pair[1] in subset.columns:
                print(delong_on_df(df, 'READMISSION', pair[0], pair[1]))
                #print(dm_test(subset['LOS'], subset[pair[0]], subset[pair[1]]))
            else:
                df1 = pd.read_csv(os.path.join(READMISSION_PATH, readmission_files[pair[0]]), index_col=0).rename({'pred_prob': pair[0]}, axis=1)
                df2 = pd.read_csv(os.path.join(READMISSION_PATH, readmission_files[pair[1]]), index_col=0).rename({'pred_prob': pair[1]}, axis=1)
                m = subset[['READMISSION', 'GENDER', 'sample_id']].merge(df1, on='sample_id').merge(df2, on='sample_id')
                for i in [0, 1]:
                    print(f"{pair[i]} AUC: {roc_auc_score(m['READMISSION'], m[pair[i]])}")
                print(delong_on_df(m, 'READMISSION', pair[0], pair[1]))