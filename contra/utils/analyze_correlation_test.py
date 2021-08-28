import os
import pandas as pd
from contra.constants import SAVE_PATH


def merge_test_results_with_cui_names(test_file='test_similarities_CUI_names_bert_pos_fp_2020_2020.csv',
                                      results1_file='test_similarities_GAN_lmb0.5.csv',
                                      results1_desc='gan0.5',
                                      results2_file='test_similarities_GAN_lmb0.0.csv',
                                      results2_desc='no_gan',
                                      output_file='exp_results/compare_gan_with_baseline_on_test.csv'):
    df_true = pd.read_csv(os.path.join(SAVE_PATH, test_file), index_col=0)
    df_true = df_true.rename({'similarity': 'sim2020'}, axis=1)
    df_gan = pd.read_csv(os.path.join(SAVE_PATH, results1_file), index_col=0)
    df_gan = df_gan.rename({'pred_similarity': results1_desc}, axis=1)
    df_nogan = pd.read_csv(os.path.join(SAVE_PATH, results2_file), index_col=0)
    df_nogan = df_nogan.rename({'pred_similarity': results2_desc}, axis=1)
    df = df_true.merge(df_gan['gan0.5'], left_index=True, right_index=True)
    df = df.merge(df_nogan['no_gan'], left_index=True, right_index=True)
    df.to_csv(output_file)
