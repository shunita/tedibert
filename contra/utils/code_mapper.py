from collections import defaultdict

import pandas as pd
import os
from contra.constants import DATA_PATH
from ast import literal_eval

import numpy as np


class CodeMapper(object):
    def __init__(self):
        df_cuis = pd.read_csv(os.path.join(DATA_PATH, 'cui_counts_in_abstracts_sent_sep.csv'))
        self.cui_to_appearances = df_cuis.set_index('cui')['count'].to_dict()
        icd9tocui = defaultdict(set)
        df1 = pd.read_csv(os.path.join(DATA_PATH, 'code_mappings', 'icd9merge_with_cuis.csv'), index_col=0)
        df1['cuis'] = df1['cuis'].apply(literal_eval)

        for i, r in df1.iterrows():
            for cui in r['cuis']:
                icd9tocui[str(r['ICD9'])].add(cui)

        df2 = pd.read_csv(os.path.join(DATA_PATH, 'code_mappings', 'cui_to_icd9_manual.csv'), index_col=0)
        df2['ICD9'] = df2['ICD9'].apply(literal_eval)
        for i, r in df2.iterrows():
            for icd9 in r['ICD9']:
                icd9tocui[icd9].add(r['CUI'])

        self.icd9nodots = defaultdict(set)
        for icd9 in icd9tocui:
            nodots = icd9.replace('.', '')
            self.icd9nodots[nodots] = self.icd9nodots[nodots].union(icd9tocui[icd9])

    def __getitem__(self, icd9):
        return self.icd9nodots[icd9.replace('.', '')]

    def __contains__(self, key):
        return key.replace('.', '') in self.icd9nodots

    def get_cui_appearances(self, cui):
        if cui in self.cui_to_appearances:
            return self.cui_to_appearances[cui]
        return 0


def count_matched(list_of_diags, icd_to_cui):
    c = 0
    for d in set(list_of_diags):
        if d in icd_to_cui:
            c += 1
    return c


def string_to_array(s):
    if s.startswith('['):
        parts = s.strip('[]').split()
    else:
        parts = s.split(",")
    return np.array([float(x) for x in parts])


def read_emb(path):
    # path = os.path.join(DATA_PATH, 'embs', 'fem40_heur_emb.tsv')
    # df = pd.read_csv(LOS_TEST_PATH, index_col=0)
    # df['count_matched'] = df['PREV_DIAGS'].apply(count_matched)
    # df['num_diags'] = df.PREV_DIAGS.apply(lambda x: len(set(x)))
    emb = pd.read_csv(path, sep='\t',  header=None, index_col=0, names=['cui', 'vector'])
    emb['vector'] = emb['vector'].apply(string_to_array)
    emb = emb.to_dict(orient='dict')['vector']
    return emb


def count_got_emb(list_of_diags, emb, icd_to_cuis):
    count = 0
    for diag in set(list_of_diags):
        if diag not in icd_to_cuis:
            continue
        cuis = icd_to_cuis[diag]
        for cui in cuis:
            if cui in emb:
                c += 1
                break
    return count