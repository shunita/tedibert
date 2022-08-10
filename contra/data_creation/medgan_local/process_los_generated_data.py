import sys
import numpy as np
import pickle
import pandas as pd

ORIGINAL_DATA = '~/mimic3/custom_tasks/data/los_by_diag_v4.csv'

def index_to_field(index, field_to_first_idx):
    if index >= field_to_first_idx['PREV_DIAGS']:
        return 'PREV_DIAGS'
    return 'ICD9_CODE'
    
def clean_nans(string_rep_of_seq):
    return string_rep_of_seq.replace('nan, ', '').replace(', nan', '').replace('nan', '')
    

def combine_sampled_with_original(sampled):
    df = pd.read_csv(ORIGINAL_DATA, index_col=0)
    df['source'] = 'orig'
    # planting a fake age so these records will not get filtered out
    sampled['AGE'] = 20
    sampled['source'] = 'medgan'
    sampled['ASSIGNMENT'] = 'train'
    sampled.loc[sampled.PREV_DIAGS == 'set()', 'PREV_DIAGS'] = '{}'
    combined = pd.concat((df, sampled))
    field_to_filler = {'ICD9_CODE': '[]', 'PREV_DIAGS':'{}'}
    for f in field_to_filler:
        combined[f] = combined[f].fillna(field_to_filler[f])
        combined[f] = combined[f].apply(clean_nans)
    combined.to_csv('~/fairemb/data/los_by_diags_sampled_medgan.csv')

if __name__ == '__main__':
    generated_los_data_file = sys.argv[1]
    metadata_path = sys.argv[2]
    out_file = sys.argv[3]
    # number of male - female admission rows in train
    df = pd.read_csv(ORIGINAL_DATA, index_col=0)
    train = df[(df['ASSIGNMENT'] == 'train') &
               (df['AGE'] >= 18)]
    n_sample = len(train[train['GENDER'] == 'M']) - len(train[train['GENDER'] == 'F'])
    print(f"sampling {len(train[train['GENDER'] == 'M'])}-{len(train[train['GENDER'] == 'F'])}={n_sample} new train records.")
    
    mat = np.load(generated_los_data_file)
    
    field_to_mapping = pickle.load(open(metadata_path+'.map', 'rb'))
    field_to_reverse_mapping = pickle.load(open(metadata_path+'.reverse_map', 'rb'))
    field_to_first_idx = pickle.load(open(metadata_path+'.first_index', 'rb'))
    
    # make records out of the generated matrix.
    records = []
    # TODO: maybe this needs another threshold??
    threshold = 0
    i=0
    while len(records) < n_sample and i < mat.shape[0]:
    #for i in range(min(mat.shape[0], n_sample)):
        row = mat[i]
        los = row[-1]
        if los < 0:
            i += 1
            continue
        record = {'ICD9_CODE': [], 'PREV_DIAGS': []}
        for col in range(len(row)-1): # e.g. col=10
            field = index_to_field(col, field_to_first_idx) # field = 'ICD9_CODE'
            if row[col] > threshold: 
                #print(f"field: {field}, col index: {col} first_idx: {field_to_first_idx[field]}")
                item = field_to_reverse_mapping[field][col - field_to_first_idx[field]] # diags_reverse_mapping[10-0]
                record[field].append(item)
        if len(record['ICD9_CODE']) == 0:
            i += 1
            continue
        record['PREV_DIAGS'] = set(record['PREV_DIAGS'])
        record['LOS'] = los
        records.append(record)
        i += 1
        
    df = pd.DataFrame.from_records(records)
    print
    print(df.head())
    print(f"los mean in generated records: {df.LOS.mean()}, range: {df.LOS.min()}-{df.LOS.max()}")
    df.to_csv(out_file+'.csv')
    df = pd.read_csv(out_file + '.csv', index_col=0)
    combine_sampled_with_original(df)
    
        
     