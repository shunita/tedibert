import sys
import numpy as np
import pickle
import pandas as pd

def index_to_field(index, field_to_first_idx):
    if index >= field_to_first_idx['DRUG']:
        return 'DRUG'
    if index >= field_to_first_idx['PREV_DIAGS']:
        return 'PREV_DIAGS'
    return 'DIAGS'
    
def clean_nans(string_rep_of_seq):
    return string_rep_of_seq.replace('nan, ', '').replace(', nan', '').replace('nan', '')
    

def combine_sampled_with_original(sampled):
    df = pd.read_csv('~/mimic3/custom_tasks/data/stays_readmission_plus_measurements.csv', index_col=0)
    df['source'] = 'orig'
    # planting a fake age so these records will not get filtered out
    sampled['AGE'] = 20
    sampled['source'] = 'medgan'
    sampled['ASSIGNMENT'] = 'train'
    sampled.loc[sampled.PREV_DIAGS == 'set()', 'PREV_DIAGS'] = '{}'
    combined = pd.concat((df, sampled))
    field_to_filler = {'DIAGS': '[]', 'PREV_DIAGS':'{}', 'DRUG':'[]'}
    for f in field_to_filler:
        combined[f] = combined[f].fillna(field_to_filler[f])
        combined[f] = combined[f].apply(clean_nans)
    combined.to_csv('~/fairemb/data/readmission_by_diags_sampled_medgan_v2.csv')

if __name__ == '__main__':
    generated_admissions_data_file = sys.argv[1]
    metadata_path = sys.argv[2]
    out_file = sys.argv[3]
    # number of male - female admission rows in train (after removing those who died in the hospital)
    df = pd.read_csv('~/mimic3/custom_tasks/data/stays_readmission_plus_measurements.csv', index_col=0)
    train = df[(df['ASSIGNMENT'] == 'train') &
               (df['READMISSION'] != 2) &
               (df['AGE'] >= 18)]
    n_sample = len(train[train['GENDER'] == 'M']) - len(train[train['GENDER'] == 'F'])
    print(f"sampling {len(train[train['GENDER'] == 'M'])}-{len(train[train['GENDER'] == 'F'])}={n_sample} new train records.")
    
    mat = np.load(generated_admissions_data_file)
    
    field_to_mapping = pickle.load(open(metadata_path+'.map', 'rb'))
    field_to_reverse_mapping = pickle.load(open(metadata_path+'.reverse_map', 'rb'))
    field_to_first_idx = pickle.load(open(metadata_path+'.first_index', 'rb'))
    
    # make records out of the generated matrix.
    records = []
    # TODO: maybe this needs another threshold??
    threshold = 0
    for i in range(min(mat.shape[0], n_sample)):
        row = mat[i]
        record = {'DIAGS': [], 'PREV_DIAGS': [], 'DRUG': []}
        for col in range(len(row)-1): # e.g. col=10
            field = index_to_field(col, field_to_first_idx) # field = 'DIAGS'
            if row[col] > threshold: 
                #print(f"field: {field}, col index: {col} first_idx: {field_to_first_idx[field]}")
                item = field_to_reverse_mapping[field][col - field_to_first_idx[field]] # diags_reverse_mapping[10-0]
                record[field].append(item)
        record['PREV_DIAGS'] = set(record['PREV_DIAGS'])
        record['READMISSION'] = 1 if row[-1] > threshold else 0
        records.append(record)
        
    df = pd.DataFrame.from_records(records)
    print(df.head())
    df.to_csv(out_file+'.csv')
    combine_sampled_with_original(df)
    
        
     