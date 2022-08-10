# This script processes MIMIC-III dataset and builds a binary matrix or a count matrix depending on your input.
# The output matrix is a Numpy matrix of type float32, and suitable for training medGAN.
# Written by Edward Choi (mp2893@gatech.edu)
# Usage: Put this script to the folder where MIMIC-III CSV files are located. Then execute the below command.
# python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv <output file> <"binary"|"count">
# Note that the last argument "binary/count" determines whether you want to create a binary matrix or a count matrix.

# Output files
# <output file>.pids: cPickled Python list of unique Patient IDs. Used for intermediate processing
# <output file>.matrix: Numpy float32 matrix. Each row corresponds to a patient. Each column corresponds to a ICD9 diagnosis code.
# <output file>.types: cPickled Python dictionary that maps string diagnosis codes to integer diagnosis codes.

import sys
import _pickle as pickle
import numpy as np
from datetime import datetime
import pandas as pd
from ast import literal_eval


def clean_nans(string_rep_of_seq):
    return string_rep_of_seq.replace('nan, ', '').replace(', nan', '').replace('nan', '')

if __name__ == '__main__':
    admissions_data_file = sys.argv[1]
    #admissionFile = sys.argv[1]
    #diagnosisFile = sys.argv[2]
    outFile = sys.argv[2]
    binary_count = sys.argv[3]

    if binary_count not in ['binary', 'count']:
        print('You must choose either binary or count.')
        sys.exit()

    admissions = pd.read_csv(admissions_data_file, index_col=0)
    # generate more admissions based only on female patients from the train, without patients who dies in the hospital.
    admissions = admissions[(admissions['ASSIGNMENT'] == 'train') &
                            (admissions['GENDER'] == 'F') & 
                            (admissions['READMISSION'] != 2) &
                            (admissions['AGE'] >= 18)]
    print(f"Found {len(admissions)} female admissions with age >= 18 in the train.")

    # mappings from code to matrix column index
    current_diags_to_idx = {}
    idx_to_cur_diag = []
    prev_diags_to_idx = {}
    idx_to_prev_diag = []
    #procs_to_idx = {}
    #idx_to_proc = []
    drugs_to_idx = {}
    idx_to_drug = []
    
    field_to_mapping = {'DIAGS': current_diags_to_idx, 'PREV_DIAGS':prev_diags_to_idx, 'DRUG':drugs_to_idx}
    field_to_reverse_mapping = {'DIAGS': idx_to_cur_diag, 'PREV_DIAGS':idx_to_prev_diag, 'DRUG':idx_to_drug}
    field_to_nan_filler = {'DIAGS': '[]', 'PREV_DIAGS': '{}', 'DRUG': '[]'}
    
    for field in field_to_mapping:
        admissions[field] = admissions[field].fillna(field_to_nan_filler[field]).apply(clean_nans).apply(literal_eval)
    
    for i, row in admissions.iterrows():
        for field in field_to_mapping:
            for item in row[field]:
                if item not in field_to_mapping[field]:
                    field_to_mapping[field][item] = len(field_to_mapping[field])
                    field_to_reverse_mapping[field].append(item)
                

    print('Constructing the matrix')
    # This makes a patients * codes matrix. For our tasks we want a admissions * (current_codes, prev_codes, drugs) matrix.
    num_rows = len(admissions)
    num_cols = sum([len(field_to_mapping[field]) for field in field_to_mapping]) + 1
    print(f"matrix size: {num_rows} rows * {num_cols} columns")
    field_to_first_idx = {'DIAGS': 0, 
                          'PREV_DIAGS': len(field_to_mapping['DIAGS']),
                          'DRUG': len(field_to_mapping['DIAGS'])+ len(field_to_mapping['PREV_DIAGS']),
                          }
    print(field_to_first_idx)
    matrix = np.zeros((num_rows, num_cols)).astype('float32')
    row_index = 0
    for i, row in admissions.iterrows():
        matrix[row_index][num_cols-1] = row['READMISSION']
        for field in field_to_mapping:
            for item in row[field]:
                col_idx = field_to_first_idx[field] + field_to_mapping[field][item]
                if binary_count == 'binary':
                    matrix[row_index][col_idx] = 1.
                else:
                    matrix[row_index][col_idx] += 1.
        row_index += 1

    
    # only the matrix is used in the medgan training/generation. The the other out files are saved to be able to trace back the conditions and admissions.
    pickle.dump(matrix, open(outFile+'.matrix', 'wb'), -1)
    pickle.dump(field_to_mapping, open(outFile+'.map', 'wb'), -1)
    pickle.dump(field_to_reverse_mapping, open(outFile+'.reverse_map', 'wb'), -1)
    pickle.dump(field_to_first_idx, open(outFile+'.first_index', 'wb'), -1)
    pickle.dump(admissions.index, open(outFile+'.admissions_index', 'wb'), -1)

