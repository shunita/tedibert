import sys
import _pickle as pickle
import numpy as np
from datetime import datetime
import pandas as pd
from ast import literal_eval
from imblearn.over_sampling import SMOTE

def clean_nans(string_rep_of_seq):
    return string_rep_of_seq.replace('nan, ', '').replace(', nan', '').replace('nan', '')


def index_to_field(index, field_to_first_idx, readmission_or_los):
    if readmission_or_los == 'readmission':
        if index >= field_to_first_idx['DRUG']:
            return 'DRUG'
        if index >= field_to_first_idx['PREV_DIAGS']:
            return 'PREV_DIAGS'
        return 'DIAGS'
    else:
        if index >= field_to_first_idx['PREV_DIAGS']:
            return 'PREV_DIAGS'
        return 'ICD9_CODE'


def matrix_row_to_record(row, gender, readmission_or_los):
    threshold = 0  # TODO: maybe another threshold?
    if readmission_or_los == 'readmission':
        record = {'DIAGS': [], 'PREV_DIAGS': [], 'DRUG': []}
    else:
        record = {'ICD9_CODE': [], 'PREV_DIAGS': []}
    for col in range(len(row) - 1):  # e.g. col=10
        field = index_to_field(col, field_to_first_idx, readmission_or_los)  # field = 'DIAGS'
        if row[col] > threshold:
            # print(f"field: {field}, col index: {col} first_idx: {field_to_first_idx[field]}")
            item = field_to_reverse_mapping[field][col - field_to_first_idx[field]]  # diags_reverse_mapping[10-0]
            record[field].append(item)
    record['PREV_DIAGS'] = set(record['PREV_DIAGS'])
    if readmission_or_los == 'readmission':
        record['READMISSION'] = 1 if row[-1] > threshold else 0
    else:
        record['LOS'] = row[-1]
    record['GENDER'] = gender
    return record


def combine_sampled_with_original(sampled, readmission_or_los):
    if readmission_or_los == 'readmission':
        orig_data_file = '~/mimic3/custom_tasks/data/stays_readmission_plus_measurements.csv'
        field_to_filler = {'DIAGS': '[]', 'PREV_DIAGS': '{}', 'DRUG': '[]'}
        out_file = '~/fairemb/data/readmission_by_diags_with_smote.csv'
    else:
        orig_data_file = '~/mimic3/custom_tasks/data/los_by_diag_v4.csv'
        field_to_filler = {'ICD9_CODE': '[]', 'PREV_DIAGS': '{}'}
        out_file = '~/fairemb/data/los_by_diags_with_smote.csv'
    df = pd.read_csv(orig_data_file, index_col=0)
    df['source'] = 'orig'
    # planting a fake age so these records will not get filtered out
    sampled['AGE'] = 20
    sampled['source'] = 'smote'
    sampled['ASSIGNMENT'] = 'train'
    sampled.loc[sampled.PREV_DIAGS == 'set()', 'PREV_DIAGS'] = '{}'
    combined = pd.concat((df, sampled))

    for f in field_to_filler:
        combined[f] = combined[f].fillna(field_to_filler[f])
        combined[f] = combined[f].apply(clean_nans)
    combined.to_csv(out_file)


if __name__ == '__main__':
    READ_MATRICES = True
    READ_GENERATED = True
    admissions_data_file = sys.argv[1]
    # admissionFile = sys.argv[1]
    # diagnosisFile = sys.argv[2]
    outFile = sys.argv[2]
    binary_count = sys.argv[3]
    readmission_or_los = sys.argv[4]

    if binary_count not in ['binary', 'count']:
        print('You must choose either binary or count.')
        sys.exit()

    if not READ_MATRICES:

        admissions = pd.read_csv(admissions_data_file, index_col=0)
        # generate more admissions based only on patients from the train
        admissions = admissions[(admissions['ASSIGNMENT'] == 'train') &
                                (admissions['AGE'] >= 18)]
        # For readmission, exclude patients who died in the hostpital
        if readmission_or_los == "readmission":
            admissions = admissions[admissions['READMISSION'] != 2]
        print(f"Found {len(admissions)} admissions with age >= 18 in the train.")

        # mappings from code to matrix column index
        if readmission_or_los == 'readmission':
            current_diags_to_idx = {}
            idx_to_cur_diag = []
            prev_diags_to_idx = {}
            idx_to_prev_diag = []
            # procs_to_idx = {}
            # idx_to_proc = []
            drugs_to_idx = {}
            idx_to_drug = []

            field_to_mapping = {'DIAGS': current_diags_to_idx, 'PREV_DIAGS': prev_diags_to_idx, 'DRUG': drugs_to_idx}
            field_to_reverse_mapping = {'DIAGS': idx_to_cur_diag, 'PREV_DIAGS': idx_to_prev_diag, 'DRUG': idx_to_drug}
            field_to_nan_filler = {'DIAGS': '[]', 'PREV_DIAGS': '{}', 'DRUG': '[]'}
        else:
            primary_diag_to_idx = {}
            idx_to_primary_diag = []
            prev_diags_to_idx = {}
            idx_to_prev_diag = []

            field_to_mapping = {'ICD9_CODE': primary_diag_to_idx, 'PREV_DIAGS': prev_diags_to_idx}
            field_to_reverse_mapping = {'ICD9_CODE': idx_to_primary_diag, 'PREV_DIAGS': idx_to_prev_diag}
            field_to_nan_filler = {'ICD9_CODE': '[]', 'PREV_DIAGS': '{}'}

        for field in field_to_mapping:
            admissions[field] = admissions[field].fillna(field_to_nan_filler[field]).apply(clean_nans).apply(literal_eval)

        for i, row in admissions.iterrows():
            for field in field_to_mapping:
                for item in row[field]:
                    if item not in field_to_mapping[field]:
                        field_to_mapping[field][item] = len(field_to_mapping[field])
                        field_to_reverse_mapping[field].append(item)
                    if readmission_or_los == 'los' and field == 'ICD9_code': # take only the first diagnosis (primary)
                        break

        print('Constructing the matrix')
        # This makes a patients * codes matrix. For our tasks we want a admissions * (current_codes, prev_codes, drugs) matrix.
        num_rows = len(admissions)
        num_cols = sum([len(field_to_mapping[field]) for field in field_to_mapping]) + 1
        print(f"matrix size: {num_rows} rows * {num_cols} columns")
        if readmission_or_los == 'readmission':
            field_to_first_idx = {'DIAGS': 0,
                              'PREV_DIAGS': len(field_to_mapping['DIAGS']),
                              'DRUG': len(field_to_mapping['DIAGS']) + len(field_to_mapping['PREV_DIAGS']),
                              }
        else:
            field_to_first_idx = {'ICD9_CODE': 0,
                                  'PREV_DIAGS': len(field_to_mapping['ICD9_CODE']),
                                  }
        print(field_to_first_idx)
        matrix = np.zeros((num_rows, num_cols)).astype('float32')
        row_index = 0
        for i, row in admissions.iterrows():
            if readmission_or_los == 'readmission':
                matrix[row_index][num_cols - 1] = row['READMISSION']
            else:
                matrix[row_index][num_cols - 1] = row['LOS']
            for field in field_to_mapping:
                for item in row[field]:
                    col_idx = field_to_first_idx[field] + field_to_mapping[field][item]
                    if binary_count == 'binary':
                        matrix[row_index][col_idx] = 1.
                    else:
                        matrix[row_index][col_idx] += 1.
            row_index += 1
        print("Constructing gender labels")
        y = admissions['GENDER']

        print("sampling new records with SMOTE")
        smt = SMOTE(random_state=0)
        X_train_SMOTE, y_train_SMOTE = smt.fit_resample(matrix, y)
        print(f"size of matrix: {matrix.shape}, y: {y.shape}")
        print(f"size of sampled matrix after smote: {X_train_SMOTE.shape}, y: {y_train_SMOTE.shape}")
        print(f"first rows equal? {np.all(X_train_SMOTE[:matrix.shape[0]] == matrix)}")
        print(f"orig label count: {y.value_counts()}")
        print(f"new matrix first {len(y)} rows labels count: {pd.Series(y_train_SMOTE[:len(y)]).value_counts()}")
        print(f"smote label distribution: {pd.Series(y_train_SMOTE[-5014:]).value_counts()} ")

        # only the matrix is used in the medgan training/generation. The the other out files are saved to be able to trace back the conditions and admissions.
        pickle.dump(matrix, open(outFile + '.orig_matrix', 'wb'), -1)
        pickle.dump(X_train_SMOTE, open(outFile + '.smote_matrix', 'wb'), -1)
        pickle.dump(y_train_SMOTE, open(outFile + '.smote_labels', 'wb'), -1)
        pickle.dump(field_to_mapping, open(outFile + '.map', 'wb'), -1)
        pickle.dump(field_to_reverse_mapping, open(outFile + '.reverse_map', 'wb'), -1)
        pickle.dump(field_to_first_idx, open(outFile + '.first_index', 'wb'), -1)
        pickle.dump(admissions.index, open(outFile + '.admissions_index', 'wb'), -1)

    if READ_MATRICES:
        matrix = pickle.load(open(outFile + '.orig_matrix', 'rb'))
        X_train_SMOTE = pickle.load(open(outFile + '.smote_matrix', 'rb'))
        y_train_SMOTE = pickle.load(open(outFile + '.smote_labels', 'rb'))
        field_to_mapping = pickle.load(open(outFile + '.map', 'rb'))
        field_to_reverse_mapping = pickle.load(open(outFile + '.reverse_map', 'rb'))
        field_to_first_idx = pickle.load(open(outFile + '.first_index', 'rb'))

    if not READ_GENERATED:
        # transform back to df
        # Assuming that the last records are the generated ones.
        # make records out of the generated matrix.
        records = []

        for i in range(matrix.shape[0], X_train_SMOTE.shape[0]):
            row = X_train_SMOTE[i]
            record = matrix_row_to_record(row, y_train_SMOTE[i], readmission_or_los)
            records.append(record)

        df = pd.DataFrame.from_records(records)
        print(f"new df has {len(records)} records")
        if readmission_or_los == 'los':
            df = df[df.LOS > 0]
            print(f"removed {len(records) - len(df)} records with negative LOS")
        df.to_csv(outFile + '_generated_by_smote.csv')

    if READ_GENERATED:
        df = pd.read_csv(outFile + '_generated_by_smote.csv')
        combine_sampled_with_original(df, readmission_or_los)



