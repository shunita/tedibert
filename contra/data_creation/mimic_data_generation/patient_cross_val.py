import pandas as pd
import os
from contra.constants import DATA_PATH
from sklearn.model_selection import ShuffleSplit

rdf = pd.read_csv(os.path.join(DATA_PATH, 'readmission_by_diags_sampled_medgan_v2.csv'), index_col=0)
rdf = rdf[rdf.source == 'orig']

ldf = pd.read_csv(os.path.join(DATA_PATH, 'los_by_diags_sampled_medgan.csv'), index_col=0)
ldf = ldf[ldf.source == 'orig']

lpatients = ldf.SUBJECT_ID.unique()
rpatients = rdf.SUBJECT_ID.unique()
all_patients = list(set(lpatients).union(set(rpatients)))
cv = ShuffleSplit(n_splits=10, test_size=0.2)
splits = cv.split(all_patients)

for i, (train, test) in enumerate(splits):
    print(i)
    # TODO: this is the problem! we used indexes in the patients array instead of patient ids. It caused the train to be too small in each fold.
    train_patients = [all_patients[j] for j in train]
    rdf.loc[rdf.SUBJECT_ID.apply(lambda x: x in train_patients), f'ASSIGNMENT_{i}'] = 'train'
    rdf.loc[rdf[f'ASSIGNMENT_{i}'].isna(), f'ASSIGNMENT_{i}'] = 'test'
    ldf.loc[ldf.SUBJECT_ID.apply(lambda x: x in train_patients), f'ASSIGNMENT_{i}'] = 'train'
    ldf.loc[ldf[f'ASSIGNMENT_{i}'].isna(), f'ASSIGNMENT_{i}'] = 'test'

rdf.to_csv(os.path.join(DATA_PATH, 'readmission_by_diags_10CV.csv'))
ldf.to_csv(os.path.join(DATA_PATH, 'los_by_diags_10CV.csv'))
