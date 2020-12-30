import os
import pandas as pd
from contra.constants import FULL_PUMBED_2019_PATH

def read_year(path_or_year):
    path = path_or_year
    if type(path_or_year) == int:  # it's a year
        path = os.path.join(FULL_PUMBED_2019_PATH, f'pubmed_{path_or_year}.csv')
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    df = pd.read_csv(path, index_col=0)
    df = df.dropna(subset=['date'], axis=0)
    df['title'] = df['title'].fillna('')
    df['title'] = df['title'].apply(lambda x: x.strip('[]'))
    df['title_and_abstract'] = df['title'] + df['abstract']
    return df
