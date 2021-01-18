import os
import pandas as pd
import pickle

import sys
sys.path.append('/home/shunita/fairemb')
from contra.constants import FULL_PUMBED_2019_PATH, FULL_PUMBED_2020_PATH, DATA_PATH
DEFAULT_PUBMED_VERSION=2019

def read_year(path_or_year, version=DEFAULT_PUBMED_VERSION, subsample=False):
    path = path_or_year
    year = -1
    if type(path_or_year) == int:  # it's a year
        folder = {2019: FULL_PUMBED_2019_PATH, 2020: FULL_PUMBED_2020_PATH}[version]    
        path = os.path.join(folder, f'pubmed_{path_or_year}.csv')
        year = path_or_year
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    # Don't read 'labels', 'pub_types', 'mesh_headings', 'keywords'
    # Columns: [1 'title',2 'abstract',3 'labels',4 'pub_types',5 'date',6 'file',
    #           7 'mesh_headings',8 'keywords',9 'ncts',10 'year']
    df = pd.read_csv(path, dtype={'mesh_headings':'object'}).rename({'Unnamed: 0': 'PMID'}, axis=1)
    df = df.drop_duplicates(subset=['PMID'])
    df = df.set_index('PMID')
    df = df.drop(labels=['labels', 'pub_types', 'file', 'mesh_headings', 'keywords'], axis=1) 
    df = df.dropna(subset=['date'], axis=0)
    
    df['title'] = df['title'].fillna('')
    df['title'] = df['title'].apply(lambda x: x.strip('[]'))
    df['title_and_abstract'] = df['title'] + df['abstract']
    if subsample:
        if year<0:
            print("subsample only works if read_year is given a year and not a path.")
            return df
        index_path = os.path.join(folder, f'pubmed_{year}_sample_index.pickle')
        if os.path.exists(index_path):
            sample_index = pickle.load(open(index_path, 'rb'))
            return df.loc[sample_index]
        print(f"sample index path: {index_path} not found.")
    return df


def read_year_to_ndocs(version=DEFAULT_PUBMED_VERSION):
    year_to_ndocs = pd.read_csv(os.path.join(DATA_PATH, 'year_to_ndocs.csv'),
                                index_col=0,
                                dtype={'year': int, 'ndocs_2019': int, 'ndocs_2020': int}).to_dict(orient='dict')[f'ndocs_{version}']
    return year_to_ndocs


def subsample_by_minimum_year(years_list, version=DEFAULT_PUBMED_VERSION):
    folder = {2019: FULL_PUMBED_2019_PATH, 2020: FULL_PUMBED_2020_PATH}[version]
    year_to_ndocs = read_year_to_ndocs()
    num_samples = min([year_to_ndocs[year] for year in years_list])
    print(f"num_samples: {num_samples}")
    for year in years_list:
        df = read_year(year, version)
        sample = df.sample(num_samples, axis=0).index
        print(f"sampled {len(sample)}/{len(df)} records from {year}")
        with open(os.path.join(folder, f'pubmed_{year}_sample_index.pickle'), 'wb') as out:
            pickle.dump(sample, out)

def read_subsample(year, version=DEFAULT_PUBMED_VERSION):
    folder = {2019: FULL_PUMBED_2019_PATH, 2020: FULL_PUMBED_2020_PATH}[version]
    df = read_year(year, version)
    sample_index = pickle.load(open(os.path.join(folder, f'pubmed_{year}_sample_index.pickle'), 'rb'))
    return df.loc[sample_index]
    

if __name__ == "__main__":
    compare_versions(2010, 2020)
