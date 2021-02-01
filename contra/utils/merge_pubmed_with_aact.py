import os
import pandas as pd
import sys
sys.path.append('/home/shunita/fairemb')
from contra.utils.pubmed_utils import read_year


def merge_pmids_with_ncts(pubmed_folder='pubmed_2020_by_years',
                          nct_file='ncts_with_participants.csv',
                          output_csv='pubmed2020_abstracts_with_participants.csv'):
    pubmed_path = os.path.join(os.path.expanduser('~'), pubmed_folder)
    DATA_PATH =  os.path.join(os.path.expanduser('~'), 'fairemb', 'data')
    ncts = pd.read_csv(os.path.join(DATA_PATH, nct_file), index_col=0)
    print(f'read {len(ncts)} ncts with participants')

    def process_nct_list(ncts_string, field):
        row_ncts = ncts_string.split(';')
        sum_of_field = 0
        found_match = False
        for nct in set(row_ncts):
            if nct in ncts.index:
                found_match = True
                matching = ncts.loc[nct]
                sum_of_field += ncts.loc[nct][field]
        if found_match:
            return sum_of_field
        return None

    files = sorted([x for x in os.listdir(pubmed_path) if x.endswith('csv')])

    for fname in files:
        print(f'working on {fname}')
        df = read_year(os.path.join(pubmed_path, fname))
        df = df[~df['ncts'].isna()]
        df['female'] = df['ncts'].apply(lambda x: process_nct_list(x, 'female'))
        df['male'] = df['ncts'].apply(lambda x: process_nct_list(x, 'male'))
        output_path = os.path.join(DATA_PATH, output_csv)
        write_header = not os.path.exists(output_path)
        df.to_csv(output_path, mode='a', header=write_header)



if __name__ == '__main__':
    merge_pmids_with_ncts(pubmed_folder='pubmed_2020_by_years',
                          nct_file='ncts_with_participants20200126.csv',
                          output_csv='pubmed2020_abstracts_with_participants20200126.csv')

                                                                    
