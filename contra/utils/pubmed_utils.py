import os
import string
from collections import defaultdict

import pandas as pd
import pickle
from tqdm import tqdm
import re
import sys
import numpy as np

from transformers import AutoTokenizer

sys.path.append('/home/shunita/fairemb')
from contra.constants import FULL_PUMBED_2019_PATH, FULL_PUMBED_2020_PATH, DATA_PATH, DEFAULT_PUBMED_VERSION
from contra.utils import text_utils as tu


def pubmed_version_to_folder(version=DEFAULT_PUBMED_VERSION):
    return {2019: FULL_PUMBED_2019_PATH, 2020: FULL_PUMBED_2020_PATH}[version] 


def params_to_description(abstract_weighting_mode, only_aact_data, pubmed_version=DEFAULT_PUBMED_VERSION):
    desc = ''
    if abstract_weighting_mode == 'subsample':
        desc = 'sample'
    if only_aact_data:
        desc += '_aact'
    if pubmed_version != 2019:
        desc += 'v2020'
    return desc


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


def process_year_range_into_sentences(start_year, end_year, pubmed_version, abstract_weighting_mode):
    text_utils = tu.TextUtils()
    pubmed_folder = pubmed_version_to_folder(pubmed_version)
    desc = params_to_description(abstract_weighting_mode, only_aact_data=False, pubmed_version=pubmed_version)
    for year in range(start_year, end_year + 1):
        sentences = []
        year_sentences_path = os.path.join(pubmed_folder, f'{year}{desc}_sentences.pickle')
        if os.path.exists(year_sentences_path):
            continue
        relevant = read_year(year, version=pubmed_version, subsample=(abstract_weighting_mode=='subsample'))
        print(f'splitting {year} abstracts to sentences...')
        relevant['sentences'] = relevant['title_and_abstract'].apply(text_utils.split_abstract_to_sentences)
        print(f'saving sentence list...')
        for pmid, r in tqdm(relevant.iterrows(), total=len(relevant)):
            sentences.extend(r['sentences'])
        pickle.dump(sentences, open(year_sentences_path, 'wb'))
        print(f'saved {len(sentences)} sentences from {year} to pickle file {year_sentences_path}.')


def read_year_to_ndocs(version=DEFAULT_PUBMED_VERSION):
    year_to_ndocs = pd.read_csv(os.path.join(DATA_PATH, 'year_to_ndocs.csv'),
                                index_col=0,
                                dtype={'year': int, 'ndocs_2019': int, 'ndocs_2020': int}).to_dict(orient='dict')[f'ndocs_{version}']
    return year_to_ndocs


def subsample_by_minimum_year(years_list, version=DEFAULT_PUBMED_VERSION):
    folder = pubmed_version_to_folder(version)
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
    folder = pubmed_version_to_folder(version)
    df = read_year(year, version)
    sample_index = pickle.load(open(os.path.join(folder, f'pubmed_{year}_sample_index.pickle'), 'rb'))
    return df.loc[sample_index]


def load_aact_data(version, year_range=None, sample=False):
    '''
    @param version: 2019 or 2020
    @param year_range: a tuple of (start_year, end_year). If given, will be used to filter the abstracts to these years.
    '''
    file_path = os.path.join(DATA_PATH, f'pubmed{version}_abstracts_with_participants.csv')
    print(f"reading: {file_path}")
    df = pd.read_csv(file_path, index_col=0)
    if sample:
        df = df.sample(1000)
    df['title'] = df['title'].fillna('')
    df['title'] = df['title'].apply(lambda x: x.strip('[]'))
    df['title_and_abstract'] = df['title'] + ' ' + df['abstract']
    if year_range is not None:
        start_year, end_year = year_range
        df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
    return df


def should_keep_sentence(sentence):
    blacklist = ['http', 'https', 'url', 'www', 'clinicaltrials.gov', 'copyright', 'funded by', 'published by', 'subsidiary', '©', 'all rights reserved']
    s = sentence.lower()
    for w in blacklist:
        if w in s:
            return False
    # re, find NCTs
    if len(re.findall('nct[0-9]+', s)) > 0:
        return False
    if len(sentence) < 40:
        return False
    return True


def clean_abstracts(df, abstract_field='title_and_abstract', output_sentences_field='sentences'):
    text_utils = tu.TextUtils()
    # consolidate some terms
    # term_replacement = {'hba(1c)': 'hba1c', 'hemoglobin A1c': 'hba1c', 'a1c': 'hba1c',
    #                     # 'â‰¤': '',  # less-than sign
    #                     # 'â‰¥': ''  # greater-than sign
    #                     }
    # printable = set(string.printable)
    #
    # def replace_terms(text):
    #     text_lower = text.lower()
    #     for k, v in term_replacement.items():
    #         text_lower = text_lower.replace(k, v)
    #     # remove all non-ascii characters
    #     text_lower = ''.join(filter(lambda x: x in printable, text_lower))
    #     return text_lower
    #
    # df['title_and_abstract'] = df['title_and_abstract'].apply(replace_terms)


    # filter sentences
    if output_sentences_field not in df.columns:
        df[output_sentences_field] = df[abstract_field].apply(text_utils.split_abstract_to_sentences)
    d = {'total': 0, 'remaining': 0}

    def pick_sentences(sentences):
        new_sents = [sent for sent in sentences if should_keep_sentence(sent)]
        d['total'] += len(sentences)
        d['remaining'] += len(new_sents)
        return new_sents

    def join_to_abstract(sentences):
        return ' '.join(sentences)

    df[output_sentences_field] = df[output_sentences_field].apply(pick_sentences)
    df[abstract_field] = df[output_sentences_field].apply(join_to_abstract)
    print(f"kept {d['remaining']}/{d['total']} sentences")
    return df


def process_aact_year_range_to_sentences(version, year_range, word_list=False):
    '''
    @param version: 2019 or 2020
    @param year_range: a tuple of (start_year, end_year). Used to filter the abstracts to these years.
    @param word_list: should this function return each sentence as a wordlist (True) or as a string (False).
    '''
    df = load_aact_data(version, year_range)
    text_utils = tu.TextUtils()
    df['sentences'] = df['title_and_abstract'].apply(text_utils.split_abstract_to_sentences)

    sentences_flat = []
    for abstract in df['sentences'].values:
        sentences_flat.extend(abstract)
    if word_list:
        sentences_flat = [tu.word_tokenize(sent) for sent in sentences_flat]
    return sentences_flat

def df_to_tokenized_sentence_list(df):
    text_utils = tu.TextUtils()
    if 'sentences' not in df.columns:
        df['sentences'] = df['title_and_abstract'].apply(text_utils.split_abstract_to_sentences)
    sentences_flat = []
    for abstract in df['sentences'].values:
        sentences_flat.extend(abstract)
    sentences_flat = [text_utils.word_tokenize(sent) for sent in sentences_flat]
    return sentences_flat


def split_abstracts_to_sentences_df(df_of_abstracts, text_field='title_and_abstract',
                                    keep=('date', 'year', 'female', 'male'), overlap=True):
    text_utils = tu.TextUtils()
    df_of_abstracts['sentences'] = df_of_abstracts[text_field].apply(text_utils.split_abstract_to_sentences)
    sentences = []
    keep = list(keep)+['sentences']
    for pmid, r in df_of_abstracts.iterrows():
        d = {field: r[field] for field in keep}
        for pos, sent in enumerate(r['sentences']):
            if overlap or pos % 3 == 1:
                new_row = d.copy()
                new_row['text'] = sent
                new_row['pos'] = pos
                sentences.append(new_row)
    return pd.DataFrame.from_records(sentences)


def populate_idf_dict_bert(tokenizer_path='google/bert_uncased_L-2_H-128_A-2'):
    idf_file = os.path.join(DATA_PATH, 'pubmed2020train_idf_dict.pickle')
    if os.path.exists(idf_file):
        print(f"reading idf dict from {idf_file}")
        token_idf = pickle.load(open(idf_file, 'rb'))
        return token_idf
    print(f"populating idf dict from train.")
    df = pd.read_csv(os.path.join(DATA_PATH, 'pubmed2020_assigned.csv'), index_col=0)
    df = df[df['assignment'] == 0]  # only train
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenized = tokenizer.batch_encode_plus(df['title_and_abstract'].values.tolist(), add_special_tokens=False).input_ids
    token_counter = defaultdict(int)
    for abs in tokenized:
        for token_id in set(abs):
            token_counter[token_id] += 1
    N = len(df)
    # cnt is guaranteed to be positive
    token_idf = {}
    for token in token_counter:
        token_idf[token] = np.log(N/token_counter[token])
    # token_idf = {token_id: np.log(N/cnt) for token_id, cnt in token_counter.items()}
    pickle.dump(token_idf, open(idf_file, 'wb'))
    return token_idf

def is_cui(word):
    return word.startswith("C") and len(word) == 8


def read_cui_prevalence(with_total=False):
    dandp = pd.read_csv(os.path.join(DATA_PATH, 'diseases_and_prevalence.csv'))
    dandp['total_patients'] = dandp['W'] + dandp['M']
    dandp = dandp[dandp['total_patients'] > 0]
    dandp['fem_prevalence_prop'] = dandp['W'] / dandp['total_patients']
    #dandp[dandp['total_patients'] == 0]['fem_prevalence_prop'] = -1
    dandp = dandp.set_index(['cui'])
    if with_total:
        return dandp[['fem_prevalence_prop', 'total_patients']].to_dict(orient='index')
    return dandp['fem_prevalence_prop'].to_dict()


def read_cui_names(return_semtypes=False):
    cui_table = pd.read_csv(os.path.join(DATA_PATH, 'cui_table_for_cui2vec.tsv'), sep='\t', index_col=0)
    cui_table = cui_table.set_index(['cui'])
    if return_semtypes:
        return cui_table['name'].to_dict(), cui_table['semtypes'].to_dict()
    return cui_table['name'].to_dict()


def calculate_prevalence(abstract, prevalence_dict, prev_agg_mode=0, return_cui=False, cui_names=None):
    '''
    :param abstract: abstract as a list of strings (sentences)
    :param fem_prop: female participant proportion in the clinical trial
    :param prevalence_dict: dictionary of the form CUI->female prevalence proportion
    :param prev_agg_mode: how to estimate the expected prevalence for the abstract:
                0 - use the prevalence of the first CUI-with-prevalence in the abstract
                1 - average the prevalences for each CUI in the abstract
                2 - average the prevalences for each CUI from the first sentence of the abstract (title)
                3 - most common CUI's prevalence
    :return: the estimated bias (participant_prop - estimated female prevalence) or None if prevalence is unknown
    '''
    prevalences = []
    cuis = defaultdict(int)
    if len(abstract) == 0:
        return None
    if prev_agg_mode == 2:
        sentences = [abstract[0]]
    else:
        sentences = abstract
    for sent in sentences:
        for token in sent.split():
            if not is_cui(token):
                continue
            if token in prevalence_dict:
                cuis[token] += 1
                prevalences.append((token, prevalence_dict[token]))

    def cui2name(cui):
        if cui_names is not None:
            return cui_names[cui]
        return cui

    # return len(cuis)
    # aggregate to a single expected prevalence according to aggregation mode
    if len(cuis) == 0:
        # print(f"cuis: {cuis}")
        return None  # Unknown bias
    if prev_agg_mode == 0:
        prevalence = prevalences[0][1]
        if return_cui:
            return prevalence, cui2name(prevalences[0][0])
        return prevalence
    elif prev_agg_mode in (1, 2):  # mean of prevalences of cuis in all the abstract or the first sentence (title)
        prevalence = np.mean([x[1] for x in prevalences])
        if return_cui:
            return prevalence, None
        return prevalence
    else:  # most common cui
        cuis_and_counts = list(cuis.items())
        most_common_cui = cuis_and_counts[np.argmax([x[1] for x in cuis_and_counts])][0]
        prevalence = prevalence_dict[most_common_cui]
        if return_cui:
            return prevalence, cui2name(most_common_cui)
        return prevalence


def participants_to_repetitions(x):
    if x == 0:
        return 0
    if x <= 10:
        return 1
    if x <= 100:
        return 10
    else:
        return 20


def repeat_by_participants(df):
    """repeat some of the entries in the dataset based on the number of women in them ('female' field)"""
    df['reps'] = df['female'].apply(participants_to_repetitions)
    df2 = df.loc[df.index.repeat(df['reps'])]
    # shuffle
    df2 = df2.sample(len(df2)).reset_index(drop=True)
    return df2


def read_mesh_terms(path):
    lines = open(path, "r").read().split('\n')