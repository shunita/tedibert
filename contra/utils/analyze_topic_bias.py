import os
import numpy as np
import pandas as pd
from ast import literal_eval
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from contra.constants import DATA_PATH
from contra.utils.pubmed_utils import is_cui, read_cui_prevalence, read_cui_names


def year_range_to_description(year_range):
    if year_range[0] == year_range[1]:
        return f'{year_range[0]}'
    return f'{year_range[0]}_{year_range[1]}'


def get_topics_with_change_in_bias(first_year_range=(2010, 2013),
                                   second_year_range=(2017, 2018)):
    df = pd.read_csv('topics_years.tsv', sep='\t')
    df['participants'] = df['m'] + df['f']
    first_desc = year_range_to_description(first_year_range)
    second_desc = year_range_to_description(second_year_range)

    first = df[(df['year'] >= first_year_range[0]) & (df['year'] <= first_year_range[1])]
    first = first.groupby(by='mesh_topic')[['f', 'm', 'abstract', 'participants']].sum().reset_index()
    first['f_percent'] = first['f'] / first['participants']

    second = df[(df['year'] >= second_year_range[0]) & (df['year'] <= second_year_range[1])]
    second = second.groupby(by='mesh_topic')[['f', 'm', 'abstract', 'participants']].sum().reset_index()
    second['f_percent'] = second['f'] / second['participants']

    fields = ['mesh_topic', 'f_percent', 'abstract', 'participants']
    merged = first[fields].merge(second[fields], on='mesh_topic',
                                 suffixes=[f'_{first_desc}', f'_{second_desc}'])
    merged['min_abstracts'] = merged[[f'abstract_{first_desc}', f'abstract_{second_desc}']].min(axis=1)
    merged = merged[merged['min_abstracts'] > 5]
    merged = merged.drop(columns=['min_abstracts'])
    merged['f percent change'] = merged[f'f_percent_{second_desc}'] - merged[f'f_percent_{first_desc}']
    merged.to_csv(f'topics_fpercent_change_{first_desc}vs{second_desc}.csv', index=False)


def analyze_topics_by_years(abstract_df_path, output_path='topics_years2019.tsv'):
    df = pd.read_csv(abstract_df_path, index_col=0)
    df = df.dropna(subset=['mesh_headings'], axis=0)
    #df = df[df['ncts'] != 'NCT00198822']  # remove a specific trial with 59K women
    df['mesh_headings'] = df['mesh_headings'].apply(lambda x: x.split(';'))
    # topic -> {year -> m,f,abstracts}
    topics_years = defaultdict(lambda: defaultdict(lambda: {'m': 0, 'f': 0, 'abstracts': 0}))
    for _, r in df.iterrows():
        topics = r['mesh_headings']
        year = r['year']
        m, f = r['male'], r['female']
        for topic in topics:
            topics_years[topic][year]['m'] += m
            topics_years[topic][year]['f'] += f
            topics_years[topic][year]['abstracts'] += 1
    out = open(output_path, "w")
    out.write('mesh_topic\tyear\tm\tf\tabstracts\n')
    for topic in topics_years:
        for year in topics_years[topic]:
            out.write(f'{topic}\t{year}\t{topics_years[topic][year]["m"]}\t{topics_years[topic][year]["f"]}\t{topics_years[topic][year]["abstracts"]}\n')
    out.close()


def topic_bias_over_time(year_ranges,
                         topics_years_path='topics_years2019.tsv',
                         output_path='topics_fpercent_change.csv'):
    df = pd.read_csv(topics_years_path, sep='\t')
    df['participants'] = df['m'] + df['f']

    merged = None
    fields = ['mesh_topic', 'f_percent', 'abstracts', 'participants']
    for year_range in year_ranges:
        subset = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
        subset = subset.groupby(by='mesh_topic')[['f', 'm', 'abstracts', 'participants']].sum().reset_index()
        subset['f_percent'] = subset['f'] / subset['participants']
        desc = year_range_to_description(year_range)
        subset = subset[fields]
        subset = subset.rename(columns={'f_percent': f'f_percent_{desc}',
                                        'abstracts': f'abstracts_{desc}',
                                        'participants': f'participants_{desc}'})
        if merged is None:
            merged = subset
        else:
            merged = merged.merge(subset, on='mesh_topic')
    descs = [year_range_to_description(year_range) for year_range in year_ranges]
    merged['min_abstracts'] = merged[[f'abstracts_{desc}' for desc in descs]].min(axis=1)
    fpercents = merged[[col for col in merged.columns if col.startswith('f_percent')]]
    merged['min max diff f percent'] = fpercents.max(axis=1) - fpercents.min(axis=1)
    merged.to_csv(output_path, index=False)




def count_cui_appearances_over_time(output_file=None):
    df = pd.read_csv(
        os.path.join(DATA_PATH, 'abstracts_and_population_tokenized_for_cui2vec_copyrightfix_sent_sep.csv'),
        index_col=0)
    df['tokenized_sents'] = df['tokenized_sents'].apply(literal_eval)
    df['tokenized'] = df['tokenized_sents'].apply(lambda seq: (" ".join(seq)).split())
    print(f"first df: {len(df)} rows")

    df1 = pd.read_csv(os.path.join(DATA_PATH, 'abstracts_population_date_topics.csv'), index_col=0)
    df1['year'] = df1['date'].apply(lambda x: int(x[-4:]))
    min_year = df1['year'].min()
    max_year = df1['year'].max()
    df = df.merge(df1[['year']], left_index=True, right_index=True)
    print(f"after merge: {len(df)}")
    # now tokenized is a list of words.
    # now count the appearances of each CUI in each year
    # cui->year->num appearances, female participants, male participants
    counter = defaultdict(lambda: defaultdict(lambda: {'abstracts': 0, 'fpercents': [], 'M': 0, 'F': 0}))
    for i, row in df.iterrows():
        abstract = row['tokenized']
        year, female, male = row['year'], row['female'], row['male']
        for word in set(abstract):
            if not is_cui(word):
                continue
            counter[word][year]['abstracts'] += 1
            if male+female > 0:
                counter[word][year]['fpercents'].append(female/(male+female))
            counter[word][year]['M'] += male
            counter[word][year]['F'] += female
    print(f"len counter: {len(counter)}, years: {min_year} - {max_year}")
    fem_prevalence = read_cui_prevalence()
    cui2name = read_cui_names()
    # make a dataframe out of this data
    rows = []
    for cui in counter.keys():
        prev = fem_prevalence[cui] if cui in fem_prevalence else None
        new_row = {'cui': cui, 'name': cui2name[cui], 'fem_prevalence_prop': prev}
        for year in range(min_year, max_year+1):
            #total = counter[cui][year]['F']+counter[cui][year]['M']
            #if total == 0:
            fpercents = counter[cui][year]['fpercents']
            if len(fpercents) == 0:
                new_row[f"{year}_fparticipants"] = None
                new_row[f"{year}_bias"] = None
            else:
                new_row[f"{year}_fparticipants"] = np.mean(fpercents)
                if prev is not None:
                    new_row[f"{year}_bias"] = new_row[f"{year}_fparticipants"] - prev
                else:
                    new_row[f"{year}_bias"] = None
        rows.append(new_row)
    data = pd.DataFrame.from_records(rows)
    if output_file is not None:
        columns = ['cui', 'name', 'fem_prevalence_prop'] + \
                  [f"{year}_fparticipants" for year in range(min_year, max_year + 1)] + \
                  [f"{year}_bias" for year in range(min_year, max_year + 1)]
        data[columns].to_csv(output_file)
    return data

def count_cui_appearances(output_file=None):
    df = pd.read_csv(
        os.path.join(DATA_PATH, 'abstracts_and_population_tokenized_for_cui2vec_copyrightfix_sent_sep.csv'),
        index_col=0)
    df['tokenized_sents'] = df['tokenized_sents'].apply(literal_eval)
    df['tokenized'] = df['tokenized_sents'].apply(lambda seq: (" ".join(seq)).split())

    # now tokenized is a list of words.
    # now count the appearances of each CUI
    # cui->num appearances, female participants, male participants
    counter = defaultdict(lambda: {'abstracts': 0, 'fpercents': [], 'M': 0, 'F': 0})
    for i, row in df.iterrows():
        abstract = row['tokenized']
        female, male = row['female'], row['male']
        for word in set(abstract):
            if not is_cui(word):
                continue
            counter[word]['abstracts'] += 1
            if male+female > 0:
                counter[word]['fpercents'].append(female/(male+female))
            counter[word]['M'] += male
            counter[word]['F'] += female

    fem_prevalence = read_cui_prevalence(with_total=True)
    cui2name = read_cui_names()
    # make a dataframe out of this data
    rows = []
    for cui in counter.keys():
        prev, total_patients = None, None
        if cui in fem_prevalence:
            prev = fem_prevalence[cui]['fem_prevalence_prop']
            total_patients = fem_prevalence[cui]['total_patients']
        new_row = {'cui': cui,
                   'name': cui2name[cui],
                   'fem_prevalence_prop': prev,
                   'total_patients': total_patients,
                   'abstracts': counter[cui]['abstracts']}
        fpercents = counter[cui]['fpercents']
        new_row['fem_participant_prop'] = None
        if len(fpercents) > 0:
            new_row['fem_participant_prop'] = np.mean(fpercents)
        rows.append(new_row)
    data = pd.DataFrame.from_records(rows)
    if output_file is not None:
        data.to_csv(output_file)
    return data


def find_slope(num_sequence):
    seq = [num for num in num_sequence if not np.isnan(num)]
    if len(seq) < 2:
        return None
    x = np.array(range(1, len(seq)+1)).reshape(-1, 1)
    lreg = LinearRegression().fit(x, seq)
    return lreg.coef_[0]


def analyze_CUI_bias_trend(df_path, year_range=(2008, 2018), output_path=None):
    df = pd.read_csv(df_path, index_col=0)
    years = range(year_range[0], year_range[1]+1)
    cols = [f'{year}_fparticipants' for year in years]
    df['fparticipants_trend'] = df.apply(lambda row: find_slope(row[cols].values), axis=1)

    cols = [f'{year}_bias' for year in years]
    df['bias_trend'] = df.apply(lambda row: find_slope(row[cols].values), axis=1)
    df['abs_bias_trend'] = df.apply(lambda row: find_slope(np.abs(row[cols].values)), axis=1)
    if output_path is not None:
        cols = ['cui', 'name', 'fem_prevalence_prop'] + \
               [c for c in df.columns if c.endswith("participants")] + \
               ['fparticipants_trend'] + \
               [c for c in df.columns if c.endswith("bias")] + \
               ['bias_trend', 'abs_bias_trend']
        df.to_csv(output_path)
    return df
