from ast import literal_eval
from collections import defaultdict

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def build_cuis_and_times(abstracts_df):
    """:return a dict of the form cui->{year->list of ratios}"""
    cuis_and_times = defaultdict(lambda: defaultdict(list))
    for i, r in abstracts_df.iterrows():
        for word in set(r['tokenized']):
            cuis_and_times[word][r['year']].append(r['female_participant_ratio'])
    return cuis_and_times


def calculate_stats_for_fp_list(fp_list, remove_outliers=False):
    # e.g., all the female percents for a given topic for a specific year range.
    mean, std = -1, -1
    a = np.array([item for sublist in fp_list for item in sublist if item != []])
    if len(a) == 0:
        return a, mean, std, 0
    # remove outliers
    removed = 0
    if remove_outliers:
        q1 = np.quantile(a, 0.25)
        q3 = np.quantile(a, 0.75)
        iqr = q3 - q1
        len_before = len(a)
        # a = a[(a > q1) & (a < q3)]
        a = a[(a > q1 - 1.5*iqr) & (a < q3 + 1.5*iqr)]
        len_after = len(a)
        removed = len_before-len_after
        print(f"q1: {q1} q3: {q3} removed: {removed}")
    if len(a) > 0:
        mean = np.mean(a)
        std = np.std(a)
    return a, mean, std, removed

def read_data_for_comparing_year_ranges():
    df1 = pd.read_csv("data/abstracts_and_population_tokenized_for_cui2vec_copyrightfix_sent_sep.csv", index_col=0)
    df2 = pd.read_csv("data/abstracts_population_date_topics.csv", index_col=0)

    df3 = df1.merge(df2[['date']], left_index=True, right_index=True)
    df3['year'] = df3['date'].apply(lambda s: int(s[-4:]))

    df3['tokenized_sents'] = df3['tokenized_sents'].apply(literal_eval)
    df3['tokenized'] = df3['tokenized_sents'].apply(lambda x: (" ".join(x)).split())
    df3['female_participant_ratio'] = df3['female'] / (df3['male'] + df3['female'])

    cuis_and_times = build_cuis_and_times(df3)
    return cuis_and_times


def compare_year_ranges(range1_start, range1_end, range2_start, range2_end):
    cuis_and_times = read_data_for_comparing_year_ranges()
    records = []
    for cui in cuis_and_times:
        year_to_ratios = cuis_and_times[cui]
        r1 = [year_to_ratios[y] for y in range(range1_start, range1_end + 1)]
        r1, r1_mean, r1_std, r1_removed = calculate_stats_for_fp_list(r1, remove_outliers=True)
        r2 = [year_to_ratios[y] for y in range(range2_start, range2_end + 1)]
        r2, r2_mean, r2_std, r2_removed = calculate_stats_for_fp_list(r2, remove_outliers=True)
        # test for proportion difference between the two ranges.
        # Each sample represents an abstract and the compared value is the proportion of female participants.
        # the test is two-sided and assumes equal variance by default.
        statistic, pval = -1, -1
        if len(r1) > 0 and len(r2) > 0:
            statistic, pval = stats.ttest_ind(r1, r2)
        rec = {
            'cui': cui,
            'abstract_count_all_years': sum([len(ratios) for ratios in year_to_ratios.values()]),
            f'{range1_start}_{range1_end}_fp': r1_mean,
            f'{range1_start}_{range1_end}_fp_std': r1_std,
            f'{range1_start}_{range1_end}_num_papers': len(r1),
            f'{range1_start}_{range1_end}_removed_outliers': r1_removed,
            f'{range2_start}_{range2_end}_fp': r2_mean,
            f'{range2_start}_{range2_end}_fp_std': r2_std,
            f'{range2_start}_{range2_end}_num_papers': len(r2),
            f'{range2_start}_{range2_end}_removed_outliers': r2_removed,
            'diff_pvalue': pval,
            }
        records.append(rec)
    records = pd.DataFrame.from_records(records)
    cuis = pd.read_csv("data/cui_stats_pubmed_2018_and_maccabi.csv", index_col=0)
    records = records.merge(cuis[['cui', 'name', 'fem_prevalence_prop', 'total_patients']], on='cui')
    cuis2 = pd.read_csv('data/cui_table_for_cui2vec_with_abstract_counts.csv', index_col=0)
    records = records.merge(cuis2[['cui', 'semtypes']], on='cui')
    records.to_csv(f'data/cui_fem_prop_ranges{range1_start}_{range1_end}vs{range2_start}_{range2_end}_outliers_removed.csv')


def violin_plot_for_concept_list(concept_list, range1_start, range1_end, range2_start, range2_end, remove_outliers=False):
    if concept_list == []:
        concept_list = ['C0027051', 'C0020538', 'C0018801', 'C0027947', 'C0002871']
    cuis_and_times = read_data_for_comparing_year_ranges()
    records = []
    for cui in concept_list:
        year_to_ratios = cuis_and_times[cui]
        r1 = [year_to_ratios[y] for y in range(range1_start, range1_end + 1)]
        r1, r1_mean, r1_std, r1_removed = calculate_stats_for_fp_list(r1, remove_outliers=remove_outliers)
        for fp in r1:
            records.append({'cui': cui, 'Year range': f'{range1_start}_{range1_end}', 'Female participants': fp})
        r2 = [year_to_ratios[y] for y in range(range2_start, range2_end + 1)]
        r2, r2_mean, r2_std, r2_removed = calculate_stats_for_fp_list(r2, remove_outliers=remove_outliers)
        for fp in r2:
            records.append({'cui': cui, 'Year range': f'{range2_start}_{range2_end}', 'Female participants': fp})
    records = pd.DataFrame.from_records(records)
    cuis = pd.read_csv("data/cui_stats_pubmed_2018_and_maccabi.csv", index_col=0)
    records = records.merge(cuis[['cui', 'name']], on='cui')

    fig, ax = plt.subplots()
    # sns.violinplot(x='name', y='fp', hue='time_range', data=records, split=True, ax=ax)
    sns.boxplot(x='name', y='Female participants', data=records, hue='Year range', fliersize=0)
    # sns.despine()
    # fig.tight_layout()
    plt.xticks(rotation=360-20)
    plt.subplots_adjust(bottom=0.15)
    plt.show()

