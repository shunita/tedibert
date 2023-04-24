
import os
import sys
sys.path.append(os.path.expanduser('~/fairemb'))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ast import literal_eval
import statistics
import scipy.stats.distributions as dist
from scipy.stats import ttest_ind
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from contra.constants import DATA_PATH
from contra.utils.pubmed_utils import is_cui, read_cui_prevalence, read_cui_names

# ghdx 2016 female prevalence stats
prevalence = {'Cardiovascular': 0.511160467,
              'Diabetes': 0.481711303,
              'Digestive': 0.596079179,
              'Hepatitis A, B, C, E': 0.43526828,
              'HIV/AIDS': 0.502764201,
              'Kidney diseases': 0.568505711,
              'Mental': 0.48438023,
              'Musculoskeletal': 0.561939186,
              'Neoplasms': 0.51244735,
              'Neurological': 0.594898349,
              'Respiratory': 0.476451018}


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


def count_cui_appearances_over_time(output_file=None, raw_participant_numbers=False):
    df = read_tokenized_abstracts_with_year()
    # df = df[df['ncts'] != 'NCT00198822']  # remove a specific trial with 59K women
    min_year = df['year'].min()
    max_year = df['year'].max()
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
    cui2name, cui2semtype = read_cui_names(return_semtypes=True)
    # make a dataframe out of this data
    rows = []
    for cui in counter.keys():
        t = cui2semtype[cui]
        is_of_selected_semtype = False
        for selected_type in ['dsyn', 'neop', 'acab','cgab', 'comd', 'inpo', 'menp', 'mobd']:
            if selected_type in t:
                is_of_selected_semtype = True
                break
        if not is_of_selected_semtype:
            continue
        prev = fem_prevalence[cui] if cui in fem_prevalence else None
        new_row = {'cui': cui, 'name': cui2name[cui], 'semtypes': t, 'fem_prevalence_prop': prev }
        for year in range(min_year, max_year+1):
            #total = counter[cui][year]['F']+counter[cui][year]['M']
            #if total == 0:
            if raw_participant_numbers:
                new_row[f"{year}_F"] = counter[cui][year]['F']
                new_row[f"{year}_M"] = counter[cui][year]['M']
            else:
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
        columns = ['cui', 'name', 'semtypes', 'fem_prevalence_prop']
        if raw_participant_numbers:
            for year in range(min_year, max_year+1):
                columns.append(f"{year}_F")
                columns.append(f"{year}_M")
        else:
            columns += [f"{year}_fparticipants" for year in range(min_year, max_year + 1)] + \
                       [f"{year}_bias" for year in range(min_year, max_year + 1)]
        data[columns].to_csv(output_file)
    return data


def count_participants_in_disease_ncts_by_year(disease_abstracts, remove_outliers=False, remove_single_sex=False):
    ncts_with_participants = pd.read_csv('data/ncts_with_participants.csv')
    if remove_outliers:
        ncts_with_participants = ncts_with_participants[ncts_with_participants.female + ncts_with_participants.male <= 10000]
    if remove_single_sex:
        ncts_with_participants = ncts_with_participants[
            (ncts_with_participants.female > 0)  & (ncts_with_participants.male > 0)]
    ncts_with_participants['fpercent'] = ncts_with_participants['female'] / (ncts_with_participants['male'] + ncts_with_participants['female'])

    #disease_abstracts = abstracts[abstracts['tokenized'].apply(lambda x: contains_terms(x, disease_cuis))]
    min_year = disease_abstracts['year'].min()
    max_year = disease_abstracts['year'].max()
    print(f"year range: {min_year}-{max_year}")
    nct_counter = {year: {'M': 0, 'F': 0, 'fpercents': []} for year in range(min_year, max_year+1)}
    for year in range(min_year, max_year+1):
        disease_ncts_year = disease_abstracts[disease_abstracts.year == year].ncts
        if len(disease_ncts_year) == 0:
            continue
        disease_ncts_year = set(disease_ncts_year.sum())
        cts = ncts_with_participants[ncts_with_participants.nct_id.apply(lambda x: x in disease_ncts_year)]
        nct_counter[year]['M'] = cts['male'].sum()
        nct_counter[year]['F'] = cts['female'].sum()
        nct_counter[year]['fpercents'] = list(cts['fpercent'].values)
    return nct_counter


def contains_terms(text_or_list, term_list):
    for term in term_list:
        if term in text_or_list:
            return True
    return False


def read_tokenized_abstracts_with_year():
    abstracts = pd.read_csv(
        os.path.join(DATA_PATH, 'abstracts_and_population_tokenized_for_cui2vec_copyrightfix_sent_sep.csv'),
        index_col=0)
    abstracts['ncts'] = abstracts['ncts'].apply(literal_eval)
    abstracts['tokenized_sents'] = abstracts['tokenized_sents'].apply(literal_eval)
    abstracts['tokenized'] = abstracts['tokenized_sents'].apply(lambda seq: (" ".join(seq)).split())
    df1 = pd.read_csv(os.path.join(DATA_PATH, 'abstracts_population_date_topics.csv'), index_col=0)
    df1['year'] = df1['date'].apply(lambda x: int(x[-4:]))
    abstracts = abstracts.merge(df1[['year', 'mesh_headings']], left_index=True, right_index=True)
    return abstracts


def make_graph_for_disease_category(counter_file, year_ranges, name_disease_filter=None, semtype_disease_filter=None,
                                    abstracts=None, remove_outliers=False, remove_single_sex=False):
    df = pd.read_csv(counter_file, index_col=0)
    df['semtypes'] = df['semtypes'].apply(literal_eval)
    cond = False
    if name_disease_filter is not None:
        cond = cond | (df['name'].apply(name_disease_filter))
    if semtype_disease_filter is not None:
        cond = cond | (df['semtypes'].apply(semtype_disease_filter))
    disease_subset = df[cond]
    disease_cuis = disease_subset.cui.values
    print(f"Found {len(disease_cuis)} cuis. Example: {disease_subset.name.values[:10]}")
    if abstracts is None:
        abstracts = read_tokenized_abstracts_with_year()
    subset = abstracts[abstracts['tokenized'].apply(lambda x: contains_terms(x, disease_cuis))]
    nct_counter = count_participants_in_disease_ncts_by_year(subset, remove_outliers, remove_single_sex)
    print('years,total_fpercent,median_fpercent,mean,std')
    for yr in year_ranges:
        f_total, m_total = 0, 0
        fpercents = []
        for year in range(yr[0], yr[1]+1):
            f_total += nct_counter[year]['F']
            m_total += nct_counter[year]['M']
            fpercents += nct_counter[year]['fpercents']
        print(f'{yr[0]}-{yr[1]},{f_total/(f_total+m_total)},{statistics.median(fpercents)},{np.mean(fpercents)},{np.std(fpercents)}')


def make_category_to_cuis_mapping():
    df = pd.read_csv('data/cuis_over_time.csv', index_col=0)
    df['semtypes'] = df['semtypes'].apply(literal_eval)
    cat_to_cuis = {}
    cat_to_cuis['Cardiovascular'] = df[df['name'].apply(lambda x: 'card' in x.lower())].cui.values
    cat_to_cuis['Diabetes'] = df[df['name'].apply(lambda x: ('diabet' in x.lower() or 'insulin' in x.lower()) and x!='insulinoma')].cui.values
    cat_to_cuis['Digestive'] = df[df['name'].apply(lambda x: 'gastro' in x.lower() or 'crohn' in x.lower())].cui.values
    cat_to_cuis['Hepatitis A, B, C, E'] = df[df['name'].apply(lambda x: ('Hepatitis A' in x or 'Hepatitis B' in x or 'Hepatitis C' in x or 'Hepatitis E' in x) and 'Carcinoma' not in x)].cui.values
    cat_to_cuis['HIV/AIDS'] = df[df['name'].apply(lambda x: 'hiv' in x.lower() or 'aids' in x.lower() or 'acquired immunodeficiency' in x.lower())].cui.values
    cat_to_cuis['Kidney diseases'] = df[df['name'].apply(lambda x: ('kidney' in x.lower() or 'renal' in x.lower()))].cui.values
    # maybe add 'menp' to the semtypes here.
    cat_to_cuis['Mental'] = df[(df['name'].apply(lambda x: 'schizophrenia' in x)) | (df['semtypes'].apply(lambda x: 'mobd' in x))].cui.values
    cat_to_cuis['Musculoskeletal'] = df[df['name'].apply(lambda x: ('arthritis' in x.lower() or 'low back pain' in x.lower()))].cui.values
    cat_to_cuis['Neoplasms'] = df[df['semtypes'].apply(lambda x: 'neop' in x)].cui.values
    cat_to_cuis['Neurological'] = df[df['name'].apply(lambda x: 'alzheimer' in x.lower() or 'parkinson' in x.lower() or
                                                                'cognition' in x.lower().split())].cui.values
    cat_to_cuis['Respiratory'] = df[df['name'].apply(lambda x: 'pulmonary' in x.lower() or 'pneumonia' in x.lower() or
                                                               'asthma' in x.lower())].cui.values
    return cat_to_cuis




def boxplot_for_category_list(category_name_and_cuis, year_ranges, remove_outliers=False, remove_single_sex=False):
    abstracts = read_tokenized_abstracts_with_year()
    records = []
    year_range_to_all_fpercents = {yr: [] for yr in year_ranges}
    for (cat_name, cui_list) in category_name_and_cuis:
        disease_abstracts = abstracts[abstracts['tokenized'].apply(lambda x: contains_terms(x, cui_list))]
        nct_counter = count_participants_in_disease_ncts_by_year(disease_abstracts, remove_outliers,
                                                                 remove_single_sex)
        print(f"{cat_name}")
        print('years,total_fpercent,median_fpercent,mean,std,n,CI')
        for yr in year_ranges:
            f_total, m_total = 0, 0
            fpercents = []
            for year in range(yr[0], yr[1] + 1):
                f_total += nct_counter[year]['F']
                m_total += nct_counter[year]['M']
                fpercents += nct_counter[year]['fpercents']
            if cat_name in prevalence:
                year_range_to_all_fpercents[yr].extend([fp - prevalence[cat_name] for fp in fpercents])
            for fp in fpercents:
                records.append({'category': cat_name, 'Year range': f'{yr[0]}_{yr[1]}', 'Female participants': fp})
            print(
                f'{yr[0]}-{yr[1]},{f_total / (f_total + m_total)},'
                f'{statistics.median(fpercents)},{np.mean(fpercents)},{np.std(fpercents)},{len(fpercents)},'
                f'{1.96*np.std(fpercents)/np.sqrt(len(fpercents))}')

    print('all_categories combined - bias results:')
    print('years,median_bias,mean_bias,std,n,CI')
    for yr in year_ranges:
        bias_results = year_range_to_all_fpercents[yr]
        if len(bias_results)>0:
            print(
                f'{yr[0]}-{yr[1]},'
                f'{statistics.median(bias_results)},{np.mean(bias_results)},{np.std(bias_results)},{len(bias_results)},'
                f'{1.96 * np.std(bias_results) / np.sqrt(len(bias_results))}')

    # records = pd.DataFrame.from_records(records)
    # records.to_csv('exp_results/bias_over_time_ranges_in_disease_categories.csv')

    # fig, ax = plt.subplots()
    # sns.boxplot(x='category', y='Female participants', data=records, hue='Year range', fliersize=0)
    # plt.xticks(rotation=360 - 20)
    # plt.subplots_adjust(bottom=0.15)
    # plt.savefig('exp_results/bias_over_time_ranges_in_disease_categories.png')


def boxplot_for_all_categories():
    cat_to_cuis = make_category_to_cuis_mapping()
    ranges = [(2010, 2013), (2016, 2018)]
    boxplot_for_category_list(cat_to_cuis.items(), ranges, remove_outliers=True,
                              remove_single_sex=False)
#     all_cuis_in_categories = []
#     for cat in cat_to_cuis:
#         all_cuis_in_categories.extend(cat_to_cuis[cat])
#     all_cuis_unique = list(set(all_cuis_in_categories))
#     print(f"collected {len(all_cuis_in_categories)} cuis from all categories, {len(all_cuis_unique)} unique cuis.")
#
#     boxplot_for_category_list([('all_categories', all_cuis_unique)], ranges, remove_outliers=False,
#                               remove_single_sex=False)

# def boxplot_for_bias_in_all_categories():
#     cat_to_cuis = make_category_to_cuis_mapping()
#     abstracts = read_tokenized_abstracts_with_year()
#     for (cat_name, cui_list) in cat_to_cuis:
#         nct_counter = count_participants_in_disease_ncts_by_year(cui_list, abstracts, remove_outliers=False,
#                                                                  remove_single_sex=False)
#         for year in nct_counter.keys():
#             nct_counter['year']['fpercents']

# By Feldman 2019 with slight changes
# def make_category_to_mesh_prefix_mapping():
#     cat_to_pref = {}
#     cat_to_pref['Cardiovascular'] = ['C14']
#     cat_to_pref['Diabetes'] = ['C19.246']
#     cat_to_pref['Digestive'] = ['C06']
#     cat_to_pref['Hepatitis A, B, C, E'] = ['C06.552.380'] # this is all hepatitis. maybe too wide.
#     cat_to_pref['HIV/AIDS'] = ['C20.673.480']
#     cat_to_pref['Kidney diseases'] = ['C12.950.419']
#     cat_to_pref['Mental'] = ['F03']
#     cat_to_pref['Musculoskeletal'] = ['C05']
#     cat_to_pref['Neoplasms'] = ['C04']
#     # cat_to_pref['Neurological'] = ['C10'] # this is nervous system. maybe too wide
#     cat_to_pref['Neurological'] = ['C10.228.140',  # brain diseases incl. parkinsons and alzheimer
#                                    'F03.615.250'  # cognition disorders
#                                    ]
#     cat_to_pref['Respiratory'] = ['C08']
#     return cat_to_pref

# # By top level mesh terms in the tree
def make_category_to_mesh_prefix_mapping():
    cat_to_pref = {}
    cat_to_pref['Infections'] = ['C01']
    cat_to_pref['Neoplasms'] = ['C04']
    cat_to_pref['Musculoskeletal'] = ['C05']
    cat_to_pref['Digestive'] = ['C06']
    # cat_to_pref['Stomatognathic Diseases'] = ['C07']
    cat_to_pref['Respiratory'] = ['C08']
    # cat_to_pref['Otorhinolaryngologic'] = ['C09']
    cat_to_pref['Nervous system'] = ['C10']
    # cat_to_pref['Eye'] = 'C11' #not enough clinical trials
    cat_to_pref['Urogenital'] = ['C12']
    cat_to_pref['Cardiovascular'] = ['C14']
    cat_to_pref['Hemic and Lymphatic'] = ['C15']
    cat_to_pref['Skin and Connective Tissue'] = ['C17']
    cat_to_pref['Nutritional and Metabolic Diseases'] = ['C18']
    cat_to_pref['Immune system'] = ['C20']
    # cat_to_pref['Environmental'] = ['C21']
    cat_to_pref['pathological conditions signs and symptoms'] = ['C23']
    # cat_to_pref['Occupational'] = ['C24']
    cat_to_pref['chemically induced'] = ['C25']
    # cat_to_pref['wounds and injuries'] = ['C26'] #not enough clinical trials
    cat_to_pref['Mental'] = ['F03']
    return cat_to_pref


def preprocess_abstract_mesh_field(text):
    topics = text.split(';')
    return [t.replace('  ', ', ') for t in topics]

def read_mesh_from_bin():
    records = open("data/mesh/mesh_d2023.bin","r").read().split("\n\n")
    pd_recs = []
    for r in records:
        lines = r.split("\n")[1:]
        d = {}
        for line in lines:
            parts = line.split(" = ")
            if parts[0] == 'AN':
                value = " = ".join(parts[1:])
            else:
                value = parts[1]
            if parts[0] in d:
                d[parts[0]].append(value)
            else:
                d[parts[0]] = [value]
        pd_recs.append(d)
    df = pd.DataFrame.from_records(pd_recs)
    return df


def check_term_prefix(term_list, prefs):
    for t in term_list:
        for pref in prefs:
            if t.startswith(pref):
                return True
    return False


def analyze_by_mesh_terms(year_ranges, remove_outliers=False, remove_single_sex=False):
    mesh = read_mesh_from_bin()
    b = len(mesh)
    mesh = mesh[~mesh.MN.isna()]
    print(f"removed mesh records without mesh number. before: {b}, after: {len(mesh)}")

    abstracts = read_tokenized_abstracts_with_year()
    b = len(abstracts)
    abstracts = abstracts[~abstracts['mesh_headings'].isna()]
    print(f"removed abstracts without mesh headings. before: {b}, after: {len(abstracts)}")
    abstracts['mesh_headings'] = abstracts['mesh_headings'].apply(preprocess_abstract_mesh_field)
    cat_to_pref = make_category_to_mesh_prefix_mapping()
    totals = {yr: {'f_total': 0, 'm_total': 0, 'fpercents': []} for yr in year_ranges}
    print(totals)
    for cat_name, prefixes in cat_to_pref.items():
        terms = mesh[mesh.MN.apply(lambda x: check_term_prefix(x, prefixes))].MH.sum()
        #print(f"{cat_name}: found {len(terms)} mesh terms. Examples: {terms[:10]}")
        matching_abstracts = abstracts[abstracts['mesh_headings'].apply(lambda x: contains_terms(x, terms))]
        print(f"{cat_name} found {len(terms)} mesh terms which match {len(matching_abstracts)} abstracts")
        if len(matching_abstracts) == 0:
            continue
        remove_single_sex_value = remove_single_sex
        if cat_name == 'Neoplasms':
            remove_single_sex_value = True
        nct_counter = count_participants_in_disease_ncts_by_year(matching_abstracts, remove_outliers, remove_single_sex_value)
        print('years,total_fpercent,median_fpercent,mean,std,n,CI')
        prev_f_total, prev_m_total = None, None
        prev_fprecents = None
        for yr in year_ranges:
            f_total, m_total = 0, 0
            fpercents = []
            for year in range(yr[0], yr[1] + 1):
                if year in nct_counter:
                    f_total += nct_counter[year]['F']
                    m_total += nct_counter[year]['M']
                    fpercents += nct_counter[year]['fpercents']
            totals[yr]['f_total'] += f_total
            totals[yr]['m_total'] += m_total
            totals[yr]['fpercents'].extend(fpercents)

            print(
                f'{yr[0]}-{yr[1]},{f_total / (f_total + m_total)},'
                f'{statistics.median(fpercents)},{np.mean(fpercents)},{np.std(fpercents)},{len(fpercents)},'
                f'{1.96 * np.std(fpercents) / np.sqrt(len(fpercents))}')
            if prev_f_total is not None:
                calculate_proportion_diff(f_total, m_total, prev_f_total, prev_m_total)
            prev_f_total, prev_m_total = f_total, m_total
            if prev_fprecents is not None:
                statistic, pval = ttest_ind(fpercents, prev_fprecents, equal_var=False)
                print(f"mean fpercent: proportion diff test statistic: {statistic} , 2 tailed p-value: {pval}")
            prev_fprecents = fpercents
    print("calculating for all mesh terms combined")
    # totals = {yr: {'f_total': 0, 'm_total': 0, 'fpercents': []} for yr in year_ranges}
    for yr in year_ranges:
        fpercents = totals[yr]["fpercents"]
        print(
            f'{yr[0]}-{yr[1]},{totals[yr]["f_total"] / (totals[yr]["f_total"] + totals[yr]["m_total"])},'
            f'{statistics.median(fpercents)},{np.mean(fpercents)},{np.std(fpercents)},{len(fpercents)},'
            f'{1.96 * np.std(fpercents) / np.sqrt(len(fpercents))}')
    if len(year_ranges) == 2:
        yr1, yr2 = year_ranges
        calculate_proportion_diff(totals[yr1]['f_total'], totals[yr1]['m_total'], totals[yr2]['f_total'], totals[yr2]['m_total'])
        statistic, pval = ttest_ind(totals[yr1]['fpercents'], totals[yr2]['fpercents'], equal_var=False)
        print(f"mean fpercent: proportion diff test statistic: {statistic} , 2 tailed p-value: {pval}")


def calculate_proportion_diff(f1, m1, f2, m2):
    n1 = f1 + m1
    n2 = f2 + m2
    p1 = f1 / n1
    p2 = f2 / n2
    p = (f1+f2) / (n1 + n2)
    Z = (p1 - p2) / (p * (1 - p) * np.sqrt(1.0 / n1 + 1.0 / n2))
    print(f"total fpercent: proportion diff test: Z={Z}, 2 tailed p-value: {2 * dist.norm.cdf(-np.abs(Z))}")


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


if __name__ == '__main__':
    count_cui_appearances_over_time(output_file='data/cuis_over_time.csv', raw_participant_numbers=True)
