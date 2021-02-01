import pandas as pd
from collections import defaultdict


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
