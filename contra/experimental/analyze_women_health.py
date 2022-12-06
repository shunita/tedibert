import sys
sys.path.append('/home/shunita/fairemb/')
import pandas as pd
import os
from contra.constants import FULL_PUMBED_2022_PATH, SAVE_PATH
from bertopic import BERTopic

journals = pd.read_csv('data/women_health_journals.csv')
# journal_names = journals.name.values
journal_short_names = journals.pubmed_short_name.values


def name_in_journal_list(name, list):
    if pd.isna(name):
        return False
    if name in list or name.upper() in list:
        return True
    return False


def read_from_relevant_journals(start_year=None):
    files = sorted(os.listdir(FULL_PUMBED_2022_PATH))
    for fname in files:
        print(fname)
        year = int(fname.split('_')[1].split(".")[0])
        if start_year is not None and year < start_year:
            continue
        df = pd.read_csv(os.path.join(FULL_PUMBED_2022_PATH, fname))
        subset = df[#df.journal_name.apply(lambda x: name_in_journal_list(x, journal_names)) |
                    df.journal_abbrv.apply(lambda x: name_in_journal_list(x, journal_short_names))]
        write_header = not os.path.exists('data/women_health_abstracts.csv')
        subset.to_csv('data/women_health_abstracts.csv', mode='a', header=write_header)

def data_stats():
    df = pd.read_csv('data/women_health_abstracts.csv', index_col=0)
    df.groupby('journal_name')['year'].agg(['count', 'min', 'max']).to_csv('exp_results/women_health/journal_stats.csv')
    df.groupby('year')['title'].count().to_csv('exp_results/women_health/year_histogram.csv')

def topic_modelling(df):
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(df.abstract.values)
    topic_model.save('exp_results/women_health/women_health_topics.model')
    fig = topic_model.visualize_barchart()
    fig.write_html('exp_results/women_health/topics_barchart.html')
    fig2 = topic_model.visualize_topics()
    fig2.write_html('exp_results/women_health/topics_map.html')


if __name__ == '__main__':
    # read_from_relevant_journals()
    df = pd.read_csv('data/women_health_abstracts.csv', index_col=0)
    topic_modelling(df)
