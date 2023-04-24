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
    df['topic'] = topics
    df['probability'] = probs
    df.to_csv('exp_results/women_health/abstracts_and_topics.csv')
    topic_model.save('exp_results/women_health/women_health_topics.model')
    fig = topic_model.visualize_barchart()
    fig.write_html('exp_results/women_health/topics_barchart.html')
    fig2 = topic_model.visualize_topics()
    fig2.write_html('exp_results/women_health/topics_map.html')

def apply_topic_model(df, model_path):
    model = BERTopic.load(model_path)
    info_df = model.get_topic_info()
    for i in range(1, len(info_df)):
        info_df.loc[i, 'top_words'] = ';'.join(['{}:{:.4f}'.format(w, p) for w, p in model.get_topic(i-1)])
    topics = model.transform(df.abstract.values)

def analyze_labelled_topics(path):
    df = pd.read_csv(path, index_col=0)
    df = df[~df.category.isna()]
    df['category'] = df['category'].apply(lambda x: x.strip())

    df['category'] = df['category'].apply(lambda x: x.replace('heath', 'health'))
    df['category'] = df['category'].apply(lambda x: x.replace('onology', 'oncology'))
    df['category'] = df['category'].apply(lambda x: x.replace('neoplasm/oncology', 'neoplasm-oncology'))
    df['category'] = df['category'].apply(lambda x: x.replace('neoplasm/ oncology', 'neoplasm-oncology'))
    df['category'] = df['category'].apply(lambda x: x.replace('miscarrige', 'miscarriage'))

    non_repro = ['domestic violence', 'endocrinology', 'endometriosis', 'faculty', 'hematology', 'immunology', 'menopause', 'menstration', 'neoplasm-oncology', 'psychology/psychiatry', 'sexual function', 'sociology', 'STD', 'urogynecology', 'uterine health', 'vaginal & vulval health', 'UTI', 'HPV', 'PCOS', 'male sexual health', 'breast health']
    repro = ['abortion ', 'breastfeeding', 'child care & development', 'contraception', 'delivery', 'fertility', 'fetal health', 'genetics', 'induction', 'male fertility', 'miscarriage', 'multiple pregnancy', 'perinatal', 'preganancy']

if __name__ == '__main__':
    # read_from_relevant_journals()
    df = pd.read_csv('data/women_health_abstracts.csv', index_col=0)
    topic_modelling(df)
