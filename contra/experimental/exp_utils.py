import numpy as np
from collections import defaultdict
from scipy.sparse import lil_matrix
from tqdm import tqdm
from contra.utils.pubmed_utils import load_aact_data, clean_abstracts
from contra.utils.text_utils import TextUtils

tu = TextUtils()

def get_vocab(list_of_word_lists):
    vocab_set = set()
    for lst in list_of_word_lists:
        for w in lst:
            vocab_set.add(w)
    vocab = sorted(list(vocab_set))
    word_to_index = {w: i for i, w in enumerate(vocab)}
    print("vocab size: {}".format(len(vocab)))
    return word_to_index, vocab


def texts_to_BOW(texts_list, vocab):
    """embed_with_bert abstracts using BOW. (set representation).
    :param texts_list: list of abstracts, each is a list of words.
    :param vocab: dictionary of word to index
    :return:
    """
    X = lil_matrix((len(texts_list), len(vocab)))
    for i, abstract in tqdm(enumerate(texts_list), total=len(texts_list)):
        # if the word is unknown we ignore it (could possibly add UNK token)
        word_indices = [vocab[w] for w in sorted(set(abstract)) if w in vocab]
        X[i, word_indices] = 1
    return X.tocsr()


def text_seq_to_BOW(word_list, vocab):
    x = np.zeros(len(vocab))
    word_indices = [vocab[w] for w in sorted(set(word_list)) if w in vocab]
    x[word_indices] = 1
    return x


def year_to_binary_label(year, first_time_range, second_time_range):
    if first_time_range[0].year <= year <= first_time_range[1].year:
        return 0
    if second_time_range[0].year <= year <= second_time_range[1].year:
        return 1


def fill_binary_year_label(df, first_time_range, second_time_range):
    df['label'] = df['year'].apply(lambda x: year_to_binary_label(x, first_time_range, second_time_range))
    df = df.dropna(subset=['label'])
    return df


def get_binary_labels_from_df(df):
    df['label'] = df['year'].apply(year_to_binary_label)
    df = df.dropna(subset=['label'])
    return df, df['label'].values


def read_abstracts(tokenize=True):
    df = load_aact_data(2019)
    df['all_participants'] = df['male'] + df['female']
    df['percent_female'] = df['male'] / df['all_participants']
    if tokenize:
        df['tokenized'] = df['title_and_abstract'].apply(tu.word_tokenize)
    return df


def count_old_new_appearances(df_with_old_new_label):
    if df_with_old_new_label is None:
        df = read_abstracts(tokenize=False)
        df = clean_abstracts(df)
        df['tokenized'] = df['title_and_abstract'].apply(tu.word_tokenize)
        df, _ = get_binary_labels_from_df(df)
        df_with_old_new_label = df

    d = defaultdict(lambda: {0: 0, 1: 0})

    def word_origins(text, label):
        for w in text:
            d[w][label] += 1

    for i in (0, 1):
        df_with_old_new_label[df_with_old_new_label.label == i]['tokenized'].apply(
            lambda x: word_origins(x, i))
    return d
