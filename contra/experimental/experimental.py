import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from xgboost import XGBClassifier

import sys
sys.path.append('/home/shunita/fairemb')

from contra.utils.pubmed_utils import split_abstracts_to_sentences_df, load_aact_data
from contra.utils.text_utils import TextUtils

from transformers import AutoTokenizer, AutoModel
from contra.common.utils import mean_pooling
import os
from scipy.spatial.distance import cosine
from contra.constants import DATA_PATH, SAVE_PATH

tu = TextUtils()


def filter_word_list(word_list):
    # decide_on_word removes punctuation, lowercases the words and replaces numbers with a specific token.
    return [w for w in map(TextUtils.decide_on_word, word_list) if len(w) > 0]


def word_tokenize(text):
    return filter_word_list(tu.word_tokenize_abstract(text))


def read_abstracts():
    df = load_aact_data(2019)
    df['all_participants'] = df['male'] + df['female']
    df['percent_female'] = df['male'] / df['all_participants']
    df['tokenized'] = df['title_and_abstract'].apply(word_tokenize)
    return df



def get_vocab(list_of_word_lists):
    vocab_set = set()
    for lst in list_of_word_lists:
        for w in lst:
            vocab_set.add(w)
    vocab = sorted(list(vocab_set))
    word_to_index = {w: i for i, w in enumerate(vocab)}
    print("vocab size: {}".format(len(vocab)))
    return word_to_index, vocab
# get_vocab(abstracts['tokenized'])


def texts_to_BOW(texts_list, vocab):
    """embed_with_bert abstracts using BOW. (set representation).
    :param texts_list: list of abstracts, each is a list of words.
    :param vocab: dictionary of word to index
    :return:
    """
    X = lil_matrix((len(texts_list), len(vocab)))
    for i, abstract in tqdm(enumerate(texts_list), total=len(texts_list)):
        word_indices = [vocab[w] for w in sorted(set(abstract))]
        X[i, word_indices] = 1
    return X.tocsr()


def shuffle_csr(mat):
    # utility function to shuffle sparse csr_matrix rows
    index = np.arange(mat.shape[0])
    np.random.shuffle(index)
    return mat[index, :]


def MSE_score(X, y, model):
    pred = model.predict(X)
    return mean_squared_error(y, pred)


def regression_for_percent_female(df):
    """regression for the percent of female participants."""
    vocab, index_to_word = get_vocab(df['tokenized'])
    X = texts_to_BOW(df['tokenized'], vocab)
    y = df['percent_female']
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)

    # shuffle the data for a random baseline.
    shuff_Xtrain = shuffle_csr(Xtrain)
    shuff_Xtest = shuffle_csr(Xtest)

    models = [("Vanilla Linear Regression", LinearRegression),
              ("Ridge Regression", Ridge),
              ("Lasso Regression", Lasso)]
    for model_desc, model_class in models:
        model = model_class()
        print("\n\n{}:".format(model_desc))
        model.fit(Xtrain, ytrain)
        print("score on train: {}/1".format(model.score(Xtrain, ytrain)))
        print("MSE on train: {}".format(MSE_score(Xtrain, ytrain, model)))

        print("score on test: {}/1".format(model.score(Xtest, ytest)))
        print("MSE on test: {}".format(MSE_score(Xtest, ytest, model)))

        c = model.coef_
        words_and_weights = zip(index_to_word, c)
        print("coefficients: min={}, max={}, mean:{}".format(max(c), min(c), np.mean(c)))
        prominent = sorted(words_and_weights, key=lambda x: np.abs(x[1]), reverse=True)[:10]
        print("top 10 prominent features: {}".format(prominent))

        model = model_class()
        model.fit(shuff_Xtrain, ytrain)
        print("score on random train: {}/1".format(model.score(shuff_Xtrain, ytrain)))
        print("MSE on random train: {}".format(MSE_score(shuff_Xtrain, ytrain, model)))
        print("score on random test: {}/1".format(model.score(shuff_Xtest, ytest)))
        print("MSE on random test: {}".format(MSE_score(shuff_Xtest, ytest, model)))

    return vocab, Xtrain, Xtest, ytrain, ytest


def year_to_binary_label(year):
    if 2010 <= year <= 2013:
        return 0
    if 2016 <= year <= 2018:
        return 1


def get_binary_labels_from_df(df):
    df['label'] = df['year'].apply(year_to_binary_label)
    df = df.dropna(subset=['label'])
    return df, df['label']


def classification_for_year(df, binary, by_sentence, model_class=LogisticRegression):
    vocab, index_to_word = get_vocab(df['tokenized'])
    train, test = train_test_split(df, test_size=0.3)
    train, test = train.copy(), test.copy()
    keep_fields = ('date', 'year', 'female', 'male', 'all_participants', 'percent_female')
    if by_sentence:
        train = split_abstracts_to_sentences_df(train, text_field='title_and_abstract', keep=keep_fields)
        test = split_abstracts_to_sentences_df(test, text_field='title_and_abstract', keep=keep_fields)
        train['tokenized'] = train['text'].apply(word_tokenize)
        test['tokenized'] = test['text'].apply(word_tokenize)
    if binary:
        train, ytrain = get_binary_labels_from_df(train)
        test, ytest = get_binary_labels_from_df(test)
    else:
        ytrain = train['year']
        ytest = test['year']

    Xtrain = texts_to_BOW(train['tokenized'], vocab)
    Xtest = texts_to_BOW(test['tokenized'], vocab)

    # shuffle the data for a random baseline.
    shuff_Xtrain = shuffle_csr(Xtrain)
    shuff_Xtest = shuffle_csr(Xtest)

    model = model_class()
    model.fit(Xtrain, ytrain)
    ypred_train = model.predict_proba(Xtrain)[:, 1]
    ypred_test = model.predict_proba(Xtest)[:, 1]
    print(f"Logloss on train: {log_loss(ytrain, model.predict_proba(Xtrain))}")
    print(f"Logloss on test: {log_loss(ytest, model.predict_proba(Xtest))}")
    if binary:
        print(f"Accuracy on train: {accuracy_score(ytrain, model.predict(Xtrain))}")
        print(f"Accuracy on test: {accuracy_score(ytest, model.predict(Xtest))}")
        print(f"AUC on train: {roc_auc_score(ytrain, ypred_train)}")
        print(f"AUC on test: {roc_auc_score(ytest, ypred_test)}")

    model = model_class()
    model.fit(shuff_Xtrain, ytrain)
    ypred_train = model.predict_proba(shuff_Xtrain)[:, 1]
    ypred_test = model.predict_proba(shuff_Xtest)[:, 1]
    print(f"Logloss on train: {log_loss(ytrain, model.predict_proba(shuff_Xtrain))}")
    print(f"Logloss on test: {log_loss(ytest, model.predict_proba(shuff_Xtest))}")
    if binary:
        print(f"Accuracy on random train: {accuracy_score(ytrain, model.predict(Xtrain))}")
        print(f"Accuracy on random test: {accuracy_score(ytest, model.predict(Xtest))}")
        print(f"AUC on random train: {roc_auc_score(ytrain, ypred_train)}")
        print(f"AUC on random test: {roc_auc_score(ytest, ypred_test)}")


def embed_with_bert(texts, bert_model, bert_tokenizer):
    inputs = bert_tokenizer(texts, padding=True, return_tensors="pt", truncation=True, max_length=512)
    outputs = bert_model(**inputs)
    embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask']).detach().numpy()
    #  print(f"emb shape: {embeddings.shape}")  # len(texts) * 128
    return embeddings


def embed_in_batches(texts, bert_model, bert_tokenizer, batch_size=256):
    batches = np.array_split(texts, len(texts)//batch_size)
    embs = []
    for batch in tqdm(batches):
        embs.append(embed_with_bert(batch.tolist(), bert_model, bert_tokenizer))
    return np.concatenate(embs, axis=0)


def embed_abstracts(abstracts, bert_model, bert_tokenizer):
    embs = []
    for abstract in tqdm(abstracts):
        sentences = tu.split_abstract_to_sentences(abstract)
        sent_embeddings = embed_with_bert(sentences, bert_model, bert_tokenizer)
        embs.append(np.mean(sent_embeddings, axis=0))  # shape should be (128,)
    return np.stack(embs)


def classification_for_year_with_bert(df, binary, by_sentence, model_class=LogisticRegression):
    train, test = train_test_split(df, test_size=0.3)
    train, test = train.copy(), test.copy()
    keep_fields = ('date', 'year', 'female', 'male', 'all_participants', 'percent_female')
    if by_sentence:
        train = split_abstracts_to_sentences_df(train, text_field='title_and_abstract', keep=keep_fields)
        test = split_abstracts_to_sentences_df(test, text_field='title_and_abstract', keep=keep_fields)
    if binary:
        train, ytrain = get_binary_labels_from_df(train)
        test, ytest = get_binary_labels_from_df(test)
    else:
        ytrain = train['year']
        ytest = test['year']

    # Embed using bert - which bert?
    # Use tinybert (not trained on med)
    bert_tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
    bert_model = AutoModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
    #bert_model = AutoModel.from_pretrained(os.path.join(SAVE_PATH, 'bert_tiny_uncased_2011_2013_v2020_epoch39'))

    if by_sentence:
        print("Embedding train using bert:")
        Xtrain = embed_in_batches(train['text'].values, bert_model, bert_tokenizer)
        print("Embedding test using bert:")
        Xtest = embed_in_batches(test['text'].values, bert_model, bert_tokenizer)
    else:
        print("Embedding train using bert:")
        Xtrain = embed_abstracts(train['title_and_abstract'].values, bert_model, bert_tokenizer)
        print("Embedding test using bert:")
        Xtest = embed_abstracts(test['title_and_abstract'].values, bert_model, bert_tokenizer)

    model = model_class()
    model.fit(Xtrain, ytrain)
    ypred_train = model.predict_proba(Xtrain)[:, 1]
    ypred_test = model.predict_proba(Xtest)[:, 1]
    print(f"Logloss on train: {log_loss(ytrain, model.predict_proba(Xtrain))}")
    print(f"Logloss on test: {log_loss(ytest, model.predict_proba(Xtest))}")

    if binary:
        print(f"Accuracy on train: {accuracy_score(ytrain, model.predict(Xtrain))}")
        print(f"Accuracy on test: {accuracy_score(ytest, model.predict(Xtest))}")
        print(f"AUC on train: {roc_auc_score(ytrain, ypred_train)}")
        print(f"AUC on test: {roc_auc_score(ytest, ypred_test)}")



def CUI_diff_bert():
    df = pd.read_csv(os.path.join(DATA_PATH, 'cui_table_for_cui2vec_with_abstract_counts.csv'))

    CUI_names = df['name'].tolist()
    bert_tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
    bert1 = AutoModel.from_pretrained(os.path.join(SAVE_PATH, 'bert_tiny_uncased_2011_2013_v2020_epoch39'))
    bert2 = AutoModel.from_pretrained(os.path.join(SAVE_PATH, 'bert_tiny_uncased_2016_2018_v2020_epoch39'))
    CUI_embeddings1 = embed_with_bert(CUI_names, bert1, bert_tokenizer)
    CUI_embeddings2 = embed_with_bert(CUI_names, bert2, bert_tokenizer)
    dists = [cosine(CUI_embeddings1[i], CUI_embeddings2[i]) for i in range(len(CUI_names))]
    df['cosine_dist_old_v_new'] = dists
    df.to_csv(os.path.join(SAVE_PATH, 'CUI_emb_diff_old_v_new.csv'))


if __name__ == "__main__":
    # df = read_abstracts()
    # vocab, Xtrain, Xtest, ytrain, ytest = regression_for_percent_female(df)

    df = read_abstracts()
    print("BOW sentence representation:")
    classification_for_year(df, binary=True, by_sentence=True, model_class=LogisticRegression)
    print("avg BERT sentence representation:")
    classification_for_year_with_bert(df, binary=True, by_sentence=True, model_class=LogisticRegression)
    #CUI_diff_bert()
