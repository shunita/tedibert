import sys
sys.path.append('/home/shunita/fairemb')
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.distance import cosine
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from transformers import AutoTokenizer, AutoModel
from contra.experimental.exp_utils import get_vocab, texts_to_BOW, get_binary_labels_from_df, read_abstracts, \
    count_old_new_appearances
from contra.utils.pubmed_utils import split_abstracts_to_sentences_df, clean_abstracts
from contra.utils.text_utils import TextUtils
from contra.common.utils import mean_pooling
from contra.constants import DATA_PATH, SAVE_PATH, EXP_PATH
from scipy.linalg import orthogonal_procrustes

tu = TextUtils()


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


def run_classifier_and_print_results(Xtrain, ytrain, Xtest, ytest, model_class, binary):
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
    return model


def classification_for_year(df, binary, by_sentence, model_class=LogisticRegression,
                            words_and_weights_file=None, sentence_analysis_file=None, shuffle=False):
    print("filtering sentences from abstracts")
    df = clean_abstracts(df)
    df['tokenized'] = df['title_and_abstract'].apply(tu.word_tokenize)
    vocab, index_to_word = get_vocab(df['tokenized'])
    train, test = train_test_split(df, test_size=0.3)
    train, test = train.copy(), test.copy()
    if by_sentence:
        keep_fields = ('date', 'year', 'female', 'male', 'all_participants', 'percent_female')
        train = split_abstracts_to_sentences_df(train, text_field='title_and_abstract', keep=keep_fields)
        test = split_abstracts_to_sentences_df(test, text_field='title_and_abstract', keep=keep_fields)
        train['tokenized'] = train['text'].apply(tu.word_tokenize)
        test['tokenized'] = test['text'].apply(tu.word_tokenize)
    if binary:
        train, ytrain = get_binary_labels_from_df(train)
        test, ytest = get_binary_labels_from_df(test)
    else:
        ytrain = train['year']
        ytest = test['year']

    Xtrain = texts_to_BOW(train['tokenized'], vocab)
    Xtest = texts_to_BOW(test['tokenized'], vocab)

    model = run_classifier_and_print_results(Xtrain, ytrain, Xtest, ytest, model_class, binary)
    if words_and_weights_file is not None:
        words_and_weights = list(zip(index_to_word, model.coef_.squeeze()))
        words_df = pd.DataFrame(words_and_weights, columns=['word', 'weight'])
        df, _ = get_binary_labels_from_df(df)
        word_to_appearances = count_old_new_appearances(df)
        words_df['old_appearances'] = pd.Series([word_to_appearances[w][0] for w in index_to_word])
        words_df['new_appearances'] = pd.Series([word_to_appearances[w][1] for w in index_to_word])
        words_df.to_csv(words_and_weights_file)
    # Find which sentences confuse the model and which are easy
    # extract: sentence, probability, actual label
    if by_sentence and sentence_analysis_file is not None:
        train['model_prob'] = model.predict_proba(Xtrain)[:, 1]
        train[['text', 'year', 'label', 'model_prob']].to_csv(sentence_analysis_file)
    if shuffle:
        # shuffle the data for a random baseline.
        shuff_Xtrain = shuffle_csr(Xtrain)
        shuff_Xtest = shuffle_csr(Xtest)
        print("Results on shuffled X:")
        model = run_classifier_and_print_results(shuff_Xtrain, ytrain, shuff_Xtest, ytest, model_class, binary)
    return model


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
    #bert_model = AutoModel.from_pretrained(os.path.join(SAVE_PATH, 'bert_tiny_uncased_2010_2018_v2020_epoch39'))

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

    model = run_classifier_and_print_results(Xtrain, ytrain, Xtest, ytest, model_class, binary)


def count_cui_appearances_by_year(CUI_names_list):
    abstracts = pd.read_csv(os.path.join(DATA_PATH, 'abstracts_and_population_tokenized_for_cui2vec.csv'), index_col=0)
    abstracts = abstracts.dropna(subset=['tokenized'])
    abstracts_years = pd.read_csv(os.path.join(DATA_PATH, 'abstracts_population_date_topics.csv'), index_col=0)
    abstracts_years['year'] = abstracts_years['date'].apply(lambda x: int(x[-4:]))
    abstracts = abstracts.merge(abstracts_years['year'], left_index=True, right_index=True)
    counter = {cui: defaultdict(int) for cui in CUI_names_list}
    for i, r in abstracts.iterrows():
        year = r['year']
        for w in r['tokenized'].split():
            if w in counter:
                counter[w][year] += 1
    return counter


def CUI_diff_bert():
    df = pd.read_csv(os.path.join(DATA_PATH, 'cui_table_for_cui2vec_with_abstract_counts.csv'))
    CUI_names = df['name'].tolist()
    counter = count_cui_appearances_by_year(CUI_names)
    df['cui_tf_11_13'] = [counter[cui][2011] + counter[cui][2012] + counter[cui][2013] for cui in CUI_names]
    df['cui_tf_16_18'] = [counter[cui][2016] + counter[cui][2017] + counter[cui][2018] for cui in CUI_names]
    df.to_csv()

    bert_tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
    bert1 = AutoModel.from_pretrained(os.path.join(SAVE_PATH, 'bert_tiny_uncased_2011_2013_v2020_epoch39'))
    bert2 = AutoModel.from_pretrained(os.path.join(SAVE_PATH, 'bert_tiny_uncased_2016_2018_v2020_epoch39'))
    CUI_embeddings1 = embed_with_bert(CUI_names, bert1, bert_tokenizer)  # first matrix
    CUI_embeddings2 = embed_with_bert(CUI_names, bert2, bert_tokenizer)  # second matrix
    proc_matrix, _ = orthogonal_procrustes(CUI_embeddings1, CUI_embeddings2)
    aligned_CUI_embeddings1 = CUI_embeddings1 @ proc_matrix
    dists = [cosine(aligned_CUI_embeddings1[i], CUI_embeddings2[i]) for i in range(len(CUI_names))]
    df['cosine_dist_old_v_new'] = dists
    df.to_csv(os.path.join(SAVE_PATH, 'CUI_emb_diff_old_v_new_procrustes.csv'))


if __name__ == "__main__":
    # df = read_abstracts()
    # vocab, Xtrain, Xtest, ytrain, ytest = regression_for_percent_female(df)

    # df = read_abstracts(tokenize=False)
    # print("BOW sentence representation:")
    # sent_file = os.path.join(EXP_PATH, 'sentence_analysis_after_filter.csv')
    # words_file = os.path.join(EXP_PATH, 'BOW_words_and_weights_old_new_by_abstract_after_filter.csv')
    # classification_for_year(df, binary=True, by_sentence=True, model_class=LogisticRegression,
    #                         #words_and_weights_file=words_file,
    #                         sentence_analysis_file=sent_file
    #                         )
    #print("avg BERT sentence representation:")
    #classification_for_year_with_bert(df, binary=True, by_sentence=True, model_class=LogisticRegression)
    CUI_diff_bert()
