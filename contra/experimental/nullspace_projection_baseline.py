import os
import random
import sys

import sklearn
from sklearn.neural_network import MLPClassifier

sys.path.append(os.path.expanduser('~/fairemb'))
sys.path.append(os.path.expanduser('~/fairemb/nullspace_projection'))
sys.path.append(os.path.expanduser('~/fairemb/nullspace_projection/src'))

import tqdm

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from gensim.models import KeyedVectors

from contra.constants import READMIT_TEST_PATH, LOS_TEST_PATH_V4, SAVE_PATH
from ast import literal_eval
import debias
from sklearn.svm import LinearSVC, SVC

MIMIC_PATH = os.path.expanduser("~/mimic3/physionet.org/files/mimiciii/1.4/")
OUTPUT_PATH = os.path.expanduser("~/fairemb/exp_results/nullspace_projection")
FROZEN_EMB_PATH = os.path.join(OUTPUT_PATH, 'BERT_tiny_medical_diags_and_drugs_w2v_format.txt')

def clean_nans(string_rep_of_seq):
    return string_rep_of_seq.replace('nan, ', '').replace(', nan', '').replace('nan', '')

def eval_code_list(icd9_codes_list_as_text):
    if pd.isna(icd9_codes_list_as_text):
        return []
    icd9_codes_list_as_text = clean_nans(icd9_codes_list_as_text)
    if pd.isna(icd9_codes_list_as_text):
        return []
    try:
        codes = literal_eval(icd9_codes_list_as_text)
    except:
        print(f"could not eval: {icd9_codes_list_as_text}")
        return []
    return codes


def collect_diagnoses():
    output_file = os.path.expanduser('~/fairemb/exp_results/nullspace_projection/all_diags_and_drugs.csv')
    if os.path.exists(output_file):
        return open(output_file, 'r').read().split('\n')
    df_readmission = pd.read_csv(READMIT_TEST_PATH, index_col=0)
    df_readmission['DIAGS'] = df_readmission['DIAGS'].apply(eval_code_list)
    readmission_diag_codes = set(df_readmission['DIAGS'].sum())
    print(f'readmission diags: {len(readmission_diag_codes)}')

    df_readmission['PREV_DIAGS'] = df_readmission['PREV_DIAGS'].apply(eval_code_list).apply(list)
    readmission_prev_diags = set(df_readmission['PREV_DIAGS'].sum())
    print(f'readmission prev diags: {len(readmission_prev_diags)}')

    df_los = pd.read_csv(LOS_TEST_PATH_V4, index_col=0)
    df_los['ICD9_CODE'] = df_los['ICD9_CODE'].apply(eval_code_list)
    los_diags = set(df_los['ICD9_CODE'].sum())
    print(f'los diags: {len(los_diags)}')

    df_los['PREV_DIAGS'] = df_los['PREV_DIAGS'].apply(eval_code_list).apply(list)
    los_prev_diags = set(df_los['PREV_DIAGS'].sum())
    print(f'los prev diags: {len(los_prev_diags)}')

    all_diags = readmission_diag_codes.union(readmission_prev_diags).union(los_diags).union(los_prev_diags)
    print(f"all diags: {len(all_diags)}")

    diag_dict = pd.read_csv(os.path.join(MIMIC_PATH, 'D_ICD_DIAGNOSES.csv'), index_col=0)
    diag_dict = diag_dict.set_index('ICD9_CODE')['LONG_TITLE'].to_dict()  # icd9 code to description
    all_diag_names = []
    not_found = []
    for code in all_diags:
        if code in diag_dict:
            all_diag_names.append(diag_dict[code])
        else:
            not_found.append(code)

    print(f"{len(not_found)} codes did not have a matching title.")
    f = open(os.path.expanduser('~/fairemb/exp_results/nullspace_projection/diags_without_title.txt'), 'w')
    f.write(str(not_found))
    f.close()

    df_readmission['DRUG'] = df_readmission['DRUG'].apply(eval_code_list)
    drugs = set(df_readmission['DRUG'].sum())
    print(f"readmission drugs: {len(drugs)}")
    ret = all_diag_names + list(drugs)
    f = open(FROZEN_EMB_PATH, 'w')
    for title in ret:
        f.write(title + "\n")
    f.close()
    return ret


def save_in_word2vec_format(vecs, words, fname):
    '''taken from: https://github.com/shauli-ravfogel/nullspace_projection/blob/master/notebooks/word_vectors_debiasing.ipynb'''
    with open(fname, "w", encoding="utf-8") as f:
        f.write(str(len(vecs)) + " " + "128" + "\n")
        for i, (v, w) in tqdm.tqdm(enumerate(zip(vecs, words))):
            vec_as_str = " ".join([str(x) for x in v])
            f.write(w.replace(" ", "_") + " " + vec_as_str + "\n")


def embed_diagnoses_in_w2v_style(list_of_titles):
    texts = list_of_titles + ["he", "she", "male", "female", "masculine", "feminine", "him", "her"]
    tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
    bert_model = AutoModel.from_pretrained(os.path.join(SAVE_PATH, 'bert_tiny_uncased_2010_2018_v2020_epoch39')).cuda()

    inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation=True,
                                         max_length=70,
                                         add_special_tokens=True, return_tensors="pt")
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    # each title is embedded - we take the CLS token embedding
    outputs = bert_model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0]
    vecs = outputs.detach().cpu().numpy()
    #save_in_word2vec_format(vecs, texts, FROZEN_EMB_PATH)

    f = open(os.path.join(OUTPUT_PATH, 'BERT_tiny_medical_diags_and_drugs_frozen_emb.tsv'), "w")
    for (w, v) in zip(texts, vecs):
        f.write(f"{w}\t{','.join([str(x) for x in v])}\n")
    f.close()


def load_word_vectors(fname):
    model = KeyedVectors.load_word2vec_format(fname, binary=False)
    vecs = model.vectors
    words = list(model.vocab.keys())
    return model, vecs, words


def project_on_gender_subspaces(gender_vector, model: KeyedVectors, n=2500):
    '''taken from: https://github.com/shauli-ravfogel/nullspace_projection/blob/master/notebooks/word_vectors_debiasing.ipynb'''
    group1 = model.similar_by_vector(gender_vector, topn=n, restrict_vocab=None)
    group2 = model.similar_by_vector(-gender_vector, topn=n, restrict_vocab=None)

    all_sims = model.similar_by_vector(gender_vector, topn=len(model.vectors), restrict_vocab=None)
    eps = 0.03
    idx = [i for i in range(len(all_sims)) if abs(all_sims[i][1]) < eps]
    samp = set(np.random.choice(idx, size=n))
    neut = [s for i, s in enumerate(all_sims) if i in samp]
    return group1, group2, neut


def get_vectors(word_list: list, model: KeyedVectors):
    vecs = []
    for w in word_list:
        vecs.append(model[w])
    vecs = np.array(vecs)
    return vecs

def null_it_out_pipeline():
    ### collect biased words ###
    num_vectors_per_class = 2500
    model, vecs, words = load_word_vectors(fname=FROZEN_EMB_PATH)
    gender_direction = model["he"] - model["she"]
    gender_unit_vec = gender_direction / np.linalg.norm(gender_direction)
    masc_words_and_scores, fem_words_and_scores, neut_words_and_scores = project_on_gender_subspaces(
        gender_direction, model, n=num_vectors_per_class)
    masc_words, masc_scores = list(zip(*masc_words_and_scores))
    neut_words, neut_scores = list(zip(*neut_words_and_scores))
    fem_words, fem_scores = list(zip(*fem_words_and_scores))
    masc_vecs, fem_vecs = get_vectors(masc_words, model), get_vectors(fem_words, model)
    neut_vecs = get_vectors(neut_words, model)
    n = min(3000, num_vectors_per_class)
    all_significantly_biased_words = masc_words[:n] + fem_words[:n]
    all_significantly_biased_vecs = np.concatenate((masc_vecs[:n], fem_vecs[:n]))
    all_significantly_biased_labels = np.concatenate((np.ones(n, dtype=int),
                                                      np.zeros(n, dtype=int)))

    all_significantly_biased_words, all_significantly_biased_vecs, all_significantly_biased_labels = sklearn.utils.shuffle(
        all_significantly_biased_words, all_significantly_biased_vecs, all_significantly_biased_labels)
    # print(np.random.choice(masc_words, size = 75))
    print("TOP MASC")
    print(masc_words[:50])
    # print("LAST MASC")
    # print(masc_words[-120:])
    print("-------------------------")
    # print(np.random.choice(fem_words, size = 75))
    print("TOP FEM")
    print(fem_words[:50])
    # print("LAST FEM")
    # print(fem_words[-120:])
    print("-------------------------")
    # print(np.random.choice(neut_words, size = 75))
    print(neut_words[:50])

    sys.exit()
    ### split to train, dev, test ###
    random.seed(0)
    np.random.seed(0)

    X = np.concatenate((masc_vecs, fem_vecs, neut_vecs), axis=0)
    # X = (X - np.mean(X, axis = 0, keepdims = True)) / np.std(X, axis = 0)
    y_masc = np.ones(masc_vecs.shape[0], dtype=int)
    y_fem = np.zeros(fem_vecs.shape[0], dtype=int)
    y_neut = -np.ones(neut_vecs.shape[0], dtype=int)
    # y = np.concatenate((masc_scores, fem_scores, neut_scores))#np.concatenate((y_masc, y_fem))
    y = np.concatenate((y_masc, y_fem, y_neut))
    X_train_dev, X_test, y_train_dev, Y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3,
                                                                                        random_state=0)
    X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(X_train_dev, y_train_dev, test_size=0.3,
                                                                              random_state=0)
    print("Train size: {}; Dev size: {}; Test size: {}".format(X_train.shape[0], X_dev.shape[0], X_test.shape[0]))

    ### debias ###
    gender_clf = LinearSVC
    # gender_clf = SGDClassifier
    # gender_clf = LogisticRegression
    # gender_clf = LinearDiscriminantAnalysis
    # gender_clf = Perceptron

    params_svc = {'fit_intercept': False, 'class_weight': None, "dual": False, 'random_state': 0}
    params_sgd = {'fit_intercept': False, 'class_weight': None, 'max_iter': 1000, 'random_state': 0}
    params = params_svc
    # params = {'loss': 'hinge', 'n_jobs': 16, 'penalty': 'l2', 'max_iter': 2500, 'random_state': 0}
    # params = {}
    n = 35
    min_acc = 0
    is_autoregressive = True
    dropout_rate = 0

    P, rowspace_projs, Ws = debias.get_debiasing_projection(gender_clf, params, n, 128, is_autoregressive, min_acc,
                                                            X_train, Y_train, X_dev, Y_dev,
                                                            Y_train_main=None, Y_dev_main=None,
                                                            by_class=False, dropout_rate=dropout_rate)
    np.save(os.path.join(OUTPUT_PATH, "P.MedicalBERT.dim=128.iters=35.npy"), P)

    X_dev_cleaned = (P.dot(X_dev.T)).T
    X_test_cleaned = (P.dot(X_test.T)).T
    X_trained_cleaned = (P.dot(X_train.T)).T

    # Test accuracy of recovering the gender from the embeddings
    print("Before, linear:")
    linear_clf = LinearSVC(dual=False, max_iter=1500)
    linear_clf.fit(X_train, Y_train)
    print(linear_clf.score(X_test, Y_test))

    print("After, linear:")
    linear_clf = LinearSVC(dual=False, max_iter=1500)
    linear_clf.fit(X_trained_cleaned, Y_train)
    print(linear_clf.score(X_test_cleaned, Y_test))

    print("After, rbf-svm:")
    nonlinear_clf = SVC(kernel="rbf")
    nonlinear_clf.fit(X_trained_cleaned, Y_train)
    print(nonlinear_clf.score(X_dev_cleaned, Y_dev))

    print("After, mlp:")
    nonlinear_clf = MLPClassifier(hidden_layer_sizes=256, activation="relu")

    nonlinear_clf.fit(X_trained_cleaned, Y_train)
    print(nonlinear_clf.score(X_dev_cleaned, Y_dev))

def save_debiased_embeddings():
    model, vecs, words = load_word_vectors(fname=FROZEN_EMB_PATH)
    P = np.load(os.path.join(OUTPUT_PATH, "P.MedicalBERT.dim=128.iters=35.npy"))
    vecs_cleaned = (P.dot(vecs.T)).T
    # save in frozen embedding format
    f = open(os.path.join(OUTPUT_PATH, 'BERT_tiny_medical_diags_and_drugs_debiased.tsv'), "w")
    for (w,v) in zip(words, vecs_cleaned):
        f.write(f"{w.replace('_',' ')}\t{','.join([str(x) for x in v])}\n")

if __name__ == '__main__':
    list_of_titles = collect_diagnoses()
    print(f"found {len(list_of_titles)} titles of diags and drugs")
    embed_diagnoses_in_w2v_style(list_of_titles)

    null_it_out_pipeline()
    # save_debiased_embeddings()