import os

from sklearn.metrics import mean_squared_error, auc, roc_auc_score

from contra.constants import LOS_TEST_PATH, LOS_TEST_PATH_V2, READMIT_TEST_PATH, LOS_TEST_PATH_V3
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
from ast import literal_eval
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from contra.datasets.test_by_diags_dataset import ReadmissionbyEmbDiagsModule
from contra.constants import DATA_PATH


def make_bow(list_of_lists, word_to_index):
    matrix = lil_matrix((len(list_of_lists), len(word_to_index)))
    for row, l in enumerate(list_of_lists):
        for word in l:
            if word not in word_to_index:
                continue
            matrix[row, word_to_index[word]] += 1
    return matrix.tocsr()

def build_vocab(list_of_lists):
    vocab = set()
    for diags in list_of_lists:
        for diag in diags:
            vocab.add(diag)

    index_to_diag = list(vocab)
    diag_to_index = {diag: i for i, diag in enumerate(index_to_diag)}
    return diag_to_index


def clean_nans(string_rep_of_seq):
    return string_rep_of_seq.replace('nan, ', '').replace(', nan', '').replace('nan', '')

def test_los_by_bow():
    df = pd.read_csv(LOS_TEST_PATH_V3, index_col=0)
    df = df.rename({'ICD9_CODE': 'DIAGS'}, axis=1)
    diags_field = 'DIAGS'
    prev_diags_field = 'PREV_DIAGS'
    main_diags_field = prev_diags_field
    primary_diag_field = 'PRIMARY'

    df = df[df.AGE > 17]

    # print("removing rows with negative LOS (patient died)")
    # df = df[df.LOS > 0]

    print("Removing top 5% of LOS")
    before = len(df)
    longest_LOS = df.LOS.quantile(0.95)
    df = df[df.LOS <= longest_LOS]
    print(f"Kept {len(df)}/{before} records.")

    df = df.dropna(subset=[diags_field, 'LOS'])
    # print(f"samples: {len(df)}")
    df[diags_field] = df[diags_field].apply(clean_nans).apply(literal_eval)
    df = df[df[diags_field].apply(lambda x: len(x)>0)]

    #adding prev diags
    df[prev_diags_field] = df[prev_diags_field].apply(clean_nans).apply(literal_eval)
    # df = df[df[prev_diags_field].apply(lambda x: len(x) > 0)]
    # print(f"removed lines without prev diags. records: {len(df)}")

    print("clipped LOS to 100")
    df.LOS = df.LOS.clip(upper=100)

    df[primary_diag_field] = df['DIAGS'].apply(lambda x: [x[0]])

    train = df[df.ASSIGNMENT == 'train'].copy()
    test = df[df.ASSIGNMENT == 'test'].copy()
    print(f"read {len(train)} train, {len(test)} test")

    diag_to_index = build_vocab(train[main_diags_field].values)

    print(f"# features: {len(diag_to_index)}")
    Xtrain = make_bow(train[main_diags_field], diag_to_index)
    ytrain = train.LOS.values
    Xtest = make_bow(test[main_diags_field], diag_to_index)
    ytest = test.LOS.values

    # adding first (primary) diag
    print("Adding primary diagnosis")
    mini_vocab = build_vocab(train[primary_diag_field].values)
    print(f"primary diag features: {len(mini_vocab)}")
    Xtrain_addition = make_bow(train[primary_diag_field], mini_vocab)
    Xtest_addition = make_bow(test[primary_diag_field], mini_vocab)
    Xtrain = np.concatenate([Xtrain.todense(), Xtrain_addition.todense()], axis=1)
    Xtest = np.concatenate([Xtest.todense(), Xtest_addition.todense()], axis=1)

    # print("Adding age feature")
    # Xtrain = np.concatenate([Xtrain.todense(), train.AGE.values.reshape(len(train), 1)], axis=1)
    # Xtest = np.concatenate([Xtest.todense(), test.AGE.values.reshape(len(test), 1)], axis=1)

    # print("Using only age feature")
    # Xtrain = train.AGE.values.reshape(-1, 1)
    # Xtest = test.AGE.values.reshape(-1, 1)

    model = Ridge()
    # model = RandomForestRegressor(n_estimators=30)
    # model = GradientBoostingRegressor(n_estimators=30)
    print("Created model. training...")
    model.fit(Xtrain, ytrain)
    ypred_train = model.predict(Xtrain)
    ypred_test = model.predict(Xtest)
    print("R^2 score on train: {}/1".format(model.score(Xtrain, ytrain)))
    print("MSE on train: {}".format(mean_squared_error(ytrain, ypred_train, squared=False)))
    print("R^2 score on test: {}/1".format(model.score(Xtest, ytest)))
    print("MSE on test: {}".format(mean_squared_error(ytest, ypred_test, squared=False)))

    # compare to constant value prediction
    mean_train = ytrain.mean()
    mean_prediction_train = np.ones(len(ytrain)) * mean_train
    mean_prediction_test = np.ones(len(ytest)) * mean_train
    print("MSE on train for constant prediction: {}".format(mean_squared_error(ytrain, mean_prediction_train, squared=False)))
    print("MSE on test for constant prediction: {}".format(mean_squared_error(ytest, mean_prediction_test, squared=False)))


def truncate_icd(list_of_icd9):
    new_list = []
    for code in list_of_icd9:
        if code.startswith("V") or code.startswith("E"):
            continue
        new_list.append(code[:3])
    return new_list


def test_readmission_by_bow(paper=2):
    diags_field = 'DIAGS'
    if paper == 1:
        emb_files = [os.path.join(DATA_PATH, 'embs', 'fem40_heur_emb.tsv'),
                     os.path.join(DATA_PATH, 'embs', 'neutral40_emb.tsv')]
        dm = ReadmissionbyEmbDiagsModule(emb_files, READMIT_TEST_PATH)
        dm.prepare_data()
        train = dm.train_df.copy()
        test = dm.val_df.copy()
        for field in [diags_field, 'PROCEDURES', 'PREV_DIAGS']:
            train[field] = train[field].apply(literal_eval)
            test[field] = test[field].apply(literal_eval)
    else:
        df = pd.read_csv(READMIT_TEST_PATH, index_col=0)
        df = df[df.AGE > 17]
        df = df[df.READMISSION != 2]  # remove patients who died in the ICU
        df = df.dropna(subset=[diags_field, 'READMISSION'])
        df[diags_field] = df[diags_field].apply(clean_nans).apply(literal_eval)
        df = df[df[diags_field].apply(lambda x: len(x) > 0)]

        df['PROCEDURES'] = df['PROCEDURES'].fillna('[]')
        df['PROCEDURES'] = df['PROCEDURES'].apply(clean_nans).apply(literal_eval)
        df['DRUG'] = df['DRUG'].fillna('[]')
        df['DRUG'] = df['DRUG'].apply(literal_eval)
        df['PREV_DIAGS'] = df['PREV_DIAGS'].apply(clean_nans).apply(literal_eval)

        # print("truncating icd9 codes to 3 digits and removing V and E")
        # df[diags_field] = df[diags_field].apply(truncate_icd)

        train = df[df.ASSIGNMENT == 'train'].copy()
        test = df[df.ASSIGNMENT == 'test'].copy()

    print(f"read {len(train)} train, {len(test)} test")

    diag_to_index = build_vocab(train[diags_field].values)
    print(f"# features: {len(diag_to_index)}, first ones: {list(diag_to_index.keys())[:5]}")
    # print(f"vocab: {diag_to_index}")
    # print(f"test vocab: {build_vocab(test[diags_field].values)}")
    Xtrain = make_bow(train[diags_field], diag_to_index)
    ytrain = train.READMISSION.values
    Xtest = make_bow(test[diags_field], diag_to_index)
    ytest = test.READMISSION.values

    print("adding prev diags")
    pd_to_index = build_vocab(train['PREV_DIAGS'].values)
    print(f"# features for PREV_DIAGS: {len(pd_to_index)},  first ones: {list(pd_to_index.keys())[:5]}")
    Xpd_train = make_bow(train['PREV_DIAGS'], pd_to_index)
    Xpd_test = make_bow(test['PREV_DIAGS'], pd_to_index)
    Xtrain = np.concatenate([Xtrain.todense(), Xpd_train.todense()], axis=1)
    Xtest = np.concatenate([Xtest.todense(), Xpd_test.todense()], axis=1)

    # print("adding procedures")
    # proc_to_index = build_vocab(train['PROCEDURES'].values)
    # print(f"# features for procedures: {len(proc_to_index)}")
    # Xprocs_train = make_bow(train['PROCEDURES'], proc_to_index)
    # Xprocs_test = make_bow(test['PROCEDURES'], proc_to_index)
    # Xtrain = np.concatenate([Xtrain.todense(), Xprocs_train.todense()], axis=1)
    # Xtest = np.concatenate([Xtest.todense(), Xprocs_test.todense()], axis=1)

    # print("adding drugs")
    # drug_to_index = build_vocab(train['DRUG'].values)
    # print(f"# features for drugs: {len(drug_to_index)}")
    # Xdrugs_train = make_bow(train['DRUG'], drug_to_index)
    # Xdrugs_test = make_bow(test['DRUG'], drug_to_index)
    # Xtrain = np.concatenate([Xtrain, Xdrugs_train.todense()], axis=1)
    # Xtest = np.concatenate([Xtest, Xdrugs_test.todense()], axis=1)

    # model = LogisticRegression(max_iter=100)
    # model = RandomForestClassifier(n_estimators=5)
    # model = SVC()
    model = XGBClassifier()

    print("Training...")
    model.fit(Xtrain, ytrain)
    ypred_train = model.predict(Xtrain)
    ypred_test = model.predict(Xtest)

    const_pred = ytrain.mean()

    print(f"AUC on train: {roc_auc_score(ytrain, ypred_train)}")
    print(f"AUC on test: {roc_auc_score(ytest, ypred_test)}")

    print(f"AUC on train for const prediction ({const_pred}): {roc_auc_score(ytrain, np.ones(len(ytrain))*const_pred)}")
    print(f"AUC on test for const prediction ({const_pred}): {roc_auc_score(ytest, np.ones(len(ytest))*const_pred)}")



if __name__ == '__main__':
    # test_los_by_bow()
    test_readmission_by_bow(paper=2)
