import os
from scipy import stats
import pandas as pd
from torch import nn
from contra.datasets import pubmed_dataset as pmd
from contra.models.w2v_on_years import read_w2v_model
from contra.utils.text_utils import TextUtils
from contra.constants import SAVE_PATH


def calculate_sims_and_compare(cui_similarity_file, w2v_years, save_to_file=None):
    sim_df = pd.read_csv(cui_similarity_file, index_col=0)  # CUI1, CUI2, similarity
    tokenizer = TextUtils()
    w2v_model = read_w2v_model(*w2v_years)
    sim_df['CUI1_tokenized'] = sim_df['CUI1'].apply(tokenizer.word_tokenize_abstract)
    sim_df['CUI2_tokenized'] = sim_df['CUI2'].apply(tokenizer.word_tokenize_abstract)
    emb1 = w2v_model.embed_batch(sim_df['CUI1_tokenized'].values)
    emb2 = w2v_model.embed_batch(sim_df['CUI2_tokenized'].values)
    pred_similarity = nn.CosineSimilarity()(emb1, emb2)
    true_similarity = sim_df['similarity']
    df = pd.DataFrame({'pred_similarity': pred_similarity, 'true_similarity': true_similarity})
    if save_to_file is not None:
        df.to_csv(os.path.join(SAVE_PATH, save_to_file))
    df = df.sort_values(['true_similarity'], ascending=False).reset_index()
    true_rank = list(df.index)
    pred_rank = list(df.sort_values(['pred_similarity'], ascending=False).index)
    correlation, pvalue = stats.spearmanr(true_rank, pred_rank)
    print(f"correlation: {correlation}. Pvalue: {pvalue}")
    return correlation, pvalue


if __name__ == "__main__":
    # TODO: need to filter out cuis that don't appear in all baseline models
    cui_dataset = pmd.CUIDataset(bert=None, w2v_years=(2019, 2019), top_percentile=0.5,
                                 semtypes=['dsyn'], frac=0.001, sample_type=1, read_from_file=None,
                                 save_to_file='test_similarities_2019.csv')
    print("W2V years: 2018")
    calculate_sims_and_compare(os.path.join(SAVE_PATH, 'test_similarities_2019.csv'),
                               w2v_years=(2018, 2018),
                               save_to_file=None)

    # print("W2V years: 2010-2013")
    # calculate_sims_and_compare(os.path.join(SAVE_PATH, 'test_similarities_2019.csv'),
    #                            w2v_years=(2010, 2013),
    #                            save_to_file=None)

    # print("W2V years: 2010-2018")
    # calculate_sims_and_compare(os.path.join(SAVE_PATH, 'test_similarities_2019.csv'),
    #                            w2v_years=(2010, 2018),
    #                            save_to_file=None)
