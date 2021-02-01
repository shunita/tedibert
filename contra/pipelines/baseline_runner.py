import sys
sys.path.append('/home/shunita/fairemb')

import os
from scipy import stats
import pandas as pd
import numpy as np
from itertools import product
import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
from contra.datasets import pubmed_dataset as pmd
from contra.models.w2v_on_years import read_w2v_model
from contra.utils.text_utils import TextUtils
from contra.constants import SAVE_PATH, DATA_PATH


def find_interesting_CUI_pairs(first_w2v_years, second_w2v_years, test_w2v_years,
                               semtypes=None, top_percentile=None, filter_by_models=[], sample_fraction=0.01,
                               write_to_file=None, read_w2v_params={}):
    """Find CUI pairs that have embeddings in all given models, and whose similarity has changed the most between
    the given time ranges."""
    # Read all CUIs
    df = pd.read_csv(os.path.join(DATA_PATH, 'cui_table_for_cui2vec_with_abstract_counts.csv'))
    # Possibly filter by semantic types
    if semtypes is not None:
        if not isinstance(semtypes, list):
            semtypes = [semtypes]
        df = df[df['semtypes'].map(lambda sems: np.any([sem in semtypes for sem in eval(sems)]))]
    # Possibly filter by how common the term is in abstracts.
    if top_percentile is not None:
        # filter only CUIs that appeared in enough abstracts
        df = df[df['abstracts'] >= df['abstracts'].quantile(1 - top_percentile)]
    CUI_names = df['name'].tolist()
    tokenizer = TextUtils()
    tokenized_names = [tokenizer.word_tokenize_abstract(name) for name in CUI_names]
    models = {first_w2v_years: read_w2v_model(first_w2v_years[0], first_w2v_years[1], **read_w2v_params),
              second_w2v_years: read_w2v_model(second_w2v_years[0], second_w2v_years[1], **read_w2v_params),
              test_w2v_years: read_w2v_model(test_w2v_years[0], test_w2v_years[1], **read_w2v_params)}
    for year_range in filter_by_models:
        models[year_range] = read_w2v_model(year_range[0], year_range[1])
    embs_dict = {}
    # Filter the CUIs according to the models we want to compare.
    # CUIs in the final test set should have embeddings in all compared models.
    got_all_embs = torch.tensor([True for i in range(len(CUI_names))])
    for year_range in [first_w2v_years, second_w2v_years, test_w2v_years]+filter_by_models:
        print(f"Filtering CUIs by {year_range[0]}_{year_range[1]} model.")
        model = models[year_range]
        embs_dict[year_range] = model.embed_batch(tokenized_names)
        got_emb = torch.count_nonzero(embs_dict[year_range], dim=1) > 0
        got_all_embs = torch.logical_and(got_all_embs, got_emb)
    
    for year_range in [first_w2v_years, second_w2v_years, test_w2v_years]:
        embs_dict[year_range] = embs_dict[year_range][got_all_embs]
    
    before = len(CUI_names)
    CUI_names = [name for i, name in enumerate(CUI_names) if got_all_embs[i]]
    print(f"Keeping {len(CUI_names)}/{before} CUIs.")

    pairs = list(product(CUI_names, CUI_names))
    similarity_df = pd.DataFrame()
    similarity_df[['CUI1', 'CUI2']] = pairs
    for year_range in [first_w2v_years, second_w2v_years, test_w2v_years]:
        similarity = cosine_similarity(embs_dict[year_range], embs_dict[year_range]).flatten()
        similarity_df[f'sim_{year_range}'] = similarity
    similarity_df['similarity'] = similarity_df[f'sim_{test_w2v_years}']
    # filter duplicate pairs - (A,B) is the same as (B,A). Keep the pair where CUI1 name <CUI2 name, lexicographically.
    before = len(similarity_df)
    similarity_df = similarity_df[similarity_df['CUI1'] < similarity_df['CUI2']]
    print(f"Removed duplicate symmetric pairs, kept {len(similarity_df)}/{before} pairs.") 
    # Now filter according to the diff in similarity.
    similarity_df['sim_diff'] = similarity_df[f'sim_{second_w2v_years}'] - similarity_df[f'sim_{first_w2v_years}']
    num_samples = int(sample_fraction*len(similarity_df))
    print(f"Keeping {num_samples} out of {len(similarity_df)} pairs")
    similarity_df = similarity_df.nlargest(num_samples, 'sim_diff')
    if write_to_file is not None:
        similarity_df.to_csv(write_to_file)
    else:
        return similarity_df


def calculate_sims_and_compare(cui_similarity_file, w2v_years, read_w2v_params={}, save_to_file=None):
    sim_df = pd.read_csv(cui_similarity_file, index_col=0)  # CUI1, CUI2, similarity
    tokenizer = TextUtils()
    w2v_model = read_w2v_model(w2v_years[0], w2v_years[1], **read_w2v_params)
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
    read_w2v_params = {'abstract_weighting_mode': 'normal', 'pubmed_version': 2019, 'only_aact_data': True}
    #find_interesting_CUI_pairs((2010, 2013), (2018, 2018), (2019, 2019),
    #                           semtypes=['dsyn'], top_percentile=0.5, filter_by_models=[], 
    #                           sample_fraction=0.01,
    #                           write_to_file=os.path.join(SAVE_PATH, 'test_interesting_CUI_pairs_aact.csv'),
    #                           read_w2v_params=read_w2v_params)
    #sys.exit()

    #test_pairs_path = os.path.join(SAVE_PATH, 'test_similarities_CUI_names_2019_2019.csv')
    test_pairs_path = os.path.join(SAVE_PATH, 'test_interesting_CUI_pairs_aact.csv')
    if not os.path.exists(test_pairs_path):
        print("generating test pairs")
        cui_dataset = pmd.CUIDataset(bert=None, w2v_years=(2019, 2019),
                                     read_w2v_params=read_w2v_params,
                                     top_percentile=0.5,
                                     semtypes=['dsyn'], frac=0.001, sample_type=1, 
                                     read_from_file=None)
    else:
        print(f"Using test pairs file: {test_pairs_path}")
    
    print("W2V years: 2018")
    calculate_sims_and_compare(test_pairs_path,
                               (2018, 2018),
                               read_w2v_params,
                               save_to_file=None)

    print("W2V years: 2010-2013")
    calculate_sims_and_compare(test_pairs_path,
                               (2010, 2013),
                               read_w2v_params,
                               save_to_file=None)

    print("W2V years: 2010-2018")
    calculate_sims_and_compare(test_pairs_path,
                               (2010, 2018),
                               read_w2v_params,
                               save_to_file=None)
