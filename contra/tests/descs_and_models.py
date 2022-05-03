import os
from contra.constants import SAVE_PATH, DATA_PATH

DESCS_AND_MODELS = [('GAN10', os.path.join(SAVE_PATH, 'bert_GAN_new0.3_ref0.1_0.3_epoch9')),  # 0
                    ('BERT10-18_40eps', os.path.join(SAVE_PATH, 'bert_tiny_uncased_2010_2018_v2020_epoch39')),  # 1
                    ('BERT10-13+16-18_40eps', os.path.join(SAVE_PATH, 'bert_tiny10-13+16-18_epoch39')),  # 2
                    ('tinybert_non_medical', 'google/bert_uncased_L-2_H-128_A-2'),  # 3
                    ('GAN20_ref_bert10-13+16-18', os.path.join(SAVE_PATH, 'bert_GAN_ref_bert10-13+16-18_epoch19')),  #4
                    ('GAN20', os.path.join(SAVE_PATH, 'bert_GAN_new0.3_ref0.1_0.3_concat_epoch19')),  #5
                    ('BERT2020_40eps', os.path.join(SAVE_PATH, 'bert_tiny_uncased_2020_2020_v2020_epoch39')),  #6

                    ('female40', os.path.join(DATA_PATH, 'embs', 'fem40_heur_emb.tsv')),  # 7
                    ('neutral40', os.path.join(DATA_PATH, 'embs', 'neutral40_emb.tsv')),  # 8
                    ('randomup40', os.path.join(DATA_PATH, 'embs', 'random_upsample40_emb.tsv')),  # 9
                    ('all40', os.path.join(DATA_PATH, 'embs', 'all40_heur_emb.tsv')),  # 10
                    ('female40_triple_log', os.path.join(DATA_PATH, 'embs', 'fem40_triple_log_emb.tsv')),  # 11
                    ('female100', os.path.join(DATA_PATH, 'embs', 'female_cui2vec_style_w2v_copyrightfix_100_emb.tsv')),  # 12
                    ('neutral100', os.path.join(DATA_PATH, 'embs', 'neutral_cui2vec_style_w2v_copyrightfix_100_emb.tsv')),  # 13
                    ('female300', os.path.join(DATA_PATH, 'embs', 'female_cui2vec_style_w2v_copyrightfix_300_emb.tsv')),  # 14
                    ('neutral300', os.path.join(DATA_PATH, 'embs', 'neutral_cui2vec_style_w2v_copyrightfix_300_emb.tsv')),  # 15

                    ('GAN20_1side', os.path.join(SAVE_PATH, 'bert_GAN_new0.3_ref0.3_concat_1sideloss_epoch19')),  # 16
                    ('medical_bert_specialized20', os.path.join(SAVE_PATH, 'bert_medical_bert_specialized20_epoch19')),  # 17
                    ('BERT16-18', os.path.join(SAVE_PATH, 'bert_tiny_uncased_2016_2018_v2020_epoch39')),  # 18
                    ('BERT18_20eps', os.path.join(SAVE_PATH, 'bert_tiny_uncased_2018_2018_v2020_epoch19')),  # 19

                    ('BERTbase10-18_20eps', os.path.join(SAVE_PATH, 'bert_base_uncased_2010_2018_v2020_epoch19_take1')),  # 20
                    ('bert-base-non-med', 'bert-base-uncased'),  # 21
                    ('GAN20base', os.path.join(SAVE_PATH, 'bert_DERT_bertbase_new0.3_ref0.1_0.3_concat_epoch19')),  # 22
                    ('GAN20base1', os.path.join(SAVE_PATH, 'bert_DERT_bertbase_new0.2_ref0.4_concat_anchor40_epoch19')),  # 23
                    ('BERTbase10-18_40eps', os.path.join(SAVE_PATH, 'bert_base_uncased_2010_2018_v2020_epoch39')),  # 24

                    ('GAN20_anchor16-18', os.path.join(SAVE_PATH, 'bert_DERT_bert_tiny_new0.3_ref0.3_concat_anchor16-18_epoch19')),  #25
                    ('GAN40_1side', os.path.join(SAVE_PATH, 'bert_DERT_tiny_1sideloss_new0.3_ref0.3_concat_40eps_epoch39')),  #26
                    ('GAN20_ref0.4_new0.2_1side', os.path.join(SAVE_PATH, 'bert_DERT_tiny_1sideloss_new0.2_ref0.4_concat_20eps_epoch19')),  #27

                    ('bert_L2_H256_2010-18_40eps', os.path.join(SAVE_PATH, 'bert_uncased_L2_H256_A4_2010_2018_v2020_epoch39')),  #28
                    ('bert_L2_H256_2010-18_20eps', os.path.join(SAVE_PATH, 'bert_uncased_L2_H256_A4_2010_2018_v2020_epoch19')),  #29
                    ('GAN20_1side_new2old', os.path.join(SAVE_PATH, 'bert_DERT_tiny_new2old_new0.3_ref0.1_0.3_concat_20eps_epoch19')),  #30
                    ('GAN20_1side_old2new', os.path.join(SAVE_PATH, 'bert_DERT_tiny_old2new_new0.3_ref0.1_0.3_concat_20eps_epoch19')),  #31
                    ('GAN20_old2new0.2', os.path.join(SAVE_PATH, 'bert_DERT_tiny_o2n0.2_new0.3_ref0.1_0.3_concat_20eps_epoch19')),  #32
                    ('GAN20_old2new0.8', os.path.join(SAVE_PATH, 'bert_DERT_tiny_o2n0.8_new0.3_ref0.1_0.3_concat_20eps_epoch19')),  #33

                    ]
cui_embeddings = [7, 8, 9, 10, 11, 12, 13, 14, 15]


# DESCS_AND_MODELS_LOS = [('GAN10', os.path.join(SAVE_PATH, 'bert_GAN_new0.3_ref0.1_0.3_epoch9')),  # 0
#                     ('BERT10-18_40eps', os.path.join(SAVE_PATH, 'bert_tiny_uncased_2010_2018_v2020_epoch39')),  # 1
#                     ('BERT10-13+16-18_40eps', os.path.join(SAVE_PATH, 'bert_tiny10-13+16-18_epoch39')),  # 2
#                     ('tinybert_non_medical', 'google/bert_uncased_L-2_H-128_A-2'),  # 3
#                     ('GAN20_ref_bert10-13+16-18', os.path.join(SAVE_PATH, 'bert_GAN_ref_bert10-13+16-18_epoch19')),  # 4
#                     ('GAN20', os.path.join(SAVE_PATH, 'bert_GAN_new0.3_ref0.1_0.3_concat_epoch19')),  # 5
#                     ('BERT10-13_40eps', os.path.join(SAVE_PATH, 'bert_tiny_uncased_2010_2013_v2020_epoch39')),  # 6
#                     ('BERT16-18_40eps', os.path.join(SAVE_PATH, 'bert_tiny_uncased_2016_2018_v2020_epoch39')),  # 7
#                     ('BERT2020_40eps', os.path.join(SAVE_PATH, 'bert_tiny_uncased_2020_2020_v2020_epoch39')),  # 8
#                     ('female40', os.path.join(DATA_PATH, 'embs', 'fem40_heur_emb.tsv')),  # 9
#                     ('neutral40', os.path.join(DATA_PATH, 'embs', 'neutral40_emb.tsv')),  # 10
#                     ('randomup40', os.path.join(DATA_PATH, 'embs', 'random_upsample40_emb.tsv')),  # 11
#                     ('all40', os.path.join(DATA_PATH, 'embs', 'all40_heur_emb.tsv')),  # 12
#
#                     ('GAN20_1side', os.path.join(SAVE_PATH, 'bert_GAN_new0.3_ref0.3_concat_1sideloss_epoch19')),  # 13
#                     ('medical_bert_specialized20', os.path.join(SAVE_PATH, 'bert_medical_bert_specialized20_epoch19')),  # 14
#                     ('BERT16-18', os.path.join(SAVE_PATH, 'bert_tiny_uncased_2016_2018_v2020_epoch39')),  # 15
#                     ('BERT18_20eps', os.path.join(SAVE_PATH, 'bert_tiny_uncased_2018_2018_v2020_epoch19')),  # 16
#                     ]