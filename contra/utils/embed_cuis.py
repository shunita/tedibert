import os
import sys
sys.path.append('/home/shunita/fairemb/')
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from contra.constants import DATA_PATH, SAVE_PATH


def embed_by_bert(text_batch, tokenizer, bert_model):
    inputs = tokenizer.batch_encode_plus(text_batch, padding=True, truncation=True, max_length=30,
                                         add_special_tokens=True, return_tensors="pt")
    outputs = bert_model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0]
    return outputs.detach().numpy()


def embed_file(df, tokenizer, bert_model):
    str_embs = []
    for i in range(0, len(df), 32):
        out = embed_by_bert(df.iloc[i:i+32]['name'].values.tolist(), tokenizer, bert_model)
        for emb in out:
            str_embs.append(",".join([str(x) for x in emb]))
    df['emb'] = str_embs
    return df


bert_models = [
    #('google/bert_uncased_L-2_H-128_A-2', 'tinybert'),
    #(os.path.join(SAVE_PATH, 'bert_tiny_uncased_2010_2018_v2020_epoch39'), 'bert10-18'),
    #(os.path.join(SAVE_PATH, 'bert_GAN_new0.4_ref0.2'), 'GAN_new0.4_ref0.2')
    # (os.path.join(SAVE_PATH, 'bert_GAN_new0.4_ref0.2_0.1_refbert10eps_epoch9'), 'GAN_new0.4_ref0.2_0.1_refbert10eps'),
    # (os.path.join(SAVE_PATH, 'bert_tiny_uncased_2010_2018_v2020_epoch9'), 'bert2010_18_10eps'),
    # (os.path.join(SAVE_PATH, 'bert_tiny_uncased_2010_2013_v2020_epoch39'), 'bert2010_13_40eps'),
    # (os.path.join(SAVE_PATH, 'bert_tiny_uncased_2016_2018_v2020_epoch39'), 'bert2016_18_40eps'),
    # (os.path.join(SAVE_PATH, 'bert_tiny_uncased_2020_2020_v2020_epoch39'), 'bert2020_40eps'),
    (os.path.join(SAVE_PATH, 'bert_GAN_new0.3_ref0.1_0.3_epoch9'), 'GAN_new0.3_ref0.3_0.3_10eps'),

    # (os.path.join(SAVE_PATH, 'bert_GAN_new0.4_ref0.2_0.1_epoch9'), 'GAN_new0.4_ref0.2_0.1_10eps'),
    ]
tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
cui_and_name = pd.read_csv(os.path.join(DATA_PATH, 'cui_and_name_for_com_class.csv'), index_col=0)
for model_path, name in bert_models:
    bert_model = AutoModel.from_pretrained(model_path)
    cui_and_name = embed_file(cui_and_name, tokenizer, bert_model)
    cui_and_name[['cui', 'emb']].to_csv(os.path.join(SAVE_PATH, f'cui_{name}_emb.tsv'),
                                        sep='\t', header=False, index=False)
    cui_and_name = cui_and_name.drop(columns=['emb'])
