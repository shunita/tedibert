import sys
sys.path.append('/home/shunita/fairemb/')
from transformers import AutoModel, AutoTokenizer
from contra.constants import SAVE_PATH
import numpy as np
from scipy.spatial import procrustes
import os
import torch
from torch import nn


old_model = AutoModel.from_pretrained(os.path.join(SAVE_PATH, 'bert_tiny_uncased_2010_2013_v2020_epoch9'))
new_model = AutoModel.from_pretrained(os.path.join(SAVE_PATH, 'bert_tiny_uncased_2016_2018_v2020_epoch9'))
old_inp_emb = old_model.get_input_embeddings().weight.detach().numpy()
new_inp_emb = new_model.get_input_embeddings().weight.detach().numpy()
# Align the old to the new
mtx1, mtx2, disparity = procrustes(new_inp_emb, old_inp_emb)
aligned_emb = nn.Embedding.from_pretrained(torch.FloatTensor(mtx2), padding_idx=0)
new_model.set_input_embeddings(aligned_emb)
new_model.save_pretrained(os.path.join(SAVE_PATH, 'bert_tiny_uncased_10_13_aligned_16_18'))



