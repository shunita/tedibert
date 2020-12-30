import pandas as pd
from itertools import chain
import sys
import os
import numpy as np
from datetime import datetime
import pytorch_lightning as pl
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from scipy import stats
from contra.models.w2v_on_years import PretrainedW2V, PretrainedOldNewW2V
from contra.utils.text_utils import TextUtils
from contra.common.utils import mean_pooling
from contra.constants import SAVE_PATH, DATA_PATH

class FairEmbedding(pl.LightningModule):
    def __init__(self, hparams):
        super(FairEmbedding, self).__init__()
        self.hparams = hparams

        self.initial_emb_algorithm = hparams.emb_algorithm
        self.initial_embedding_size = hparams.initial_emb_size
        self.embedding_size = hparams.embedding_size

        if self.initial_emb_algorithm == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
            self.bert_model = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
            self.initial_embedding_size = self.bert_model.get_input_embeddings().embedding_dim

        elif self.initial_emb_algorithm == 'w2v':
            self.tokenizer = TextUtils()
            old_w2v = self.read_w2v_model(hparams.first_start_year, hparams.first_end_year)
            new_w2v = self.read_w2v_model(hparams.second_start_year, hparams.second_end_year)
            self.w2v = PretrainedOldNewW2V(old_w2v, new_w2v)
        else:
            print("unsupported initial embedding algorithm. Should be 'bert' or 'w2v'.")
            sys.exit()

        self.autoencoder = Autoencoder(in_dim=self.initial_embedding_size, hid_dim=self.embedding_size)
        self.ratio_reconstuction = Classifier(self.embedding_size, int(self.embedding_size / 2), 1)
        self.discriminator = Classifier(self.embedding_size + 1, int(self.embedding_size / 2), 1)
        self.L1Loss = torch.nn.L1Loss()
        self.BCELoss = torch.nn.BCELoss()

    def read_w2v_model(self, year1, year2):
        '''Read the pretrained embeddings of a year range.'''
        year_to_ndocs = pd.read_csv(os.path.join(DATA_PATH, 'year_to_ndocs.csv'), index_col=0,
                                    dtype={'year': int, 'ndocs': int}).to_dict(orient='dict')['ndocs']
        vectors_path = os.path.join(SAVE_PATH, f"word2vec_{year1}_{year2}.wordvectors")
        idf_path = os.path.join(SAVE_PATH, f'idf_dict{year1}_{year2}.pickle')
        num_docs_in_range = sum([year_to_ndocs[year] for year in range(year1, year2+1)])
        w2v = PretrainedW2V(idf_path, vectors_path, ndocs=docs)
        return w2v

    def forward(self, batch):
        text = batch['text']
        if self.initial_emb_algorithm == 'bert':
            # TODO: this will use the same bert model for all abstracts, regardless of year (it shouldn't).
            inputs = self.tokenizer.batch_encode_plus(text, padding=True, truncation=True, max_length=50,
                                                      add_special_tokens=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.bert_model(**inputs)
            sentence_embedding = mean_pooling(outputs, inputs['attention_mask'])
        elif self.initial_emb_algorithm == 'w2v':
            # tokenization is the same for old and new (simple word_tokenize)
            tokenized_texts = [self.tokenizer.word_tokenize_abstract(t) for t in text]
            is_new = batch['is_new']
            sentence_embedding = self.w2v.embed_batch(tokenized_texts, is_new, self.device)

        decoded, latent = self.autoencoder(sentence_embedding)

        return sentence_embedding, latent, decoded

    def step(self, batch: dict, optimizer_idx: int = None, name='train') -> dict:
        if optimizer_idx == 0:
            self.ratio_true = batch['female_ratio']
            self.is_new = batch['is_new']
            # generate
            initial_embedding, self.fair_embedding, reconstruction = self.forward(batch)
            g_loss = self.L1Loss(initial_embedding, reconstruction)
            self.ratio_pred = self.ratio_reconstuction(self.fair_embedding)
            ratio_loss = self.BCELoss(self.ratio_pred.squeeze()[self.is_new].float(), self.ratio_true[self.is_new].float())

            # discriminate
            isnew_pred = self.discriminator(torch.cat([self.fair_embedding, self.ratio_pred], dim=1))
            only_old = isnew_pred.squeeze()[~self.is_new]
            # TODO: why do we take old rows and a label as if they were new??
            isnew_loss = self.BCELoss(only_old, torch.ones(only_old.shape[0], device=self.device))

            # final loss
            loss = g_loss + self.hparams.lmb_ratio * ratio_loss + self.hparams.lmb_isnew * isnew_loss
            self.log(f'generator/{name}_reconstruction_loss', g_loss)
            self.log(f'generator/{name}_ratio_loss', ratio_loss)
            self.log(f'generator/{name}_discriminator_loss', isnew_loss)
            self.log(f'generator/{name}_loss', loss)

        if optimizer_idx == 1:
            # discriminate
            isnew_pred = self.discriminator(torch.cat([self.fair_embedding.detach(), self.ratio_pred.detach()], dim=1))
            isnew_loss = self.BCELoss(isnew_pred.squeeze(), self.is_new.float())

            # final loss
            loss = isnew_loss
            self.log(f'discriminator/{name}_loss', loss)
        return loss

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        loss = self.step(batch, optimizer_idx, name='train')
        return {'loss': loss}

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        for i in range(len(self.optimizers())):
            self.step(batch, i, name='val')

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        _, CUI1_embedding, _ = self.forward(batch['CUI1'])
        _, CUI2_embedding, _ = self.forward(batch['CUI2'])
        pred_similarity = nn.CosineSimilarity()(CUI1_embedding, CUI2_embedding)
        true_similarity = batch['true_similarity']

        return pred_similarity, true_similarity

    def test_epoch_end(self, outputs) -> None:
        rows = torch.cat([torch.stack(output) for output in outputs], axis=1).T.cpu().numpy()
        df = pd.DataFrame(rows, columns=['pred_similarity', 'true_similarity'])
        df = df.sort_values(['true_similarity'], ascending=False).reset_index()
        true_rank = list(df.index)
        pred_rank = list(df.sort_values(['pred_similarity'], ascending=False).index)

        correlation, pvalue = stats.spearmanr(true_rank, pred_rank)
        self.log('test/correlation', correlation)
        self.log('test/pvalue', pvalue)


    def configure_optimizers(self):
        optimizer_1 = torch.optim.Adam(chain(self.autoencoder.parameters(), self.ratio_reconstuction.parameters()),
                                       lr=self.hparams.learning_rate)
        optimizer_2 = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.learning_rate)
        return [optimizer_1, optimizer_2]


class Autoencoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim=None):
        super(Autoencoder, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(True),
            nn.Linear(hid_dim, hid_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(True),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, input: torch.Tensor):
        z = self.encoder(input)
        output = self.decoder(z)
        return output, z


class Classifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(True),
            nn.Linear(hid_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor):
        return self.model(input)
