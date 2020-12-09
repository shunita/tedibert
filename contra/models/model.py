import pandas as pd
from itertools import chain
import pytorch_lightning as pl
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from scipy import stats

from contra.common.utils import mean_pooling


class FairEmbedding(pl.LightningModule):
    def __init__(self, hparams):
        super(FairEmbedding, self).__init__()
        self.hparams = hparams

        self.embedding_size = hparams.embedding_size

        self.tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        self.bert_model = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        self.bert_embedding_size = self.bert_model.get_input_embeddings().embedding_dim

        self.autoencoder = Autoencoder(self.bert_embedding_size, self.embedding_size)
        self.ratio_reconstuction = Classifier(self.embedding_size, int(self.embedding_size / 2), 1)
        self.discriminator = Classifier(self.embedding_size + 1, int(self.embedding_size / 2), 1)
        self.L1Loss = torch.nn.L1Loss()
        self.BCELoss = torch.nn.BCELoss()

    def forward(self, text):
        inputs = self.tokenizer.batch_encode_plus(text, padding=True, truncation=True, max_length=50,
                                                  add_special_tokens=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.bert_model(**inputs)
        bert_sentence_embedding = mean_pooling(outputs, inputs['attention_mask'])

        decoded, latent = self.autoencoder(bert_sentence_embedding)

        return bert_sentence_embedding, latent, decoded

    def step(self, batch: dict, optimizer_idx: int = None, name='train') -> dict:
        if optimizer_idx == 0:
            self.ratio_true = batch['female_ratio']
            self.is_new = batch['is_new']
            # generate
            bert_embedding, self.fair_embedding, bert_reconstruction = self.forward(batch['text'])
            g_loss = self.L1Loss(bert_embedding, bert_reconstruction)

            self.ratio_pred = self.ratio_reconstuction(self.fair_embedding)
            ratio_loss = self.BCELoss(self.ratio_pred.squeeze()[self.is_new].float(), self.ratio_true[self.is_new].float())

            # discriminate
            isnew_pred = self.discriminator(torch.cat([self.fair_embedding, self.ratio_pred], dim=1))
            isnew_loss = self.BCELoss(isnew_pred.squeeze()[~self.is_new], torch.ones(isnew_pred.squeeze()[~self.is_new].shape[0], device=self.device))

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
        return loss

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
