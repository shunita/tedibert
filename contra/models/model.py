import os
from scipy import stats
import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from contra.constants import SAVE_PATH


class FairEmbedding(pl.LightningModule):
    def __init__(self, hparams):
        super(FairEmbedding, self).__init__()
        self.hparams = hparams
        self.bn = hparams.bn
        self.activation = hparams.activation
        self.do_ratio_prediction = (hparams.lmb_ratio > 0)

        self.initial_embedding_size = hparams.initial_emb_size
        self.embedding_size = hparams.embedding_size

        if self.do_ratio_prediction:
            self.ratio_reconstruction = Classifier(self.embedding_size, int(self.embedding_size / 2), 1, hid_layers=0)
            self.discriminator = Classifier(self.embedding_size + 1, int(self.embedding_size / 2), 1, hid_layers=2,
                                            bn=self.bn, activation=self.activation)
        else:
            self.discriminator = Classifier(self.embedding_size, int(self.embedding_size / 2), 1, hid_layers=2,
                                            bn=self.bn, activation=self.activation)
        self.L1Loss = torch.nn.L1Loss()
        self.BCELoss = torch.nn.BCELoss()

    def step(self, batch: dict, optimizer_idx: int = None, name='train') -> dict:
        if optimizer_idx == 0:
            self.ratio_true = batch['female_ratio']
            self.is_new = batch['is_new']
            # generate
            self.fair_embedding, g_loss = self.forward(batch)
            
            if self.do_ratio_prediction:
                self.ratio_pred = self.ratio_reconstruction(self.fair_embedding)
                ratio_loss = self.BCELoss(self.ratio_pred.squeeze()[self.is_new].float(), self.ratio_true[self.is_new].float())
                isnew_pred = self.discriminator(torch.cat((self.fair_embedding, self.ratio_pred), dim=1))
            else:
                isnew_pred = self.discriminator(self.fair_embedding)
            only_old_predicted = isnew_pred.squeeze()[~self.is_new]
            # we take old samples and label them as new, to train the generator.
            # Discriminator weights do not change while we train the generator because of the different optimizer_idx's.
            isnew_loss = self.BCELoss(only_old_predicted, torch.ones(only_old_predicted.shape[0], device=self.device))

            # final loss
            loss = g_loss + self.hparams.lmb_isnew * isnew_loss
            if self.do_ratio_prediction:
                loss += self.hparams.lmb_ratio * ratio_loss
                self.log(f'generator/{name}_ratio_loss', ratio_loss)
            self.log(f'generator/{name}_reconstruction_loss', g_loss)
            self.log(f'generator/{name}_discriminator_loss', isnew_loss)
            self.log(f'generator/{name}_loss', loss)

        if optimizer_idx == 1:
            # discriminate
            emb = self.fair_embedding.detach()
            
            if self.do_ratio_prediction:
                isnew_pred = self.discriminator(torch.cat([emb, self.ratio_pred.detach()], dim=1))
            else:
                isnew_pred = self.discriminator(emb)
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
            loss = self.step(batch, i, name='val')

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        CUI1_embedding, _ = self.forward(self.wrap_text_to_batch(batch['CUI1']))
        CUI2_embedding, _ = self.forward(self.wrap_text_to_batch(batch['CUI2']))
        pred_similarity = nn.CosineSimilarity()(CUI1_embedding, CUI2_embedding)
        true_similarity = batch['true_similarity']
        return pred_similarity, true_similarity
        
    def wrap_text_to_batch(self, texts):
        batch1 = {'text': texts, 
                  'is_new': torch.zeros(len(texts), dtype=bool, device=self.device)
                  }
        return batch1

    def test_epoch_end(self, outputs) -> None:
        rows = torch.cat([torch.stack(output) for output in outputs], axis=1).T.cpu().numpy()
        df = pd.DataFrame(rows, columns=['pred_similarity', 'true_similarity'])
        df.to_csv(os.path.join(SAVE_PATH, 'test_similarities.csv'))
        df = df.sort_values(['true_similarity'], ascending=False).reset_index()
        true_rank = list(df.index)
        pred_rank = list(df.sort_values(['pred_similarity'], ascending=False).index)
        correlation, pvalue = stats.spearmanr(true_rank, pred_rank)
        self.log('test/correlation', correlation)
        self.log('test/pvalue', pvalue)



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
    def __init__(self, in_dim, hid_dim, out_dim, hid_layers=0, bn=False, activation='relu'):
        super(Classifier, self).__init__()
        if activation == 'relu':
            act_func = nn.ReLU(inplace=True)
        elif activation == 'swish':
            act_func = torch.nn.SiLU()
        else:
            print(f"Unsupported option for activation function: {activation}")

        layers = [nn.Linear(in_dim, hid_dim),
                  act_func]

        for i in range(hid_layers):
            layers.append(nn.Linear(hid_dim, hid_dim))
            layers.append(act_func)
            if bn:
                layers.append(nn.BatchNorm1d(hid_dim))  

        layers.append(nn.Linear(hid_dim, out_dim))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor):
        return self.model(input)
