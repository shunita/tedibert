import sys

from sklearn.model_selection import train_test_split

sys.path.append('/home/shunita/fairemb/')

import os
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import pytz
import torch
from nltk import sent_tokenize
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from contra.constants import DATA_PATH, SAVE_PATH


def read_abstracts():
    df = pd.read_csv(
        os.path.join(DATA_PATH, 'abstracts_and_population_tokenized_for_cui2vec_copyrightfix_sent_sep.csv'),
        index_col=0)
    df['all_participants'] = df['male'] + df['female']
    df['fem_prop'] = df['female'] / df['all_participants']
    df = df.dropna(subset=['abstract'])

    def process_row(row):
        title = row['title']
        if type(title) == str:
            title = title + " "
        else:
            title = ""
        abstract = row['abstract'].replace(';', ' ')
        return title + abstract

    df['tokenized'] = df.apply(process_row, axis=1)
    return df


class PubMedRatioModule(pl.LightningDataModule):
    def __init__(self, batch_size, randomize=False):
        self.batch_size = batch_size
        self.randomize = randomize

    def prepare_data(self, *args, **kwargs):
        df = pd.read_csv(os.path.join(DATA_PATH, 'pubmed2020_assigned.csv'))
        # df = read_abstracts()
        df['num_participants'] = df['male'] + df['female']
        df['fem_prop'] = df['female'] / df['num_participants']
        if self.randomize:
            fem_prop_values = df['fem_prop'].values
            np.random.shuffle(fem_prop_values)
            df['fem_prop'] = fem_prop_values
        # self.train_df, self.val_df = train_test_split(df, test_size=0.3)
        self.train_df = df[df.assignment == 0].copy()
        self.val_df = df[df.assignment == 1].copy()

    def setup(self, stage=None):
        self.train = PubmedRatioDataset(self.train_df)
        self.val = PubmedRatioDataset(self.val_df)

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=8)


class PubmedRatioDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # TODO: return a list of sentences and not a full abstract (because of truncation)
        return {'text': row['abstract'], 'label': row['fem_prop'], 'year': row['year']}


class FemaleRatioRegression(pl.LightningModule):
    def __init__(self, tokenizer_name, model_checkpoint, learning_rate, mode='mean', with_year=False):
        super(FemaleRatioRegression, self).__init__()
        self.bert_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.bert_model = AutoModel.from_pretrained(model_checkpoint)
        self.mode = mode
        self.learning_rate = learning_rate
        # move the model to evaluation mode
        self.bert_model.eval()
        self.sentence_embedding_size = self.bert_model.get_input_embeddings().embedding_dim
        self.max_len = 70
        self.max_sentences = 20
        classifier_size = self.sentence_embedding_size
        if self.mode == 'concat':
            classifier_size = self.max_sentences * self.sentence_embedding_size
        self.with_year = with_year
        if self.with_year:
            classifier_size += 1
        self.classifier = nn.Linear(classifier_size, 1)
        self.loss_func = torch.nn.MSELoss()

    def forward(self, batch):
        indexes = []
        all_sentences = []
        index = 0
        for sample in batch['text']:
            sample_as_list = sent_tokenize(sample)
            # sample is a list of sentences
            indexes.append((index, index + len(sample_as_list)))
            index += len(sample_as_list)
            all_sentences.extend(sample_as_list)

        inputs = self.bert_tokenizer.batch_encode_plus(all_sentences, padding=True, truncation=True,
                                                       max_length=self.max_len, add_special_tokens=True,
                                                       return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.bert_model(**inputs, output_hidden_states=False)
        sent_embedding = outputs.pooler_output  # shape [num_sentences, emb_size]
        sample_embedding = []
        for start, end in indexes:
            if self.mode == 'mean':
                sample_embedding.append(torch.mean(sent_embedding[start:end], dim=0))
            elif self.mode == 'concat':
                cur_sent_embedding = sent_embedding[start:end]
                if len(cur_sent_embedding) > self.max_sentences:  # Too many sentences
                    cur_sent_embedding = cur_sent_embedding[:self.max_sentences]
                    sample_embedding.append(torch.flatten(cur_sent_embedding))
                else:  # Too few sentences - add padding
                    padding = torch.zeros(self.max_sentences - len(cur_sent_embedding), self.sentence_embedding_size,
                                          device=self.device)
                    sample_embedding.append(torch.flatten(torch.cat([cur_sent_embedding, padding], dim=0)))
        sample_embedding = torch.stack(sample_embedding)  # shape [batch_size, emb_size] or [batch_size, emb_size*max_sentences]
        if self.with_year:
            # print(f"shape of sample_emb: {sample_embedding.shape}")
            sample_embedding = torch.cat((sample_embedding, torch.unsqueeze(batch['year'], dim=1)), dim=1)
            # print(f"shape of sample_emb after concatenating year: {sample_embedding.shape}")
        y_pred = torch.sigmoid(self.classifier(sample_embedding))  # shape [batch_size, 1]
        return y_pred

    def step(self, batch, name):
        ypred = self.forward(batch).squeeze(1)  # size: [16]
        ytrue = batch['label'].float()  # size: [16]
        loss = self.loss_func(ypred, ytrue)
        self.log(f'{name}_loss', loss)
        return loss

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        return self.step(batch, 'train')

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        return self.step(batch, 'val')

    # def validation_epoch_end(self, outputs) -> None:
    #     losses = np.concatenate([batch['loss'].cpu().numpy() for batch in outputs])
    #     self.log(f'final_val_loss', np.mean(losses))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.learning_rate,
            # weight_decay=1e-4
        )
        return {'optimizer': optimizer}


if __name__ == '__main__':
    BATCH_SIZE = 128
    MODEL_CHECKPOINT = 'google/bert_uncased_L-2_H-128_A-2'
    MAX_EPOCHS = 15
    LR = 1e-5
    RANDOMIZE = True
    MODE = 'concat'
    WITH_YEAR = True
    dm = PubMedRatioModule(batch_size=BATCH_SIZE, randomize=RANDOMIZE)
    model = FemaleRatioRegression(tokenizer_name='google/bert_uncased_L-2_H-128_A-2',
                                  model_checkpoint=MODEL_CHECKPOINT,
                                  learning_rate=LR,
                                  mode=MODE,
                                  with_year=WITH_YEAR)
    logger = WandbLogger(name="fem_ratio_regression_w.year", save_dir=SAVE_PATH,
                         version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                         project='Experimental', config={'batch_size': BATCH_SIZE,
                                                         'model': MODEL_CHECKPOINT,
                                                         'step size': LR,
                                                         'randomize_labels': RANDOMIZE,
                                                         'sent_agg_mode': MODE,
                                                         'with_year': WITH_YEAR})
    # lr_logger = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(gpus=1,
                         max_epochs=MAX_EPOCHS,
                         logger=logger,
                         log_every_n_steps=20,
                         accumulate_grad_batches=1,
                         )
    trainer.fit(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)