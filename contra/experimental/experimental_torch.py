import sys
from sklearn.metrics import roc_auc_score
sys.path.append('/home/shunita/fairemb/')

import os
import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import pytz
from contra import config
from contra.datasets.pubmed_bow_dataset import PubMedBOWModule
from contra.experimental.exp_utils import count_old_new_appearances
from contra.constants import EXP_PATH
import numpy as np


class LogReg(pl.LightningModule):
    # Logistic Regression as a pytorch module
    def __init__(self, vocab_size, hparams):
        super(LogReg, self).__init__()
        self.hparams = hparams
        self.linear = nn.Linear(vocab_size, 1)
        self.BCE_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, x):
        y_pred = self.linear(x.float())
        return y_pred

    def step(self, batch, name):
        ypred = self.forward(batch['text']).squeeze()
        ytrue = batch['is_new'].float()
        loss = self.BCE_with_logits(ypred, ytrue)
        self.log(f'{name}_loss', loss)
        if not all(ytrue) and any(ytrue):
            # Calc auc only if batch has more than one class.
            self.log(f'{name}_auc', roc_auc_score(ytrue, ypred.detach()))
        return loss

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        return self.step(batch, 'train')

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        return self.step(batch, 'val')
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.linear.parameters(), lr=self.hparams.learning_rate)
        return [optimizer]


def get_top_important_words(bow_vector, model_weights, index_to_word, n=5):
    sentence_weights = np.multiply(bow_vector, model_weights)
    top_indices = sentence_weights.argsort()[-n:][::-1]
    return ' '.join(['({}, {:.2f})'.format(index_to_word[i], sentence_weights[i]) for i in top_indices])

def train_abstract_model():
    hparams = config.parser.parse_args(['--name', 'abs helper',
                                        '--first_start_year', '2011',
                                        '--first_end_year', '2013',
                                        '--second_start_year', '2016',
                                        '--second_end_year', '2018',
                                        '--batch_size', '32',
                                        '--lr', '3e-4',
                                        '--max_epochs', '20',
                                        '--test_size', '0.3',
                                        '--abstract_weighting_mode', 'normal',
                                        '--pubmed_version', '2020',
                                        '--only_aact_data',
                                        ])
    hparams.gpus = 0
    dm = PubMedBOWModule(hparams)
    dm.prepare_data()
    dm.setup()
    model = LogReg(len(dm.index_to_word), hparams)
    logger = WandbLogger(name=hparams.name, save_dir=hparams.log_path,
                         version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                         project='Experimental', config=hparams)
    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.max_epochs,
                         logger=logger,
                         log_every_n_steps=10,
                         accumulate_grad_batches=1)
    trainer.fit(model, datamodule=dm)
    w = model.linear.weight.detach().numpy().squeeze()
    words_and_weights = list(zip(dm.index_to_word, w))
    word_to_weight = {word: weight for (word, weight) in words_and_weights}
    return word_to_weight


hparams = config.parser.parse_args(['--name', 'LogReg sent',
                                    '--first_start_year', '2011',
                                    '--first_end_year', '2013',
                                    '--second_start_year', '2016',
                                    '--second_end_year', '2018',
                                    '--batch_size', '32',
                                    '--lr', '1e-4',
                                    '--max_epochs', '20',
                                    '--test_size', '0.3',
                                    '--by_sentence',
                                    '--abstract_weighting_mode', 'normal',
                                    '--pubmed_version', '2020',
                                    '--only_aact_data',
                                    ])
hparams.gpus = 0
if __name__ == '__main__':
    dm = PubMedBOWModule(hparams)
    dm.prepare_data()
    dm.setup()
    model = LogReg(len(dm.index_to_word), hparams)
    logger = WandbLogger(name=hparams.name, save_dir=hparams.log_path,
                         version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                         project='Experimental', config=hparams)
    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.max_epochs,
                         logger=logger,
                         log_every_n_steps=10,
                         accumulate_grad_batches=1)
    trainer.fit(model, datamodule=dm)
    logger.experiment.finish()
    # Extract weights
    # words_and_weights_file = os.path.join(EXP_PATH, 'BOW_words_and_weights_old_new_by_abstract_after_filter_torch.csv')
    w = model.linear.weight.detach().numpy().squeeze()
    words_and_weights = list(zip(dm.index_to_word, w))
    # words_df = pd.DataFrame(words_and_weights, columns=['word', 'weight'])
    # word_to_appearances = count_old_new_appearances(df_with_old_new_label=None)
    # words_df['old_appearances'] = pd.Series([word_to_appearances[w][0] for w in dm.index_to_word])
    # words_df['new_appearances'] = pd.Series([word_to_appearances[w][1] for w in dm.index_to_word])
    # words_df.to_csv(words_and_weights_file)

    # analyse performance by sentence
    sentence_analysis_file = os.path.join(EXP_PATH, 'sentence_analysis_torch.csv')
    train = dm.train_df
    bow_train = dm.bow_train.toarray()
    ytrain_pred = model.forward(torch.tensor(bow_train)).detach().numpy().squeeze()
    print(f"len(train)={len(train)}, shape of pred: {ytrain_pred.shape}")
    train['model_prob'] = ytrain_pred
    # which words contributed most to the model decision?
    train['top_features'] = [get_top_important_words(bow_train[i], w, dm.index_to_word)
                             for i in range(len(bow_train))]

    # TODO: which part of the abstract did the sentence come from?
    # sum of absolute weights of the words according to the abstract model (suggests the importance of the sentence)
    word_to_abs_model_weight = train_abstract_model()
    train['abstract_model_importance'] = train['tokenized'].apply(
        lambda wordlist: np.sum([word_to_abs_model_weight[word]
                                 for word in wordlist if word in word_to_abs_model_weight]))
    train[['text', 'year', 'label', 'model_prob', 'top_features', 'abstract_model_importance']].to_csv(
        sentence_analysis_file)



