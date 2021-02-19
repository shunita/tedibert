import sys
sys.path.append('/home/shunita/fairemb/')

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import pytz
from contra import config
from contra.datasets.pubmed_bow_dataset import PubMedBOWModule


class LogReg(pl.LightningModule):
    # Logistic Regression as a pytorch module
    # TODO: support wandb
    def __init__(self, hparams):
        # dim = vocab size
        super(LogReg, self).__init__()
        self.hparams = hparams
        self.linear = nn.Linear(hparams.embedding_size, 1)
        self.BCE_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

    def step(self, batch, name):
        ypred = self.forward(batch['text'])
        loss = self.BCE_with_logits(ypred, batch['is_new'])
        self.log(f'{name}_loss', loss)
        return loss

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        return self.step(batch, 'train')

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        return self.step(batch, 'val')
        
    def configure_optimizers(self):
        optimizer_1 = torch.optim.Adam(self.linear.parameters(), lr=self.hparams.learning_rate)
        return [optimizer_1]



# dataset - or can we use the pubmed_dataset?

# need to add the sentence cleaning to the pubmed_dataset anyway - done - V

hparams = config.parser.parse_args(['--name', 'LogReg exp',
                                    '--first_start_year', '2011',
                                    '--first_end_year', '2013',
                                    '--second_start_year', '2016',
                                    '--second_end_year', '2018',
                                    '--emb_algorithm', 'bert', #'w2v',
                                    '--dim', '128',  # tinybert
                                    '--bert_pretrained_path', 'google/bert_uncased_L-2_H-128_A-2',
                                    '--bert_tokenizer', 'google/bert_uncased_L-2_H-128_A-2',
                                    '--batch_size', '256',
                                    '--lr', '1e-5',
                                    '--max_epochs', '10',
                                    '--lmb_ratio', '0',
                                    '--test_size', '0.3',
                                    '--abstract_weighting_mode', 'normal',
                                    '--pubmed_version', '2020',
                                    '--only_aact_data',
                                    '--test_start_year', '2020',
                                    '--test_end_year', '2020',
                                    '--test_pairs_file', 'test_similarities_CUI_names_tinybert_2020_2020.csv'
                                    ])
hparams.gpus = 0
if __name__ == '__main__':
    dm = PubMedBOWModule(hparams)
    model = LogReg(hparams)
    logger = WandbLogger(name=hparams.name, save_dir=hparams.log_path,
                         version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                         project='Experimental', config=hparams)
    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.max_epochs,
                         logger=logger,
                         log_every_n_steps=10,
                         accumulate_grad_batches=1)
    trainer.fit(model, datamodule=dm)
