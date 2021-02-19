import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from contra.datasets import PubMedModule
from datetime import datetime
import pytz
from contra import config


class LogReg(pl.LightningModule):
    # Logistic Regression as a pytorch module
    # TODO: support wandb
    def __init__(self, dim):
        # dim = vocab size
        super(LogReg, self).__init__()
        self.linear = nn.Linear(dim, 1)
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

# Training without pytorch lightning
# criterion = torch.nn.BCELoss(size_average=True)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# for epoch in range(20):
#     model.train()
#     optimizer.zero_grad()
#     # Forward pass
#     y_pred = model(x_data)
#     # Compute Loss
#     loss = criterion(y_pred, y_data)
#     # Backward pass
#     loss.backward()
#     optimizer.step()


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
    dm = PubMedModule(hparams)
    model = LogReg(hparams.embedding_size)
    logger = WandbLogger(name=hparams.name, save_dir=hparams.log_path,
                         version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                         project='Experimental', config=hparams)
    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.max_epochs,
                         logger=logger,
                         log_every_n_steps=10,
                         accumulate_grad_batches=1)
    trainer.fit(model, datamodule=dm)
