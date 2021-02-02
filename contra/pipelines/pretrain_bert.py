import sys
sys.path.append('/home/shunita/fairemb/')

from datetime import datetime
import pytz
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.multiprocessing import freeze_support

from contra.datasets import PubMedFullModule
from contra import config
from contra.models import BertPretrainOnYears

# For debug purposes: disable randomness
# import numpy as np
# import torch
# np.random.seed(1)
# torch.manual_seed(1)

hparams = config.parser.parse_args(['--name', 'BertYears11-13', 
                                    '--start_year', '2011',
                                    '--end_year', '2013',
                                    '--by_sentence',
                                    '--max_epochs', '40',
                                    '--lr', '5e-5',
                                    '--abstract_weighting_mode', 'normal',  # subsample
                                    '--pubmed_version', '2020',
                                    #'--num_frozen_layers', '10',
                                    '--only_aact_data',
                                    '--bert_pretrained_path', 'google/bert_uncased_L-2_H-128_A-2',
                                    '--bert_save_prefix', 'bert_tiny_uncased',
                                    ])
# When running with two gpus:
# hparams.gpus = [0,1]
# When running with one gpu:
hparams.gpus = 1

dm = PubMedFullModule(hparams)
model = BertPretrainOnYears(hparams)

logger = WandbLogger(name=hparams.name, save_dir=hparams.log_path,
                     version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                     project='Contra', config=hparams)
trainer = pl.Trainer(gpus=hparams.gpus, 
                     auto_select_gpus=True, 
                     max_epochs=hparams.max_epochs, 
                     logger=logger, 
                     log_every_n_steps=10, 
                     accumulate_grad_batches=1,  # no accumulation
                     precision=16, 
                     distributed_backend='ddp')

trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    freeze_support()
