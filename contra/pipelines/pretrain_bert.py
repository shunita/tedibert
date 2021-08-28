import sys
sys.path.append('/home/shunita/fairemb/')

from datetime import datetime
import pytz
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.multiprocessing import freeze_support

# from contra.datasets import PubMedFullModule
from contra.datasets.pubmed_dataset import PubMedModule
from contra import config
from contra.models import BertPretrainOnYears

# For debug purposes: disable randomness
# import numpy as np
# import torch
# np.random.seed(1)
# torch.manual_seed(1)

hparams = config.parser.parse_args(['--name', 'BertYears2018',
                                    # TODO: originally used other params: start_year and end_year
                                    '--first_start_year', '2018',
                                    '--first_end_year', '2018',

                                    '--second_start_year', '2030',
                                    '--second_end_year', '1970',

                                    '--max_epochs', '40',
                                    '--lr', '5e-5',
                                    '--abstract_weighting_mode', 'normal',  # subsample
                                    '--pubmed_version', '2020',
                                    #'--num_frozen_layers', '10',
                                    '--serve_type', '0',
                                    '--only_aact_data',
                                    '--bert_pretrained_path', 'google/bert_uncased_L-2_H-128_A-2',
                                    '--bert_tokenizer', 'google/bert_uncased_L-2_H-128_A-2',
                                    '--bert_save_prefix', 'bert_tiny_uncased',
                                    '--batch_size', '16',
                                    # Used to set the path of the saved BERT model
                                    '--start_year', '2018',
                                    '--end_year', '2018',
                                    # not really used - just for using the PubMedModule.
                                    # test_pairs_file must be specified - so as not to override it with a new sample.
                                    '--test_start_year', '2020',
                                    '--test_end_year', '2020',
                                    '--test_pairs_file', 'test_similarities_CUI_names_bert_2020_2020_new_sample.csv',
                                    ])
# When running with two gpus:
# hparams.gpus = [0,1]
# When running with one gpu:
hparams.gpus = 1

dm = PubMedModule(hparams, reassign=True)
# dm = PubMedFullModule(hparams)
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
