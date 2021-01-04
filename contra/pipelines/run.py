import sys
sys.path.append('/home/shunita/fairemb')

from datetime import datetime
import pytz
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from contra.datasets import PubMedModule
from contra import config
from contra.models import FairEmbedding

hparams = config.parser.parse_args(['--name', 'GAN',
                                    '--first_start_year', '2010',
                                    '--first_end_year', '2013',
                                    '--second_start_year', '2018',
                                    '--second_end_year', '2018',
                                    '--emb_algorithm', 'w2v',
                                    '--initial_emb_size', '300',
                                    '--max_epochs', '50',
                                    ])

dm = PubMedModule(min_num_participants=hparams.min_num_participants,
                  first_year_range=(hparams.first_start_year, hparams.first_end_year),
                  second_year_range=(hparams.second_start_year, hparams.second_end_year),
                  train_test_split=0.8)
model = FairEmbedding(hparams)

logger = WandbLogger(name=hparams.name, save_dir=hparams.log_path,
                     version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                     project='FairEmbedding', config=hparams)
trainer = pl.Trainer(gpus=hparams.gpus, max_epochs=hparams.max_epochs, logger=logger, log_every_n_steps=10)
trainer.fit(model, datamodule=dm)
trainer.test(datamodule=dm)
