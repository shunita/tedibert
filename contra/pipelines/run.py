import sys
sys.path.append('/home/shunita/fair_temporal/contra')

from datetime import datetime
import pytz
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from contra.datasets import PubMedModule
from contra import config
from contra.models import FairEmbedding

hparams = config.parser.parse_args(['--name', 'FairEmbedding'])

dm = PubMedModule(min_num_participants=hparams.min_num_participants, pivot_datetime=hparams.pivot_datetime,
                  train_test_split=0.8)
model = FairEmbedding(hparams)

logger = WandbLogger(name=hparams.name, save_dir=hparams.log_path,
                     version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                     project='Contra', config=hparams)
trainer = pl.Trainer(gpus=hparams.gpus, max_epochs=hparams.max_epochs, logger=logger, log_every_n_steps=10)

trainer.fit(model, datamodule=dm)
trainer.test(datamodule=dm)
