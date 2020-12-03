import sys
sys.path.append('/home/shunita/fair_temporal/contra')

from datetime import datetime
import pytz
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from contra.datasets import PubMedFullModule
from contra import config
from contra.models import BertPretrainOnYears

hparams = config.parser.parse_args(['--name', 'BertYears', '--start_year', '2018', '--end_year', '2018'])

dm = PubMedFullModule(start_year=hparams.start_year, end_year=hparams.end_year, test_size=0.2)
model = BertPretrainOnYears(hparams)

logger = WandbLogger(name=hparams.name, save_dir=hparams.log_path,
                     version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                     project='Contra', config=hparams)
trainer = pl.Trainer(gpus=hparams.gpus, max_epochs=hparams.max_epochs, logger=logger, log_every_n_steps=10)

trainer.fit(model, datamodule=dm)
trainer.test(datamodule=dm)
