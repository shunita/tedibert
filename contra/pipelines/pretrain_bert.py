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

hparams = config.parser.parse_args(['--name', 'BertYears', 
                                    '--start_year', '2018',
                                    '--end_year', '2018',
                                    '--by_sentence',
                                    '--max_epochs', '40'])
hparams.gpus = [0,1]
#hparams.gpus = 1

dm = PubMedFullModule(start_year=hparams.start_year, end_year=hparams.end_year, test_size=0.2, by_sentence=hparams.by_sentence)
model = BertPretrainOnYears(hparams)

logger = WandbLogger(name=hparams.name, save_dir=hparams.log_path,
                     version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                     project='Contra', config=hparams)
trainer = pl.Trainer(gpus=hparams.gpus, 
                     auto_select_gpus=True, 
                     max_epochs=hparams.max_epochs, 
                     logger=logger, 
                     log_every_n_steps=10, 
                     accumulate_grad_batches=1, # no accumulation 
                     precision=16, 
                     distributed_backend='ddp')

trainer.fit(model, datamodule=dm)
#trainer.test(datamodule=dm)

if __name__ == '__main__':
    freeze_support()
