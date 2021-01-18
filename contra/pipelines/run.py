import sys
sys.path.append('/home/shunita/fairemb')

from datetime import datetime
import pytz
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from contra.datasets import PubMedModule
from contra import config
from contra.models import FairEmbedding

hparams = config.parser.parse_args(['--name', 'GAN no disc no ratio',
                                    '--first_start_year', '2010',
                                    '--first_end_year', '2013',
                                    '--second_start_year', '2018',
                                    '--second_end_year', '2018',
                                    '--emb_algorithm', 'w2v',
                                    '--initial_emb_size', '300',
                                    #'--lr', '1e-5',
                                    '--max_epochs', '50',
                                    #'--lmb_isnew', '0.1',
                                    '--lmb_isnew', '0',
                                    '--lmb_ratio', '0',
                                    '--abstract_weighting_mode', 'normal',
                                    '--pubmed_version', '2019',
                                    '--only_aact_data',
                                    '--test_start_year', '2019',
                                    '--test_end_year', '2019',
                                    #'--test_pairs_file', 'test_interesting_CUI_pairs.csv',
                                    '--test_pairs_file', 'test_interesting_CUI_pairs_aact.csv'
                                    ])

dm = PubMedModule(min_num_participants=hparams.min_num_participants,
                  first_year_range=(hparams.first_start_year, hparams.first_end_year),
                  second_year_range=(hparams.second_start_year, hparams.second_end_year),
                  train_test_split=0.8, 
                  test_year_range=(hparams.test_start_year, hparams.test_end_year),
                  test_fname=hparams.test_pairs_file)
model = FairEmbedding(hparams)

logger = WandbLogger(name=hparams.name, save_dir=hparams.log_path,
                     version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                     project='FairEmbedding', config=hparams)
trainer = pl.Trainer(gpus=hparams.gpus, max_epochs=hparams.max_epochs, logger=logger, log_every_n_steps=10)
trainer.fit(model, datamodule=dm)
trainer.test(datamodule=dm)
