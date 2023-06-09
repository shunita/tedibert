import sys
sys.path.append('/home/shunita/fairemb')

import os
from datetime import datetime
import pytz
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from contra.datasets import PubMedModule
from contra import config
from contra.models import FairEmbeddingBert, FairEmbeddingW2V
from contra.constants import SAVE_PATH

hparams = config.parser.parse_args([
    '--name', 'GAN tinybert simplest D no G',
    '--first_start_year', '2011',
    '--first_end_year', '2013',
    '--second_start_year', '2016',
    '--second_end_year', '2018',
    '--emb_algorithm', 'bert',  # 'w2v',
    # '--dim', '768',  # bert-base-cased
    '--dim', '128',  # tinybert
    # '--bert_pretrained_path', 'google/bert_uncased_L-2_H-128_A-2',
    '--bert_pretrained_path', os.path.join(SAVE_PATH, 'bert_tiny_uncased_2010_2018_v2020_epoch39'),
    '--bert_tokenizer', 'google/bert_uncased_L-2_H-128_A-2',
    '--batch_size', '16',
    #'--initial_emb_size', '300',  # w2v
    '--lr', '1e-5',
    '--max_epochs', '10',
    #'--lmb_isnew', '0.1',
    '--lmb_isnew', '0',
    '--lmb_ratio', '0',
    '--abstract_weighting_mode', 'normal',
    '--pubmed_version', '2020',
    '--only_aact_data',
    '--test_start_year', '2020',
    '--test_end_year', '2020',
    '--test_pairs_file', 'test_similarities_CUI_names_tinybert_2020_2020.csv',
    #'--debug',
    ])
hparams.gpus = 1
dm = PubMedModule(hparams)
if hparams.emb_algorithm == 'bert':
    model = FairEmbeddingBert(hparams)
elif hparams.emb_algorithm == 'w2v':
    model = FairEmbeddingW2V(hparams)
else:
    raise Exception(f"Unsupported embedding algorithm: {hparams.emb_algorithm}")

logger = WandbLogger(name=hparams.name, save_dir=hparams.log_path,
                     version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                     project='FairEmbedding', config=hparams)
trainer = pl.Trainer(gpus=hparams.gpus,
                     max_epochs=hparams.max_epochs,
                     logger=logger,
                     log_every_n_steps=10,
                     accumulate_grad_batches=1) # gradient_clip_val=0.5
trainer.fit(model, datamodule=dm)

if hparams.max_epochs == 0:
    # Use last available checkpoint. In this case it's also the only one.
    ckpt_path = None
else:
    # Use the best performing checkpoint
    ckpt_path = 'best'

trainer.test(datamodule=dm, ckpt_path=ckpt_path)
