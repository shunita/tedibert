import os
import numpy as np
from datetime import datetime
from itertools import chain
import pytorch_lightning as pl
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, BertForPreTraining
from contra.constants import SAVE_PATH


class BertPretrainOnYears(pl.LightningModule):
    def __init__(self, hparams):
        super(BertPretrainOnYears, self).__init__()
        self.hparams = hparams

        self.bert_embedding_size = 768
        self.start_year = hparams.start_year
        self.end_year = hparams.end_year

        # We use biobert tokenizer because it matches the bert tokenization plus word pieces.
        self.tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
        self.bert_model = BertForPreTraining.from_pretrained('bert-base-cased')


    def forward(self, batch):
        text = batch['text']
        # TODO: max_length is not 50! 512 is supposedly the maximum in BERT.
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512,
                                add_special_tokens=True, return_tensors="pt")
        # TODO: needed? for what?
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.bert_model(**inputs)
        # TODO: like this??
        return outputs.loss

    def step(self, batch: dict, name='train') -> dict:
        bert_loss = self.forward(batch)
        self.log(f'bert_{name}_loss', bert_loss)
        return bert_loss

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        loss = self.step(batch, name='train')
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        loss = self.step(batch, name='val')
        return loss

    def test_step(self, batch: dict, batch_idx: int):
        # TODO: is this the right place to save the model?
        # This is called when we call .test()
        self.bert_model.save_pretrained(os.path.join(SAVE_PATH, f'bert_base_cased_{self.start_year}_{self.end_year}'))
        loss = self.step(batch, name='test')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.bert_model.parameters(), lr=self.hparams.learning_rate)
        return [optimizer]
