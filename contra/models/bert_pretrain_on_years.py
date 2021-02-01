import os
import numpy as np
from datetime import datetime
from itertools import chain
import pytorch_lightning as pl
import torch
from torch import nn
from transformers import AutoTokenizer, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from contra.constants import SAVE_PATH


class BertPretrainOnYears(pl.LightningModule):
    def __init__(self, hparams):
        super(BertPretrainOnYears, self).__init__()
        self.hparams = hparams

        self.bert_embedding_size = 768
        self.start_year = hparams.start_year
        self.end_year = hparams.end_year
        self.pubmed_version = hparams.pubmed_version

        # We use biobert tokenizer because it matches the bert tokenization, but also has word pieces.
        self.tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

        pretrained = 'bert-base-cased'
        if hparams.bert_pretrained_path is not None:
            pretrained = hparams.bert_pretrained_path
        self.bert_model = BertForMaskedLM.from_pretrained(pretrained)
        self.num_frozen_layers = hparams.num_frozen_layers
        if self.num_frozen_layers > 0:
            modules = [self.bert_model.bert.embeddings, *self.bert_model.bert.encoder.layer[:self.num_frozen_layers]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer)
        self.by_sentence = hparams.by_sentence
        self.max_len = 70 if self.by_sentence else 200
        print(f'initialized BertPretrainOnYears with {self.by_sentence}, with max len: {self.max_len}')


    def forward(self, batch):
        text = batch['text']
        tokenized_as_list = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_len, add_special_tokens=True)
        # Collator output is a dict, and it will have 'input_ids' and 'labels'
        collated = self.data_collator(tokenized_as_list['input_ids'])
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_len,
                                add_special_tokens=True, return_tensors="pt")
        inputs['labels'] = collated['labels']
        inputs['input_ids'] = collated['input_ids']
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # At this point, inputs has the 'labels' key for LM and so the loss will not be None.
        #print(f"shape of input_ids: {inputs['input_ids'].shape}, shape of labels: {inputs['labels'].shape}")
        
        x = self.bert_model(**inputs)
        return x.loss

    def step(self, batch: dict, name='train') -> dict:
        bert_loss = self.forward(batch)
        self.log(f'bert_{name}_loss', bert_loss)
        return bert_loss

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        loss = self.step(batch, name='train')
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        path = os.path.join(SAVE_PATH, f'bert_base_cased_{self.start_year}_{self.end_year}_v{self.pubmed_version}_epoch{self.current_epoch}')
        if self.current_epoch > 0 and not os.path.exists(path):
            self.bert_model.save_pretrained(path)
        loss = self.step(batch, name='val')
        return loss

    def test_step(self, batch: dict, batch_idx: int):
        loss = self.step(batch, name='test')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.bert_model.parameters(), lr=self.hparams.learning_rate)
        return [optimizer]
