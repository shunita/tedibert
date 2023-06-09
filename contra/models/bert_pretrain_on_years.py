import os
import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from contra.constants import SAVE_PATH
from contra.common.utils import break_sentence_batch


class BertPretrainOnYears(pl.LightningModule):
    def __init__(self, hparams):
        super(BertPretrainOnYears, self).__init__()
        # self.hparams = hparams
        self.learning_rate = hparams.learning_rate
        # self.bert_embedding_size = 768  # TODO: this should come from hparams? or from the model maybe, if it is even needed
        self.start_year = hparams.start_year
        self.end_year = hparams.end_year
        self.pubmed_version = hparams.pubmed_version
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.bert_tokenizer)
        self.bert_model = BertForMaskedLM.from_pretrained(hparams.bert_pretrained_path)
        self.model_desc = hparams.bert_save_prefix
        self.num_frozen_layers = hparams.num_frozen_layers
        if self.num_frozen_layers > 0:
            modules = [self.bert_model.bert.embeddings, *self.bert_model.bert.encoder.layer[:self.num_frozen_layers]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer)
        self.max_len = 70

    def forward(self, batch):
        # break the text into sentences
        indexes, all_sentences, max_len = break_sentence_batch(batch['text'])
        inputs = self.tokenizer(all_sentences, padding=True, truncation=True, max_length=self.max_len,
                                add_special_tokens=True, return_tensors="pt")
        # Collator output is a dict, and it will have 'input_ids' and 'labels'
        collated = self.data_collator(inputs['input_ids'].tolist())
        # inputs = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_len,
        #                         add_special_tokens=True, return_tensors="pt")
        # Copy the masked inputs and the original token labels to the inputs dictionary.
        inputs['labels'] = collated['labels']
        inputs['input_ids'] = collated['input_ids']
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # At this point, inputs has the 'labels' key for LM and so the loss will not be None.
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
        path = os.path.join(SAVE_PATH, '{}_{}_{}_v{}_epoch{}'.format(
            self.model_desc, self.start_year, self.end_year, self.pubmed_version, self.current_epoch))
        #if self.current_epoch > 0 and (self.current_epoch % 10 == 9) and not os.path.exists(path):
        if self.current_epoch > 0 and (self.current_epoch % 5 == 4) and not os.path.exists(path):
            self.bert_model.save_pretrained(path)
        loss = self.step(batch, name='val')
        return loss

    def test_step(self, batch: dict, batch_idx: int):
        loss = self.step(batch, name='test')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.bert_model.parameters(), lr=self.learning_rate)
        return [optimizer]
