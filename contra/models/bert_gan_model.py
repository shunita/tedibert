from itertools import chain
import torch
from transformers import AutoTokenizer, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from contra.models.model import FairEmbedding
from contra.common.utils import mean_pooling


class FairEmbeddingBert(FairEmbedding):
    def __init__(self, hparams):
        super(FairEmbeddingBert, self).__init__(hparams)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.bert_tokenizer)
        self.bert_model = BertForMaskedLM.from_pretrained(hparams.bert_pretrained_path)
        self.initial_embedding_size = self.bert_model.get_input_embeddings().embedding_dim
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer)

    def forward(self, batch):
        text = batch['text']

        # We use the same bert model for all abstracts, regardless of year.
        inputs = self.tokenizer.batch_encode_plus(text, padding=True, truncation=True, max_length=50,
                                                  add_special_tokens=True, return_tensors="pt")
        collated = self.data_collator(inputs['input_ids'].tolist())
        inputs_for_emb = {k: v.to(self.device) for k, v in inputs.items()}

        inputs_for_lm = inputs
        inputs_for_lm['input_ids'] = collated['input_ids']
        inputs_for_lm['labels'] = collated['labels']
        inputs_for_lm = {k: v.to(self.device) for k, v in inputs_for_lm.items()}

        outputs = self.bert_model(**inputs_for_emb, output_hidden_states=True)
        loss = self.bert_model(**inputs_for_lm).loss
        # TODO: inputs['attention_mask'] is just a tensor of ones so mean_pooling is a simple average
        sentence_embedding = mean_pooling(outputs['hidden_states'][-1], inputs_for_emb['attention_mask'])
        return sentence_embedding, loss

    def step(self, batch: dict, optimizer_idx: int = None, name='train') -> dict:
        if optimizer_idx == 0:
            self.ratio_true = batch['female_ratio']
            self.is_new = batch['is_new']
            # generate
            self.fair_embedding, g_loss = self.forward(batch)
            
            if self.do_ratio_prediction:
                self.ratio_pred = self.ratio_reconstruction(self.fair_embedding)
                ratio_loss = self.BCELoss(self.ratio_pred.squeeze()[self.is_new].float(), self.ratio_true[self.is_new].float())
                isnew_pred = self.discriminator(torch.cat((self.fair_embedding, self.ratio_pred), dim=1))
            else:
                isnew_pred = self.discriminator(self.fair_embedding)
            only_old_predicted = isnew_pred.squeeze()[~self.is_new]
            # we take old samples and label them as new, to train the generator.
            # Discriminator weights do not change while we train the generator because of the different optimizer_idx's.
            isnew_loss = self.BCELoss(only_old_predicted, torch.ones(only_old_predicted.shape[0], device=self.device))

            # final loss
            loss = g_loss + self.hparams.lmb_isnew * isnew_loss
            if self.do_ratio_prediction:
                loss += self.hparams.lmb_ratio * ratio_loss
                self.log(f'generator/{name}_ratio_loss', ratio_loss)
            self.log(f'generator/{name}_reconstruction_loss', g_loss)
            self.log(f'generator/{name}_discriminator_loss', isnew_loss)
            self.log(f'generator/{name}_loss', loss)

        if optimizer_idx == 1:
            # discriminate
            emb = self.fair_embedding.detach()
            
            if self.do_ratio_prediction:
                isnew_pred = self.discriminator(torch.cat([emb, self.ratio_pred.detach()], dim=1))
            else:
                isnew_pred = self.discriminator(emb)
            isnew_loss = self.BCELoss(isnew_pred.squeeze(), self.is_new.float())
            
            # final loss
            loss = isnew_loss
            self.log(f'discriminator/{name}_loss', loss)
        return loss

    def configure_optimizers(self):
        opt1_params = self.bert_model.parameters()
        if self.do_ratio_prediction:
            opt1_params = chain(opt1_params, self.ratio_reconstruction.parameters())
        optimizer_1 = torch.optim.Adam(opt1_params, lr=self.hparams.learning_rate)
        optimizer_2 = torch.optim.Adam(self.discriminator.parameters(), 
                                       lr=self.hparams.learning_rate,
                                       weight_decay=self.hparams.regularize)
        return [optimizer_1, optimizer_2]