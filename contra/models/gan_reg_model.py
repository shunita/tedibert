
import sys

from pytorch_lightning import Callback, Trainer

sys.path.append('/home/shunita/fairemb/')
import wandb
from scipy.stats import stats
from contra.models.Transformer1D import Encoder1DLayer
import os
import pytz
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
torch.cuda.empty_cache()
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer, AutoModel, BertConfig, AutoConfig, DataCollatorForLanguageModeling, \
    BertForMaskedLM
from contra import config
# from contra.datasets.pubmed_bow_dataset import PubMedExpModule
from contra.datasets.pubmed_dataset import PubMedModule
from contra.constants import EXP_PATH, SAVE_PATH, FULL_PUMBED_2020_PATH, DATA_PATH
from contra.common.utils import mean_pooling, break_sentence_batch
from itertools import chain
from pytorch_lightning.callbacks import LearningRateMonitor


class GAN(pl.LightningModule):
    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.hparams = hparams

        self.bert_tokenizer = AutoTokenizer.from_pretrained(hparams.bert_tokenizer)
        self.bert_model = BertForMaskedLM.from_pretrained(hparams.bert_pretrained_path)
        self.frozen_bert_model = AutoModel.from_pretrained(os.path.join(SAVE_PATH, 'bert_tiny_uncased_2010_2018_v2020_epoch39'))
        # self.frozen_bert_model = AutoModel.from_pretrained(
        #     os.path.join(SAVE_PATH, 'bert_bert10-13+16-18_epoch39'))
        self.data_collator = DataCollatorForLanguageModeling(self.bert_tokenizer)
        self.sentence_embedding_size = self.bert_model.get_input_embeddings().embedding_dim
        self.max_len = 50
        self.max_sentences = 20

        self.heads_num = int(self.sentence_embedding_size/64)
        self.sentence_transformer_encoder = Encoder1DLayer(d_model=self.sentence_embedding_size, n_head=self.heads_num)
        # Embedding for abstract-level CLS token:
        self.cls = nn.Embedding(1, self.sentence_embedding_size)
        if self.hparams.agg_sentences == 'transformer':
            self.classifier = nn.Linear(self.sentence_embedding_size, 1)
        elif self.hparams.agg_sentences == 'concat':
            # TODO: masking?
            self.classifier = nn.Linear(self.sentence_embedding_size*self.max_sentences, 1)
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.test_mlm = False

    def forward(self, texts):
        inputs = self.bert_tokenizer.batch_encode_plus(texts, padding=True, truncation=True,
                                                       max_length=self.max_len,
                                                       add_special_tokens=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.bert_model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0]
        return outputs

    def y_pred_to_probabilities(self, y_pred):
        return torch.sigmoid(y_pred)

    def generator_step(self, inputs):
        """returns MLM loss"""
        collated_inputs = self.data_collator(inputs['input_ids'].tolist())
        collated_inputs = {k: v.to(self.device) for k, v in collated_inputs.items()}

        inputs['input_ids'] = collated_inputs['input_ids']
        inputs['labels'] = collated_inputs['labels']
        loss = self.bert_model(**inputs).loss

        return loss

    def discriminator_step(self, sent_embedding, indexes, max_len):
        if self.hparams.agg_sentences == 'transformer':
            return self.discriminator_step_transformer(sent_embedding, indexes, max_len)
        else:
            return self.discriminator_step_concat(sent_embedding, indexes, max_len)

    def discriminator_step_concat(self, sent_embedding, indexes, max_len):
        """returns old/new prediction probabilities"""
        sample_embedding = []
        for start, end in indexes:
            cur_sent_embedding = sent_embedding[start:end]
            if len(cur_sent_embedding) > self.max_sentences:  # Too many sentences
                cur_sent_embedding = cur_sent_embedding[:self.max_sentences]
                sample_embedding.append(torch.flatten(cur_sent_embedding))
            else:  # Too few sentences - add padding
                padding = torch.zeros(self.max_sentences - len(cur_sent_embedding), self.sentence_embedding_size,
                                      device=self.device)
                sample_embedding.append(torch.flatten(torch.cat([cur_sent_embedding, padding], dim=0)))

        aggregated = torch.stack(sample_embedding)
        # at this point sample_embedding holds a row for each abstract. Each row is max_sentences * sentence_embedding_size long.
        y_pred = self.classifier(aggregated).squeeze(1)
        return y_pred

    def discriminator_step_transformer(self, sent_embedding, indexes, max_len):
        """returns old/new prediction probabilities"""
        sample_embedding, mask = [], []
        for start, end in indexes:
            # embedding of CLS token.
            cls = self.cls(torch.LongTensor([0]).to(self.device))
            cur_sent_embedding = sent_embedding[start:end]
            padding = torch.zeros(max_len - len(cur_sent_embedding), self.sentence_embedding_size, device=self.device)
            sample_embedding.append(torch.cat([cls, cur_sent_embedding, padding], dim=0))

            cur_mask = torch.ones(max_len + 1, device=self.device).bool()
            cur_mask[len(cur_sent_embedding) + 1:] = False
            mask.append(cur_mask)
        sample_embedding = torch.stack(sample_embedding)
        mask = torch.stack(mask)
        # at this point sample_embedding holds a row for each sentence, and inside the embedded words + embedded 0 (padding).
        # the mask holds True for real words or False for padding.
        # TODO: understand the dimensions here:
        # only take the output for the cls - which represents the entire abstract.
        aggregated = self.sentence_transformer_encoder(sample_embedding, slf_attn_mask=mask.unsqueeze(1))[0][:, 0, :]

        y_pred = self.classifier(aggregated).squeeze(1)

        return y_pred

    def step(self, batch, optimizer_idx, name):
        #print(f"step: optimizer_idx = {optimizer_idx}")
        y_true = batch['is_new'].float()

        indexes, all_sentences, max_len = break_sentence_batch(batch['text'])

        # We use the same bert model for all abstracts, regardless of year.
        inputs = self.bert_tokenizer.batch_encode_plus(all_sentences, padding=True, truncation=True,
                                                       max_length=self.max_len,
                                                       add_special_tokens=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.bert_model(**inputs, output_hidden_states=True).hidden_states[-1]
        reference_outputs = self.frozen_bert_model(**inputs, output_hidden_states=True).hidden_states[-1]
        # compute loss based on the difference of the two outputs
        diff_from_frozen = torch.norm(outputs - reference_outputs, 2)
        # Going further, only the last output of the first (CLS) token is needed
        outputs = outputs[:, 0]
        if optimizer_idx == 0:
            mlm_loss = 0
            y_pred = self.discriminator_step(outputs, indexes, max_len)
            losses = self.loss_func(y_pred, y_true)  # calculates loss per sample
            loss = losses.mean()
            self.log(f'discriminator/{name}_loss', loss)

            y_proba = self.y_pred_to_probabilities(y_pred).cpu().detach()

            if not all(y_true) and any(y_true):
                # Calc auc only if batch has more than one class.
                self.log(f'discriminator/{name}_auc', roc_auc_score(y_true.cpu().detach(), y_proba))
            self.log(f'discriminator/{name}_accuracy', accuracy_score(y_true.cpu().detach(), y_proba.round()))

        if optimizer_idx == 1:
            mlm_loss = self.generator_step(inputs)

            y_pred = self.discriminator_step(outputs, indexes, max_len)
            y_proba = self.y_pred_to_probabilities(y_pred).cpu().detach()

            # we take old samples and label them as new (old - 0, new - 1), to train the generator.
            # ypred_only_old = y_pred[~y_true.long()]
            # losses = self.loss_func(ypred_only_old, torch.ones(ypred_only_old.shape[0], device=self.device))
            # instead, we could train on confusing the discriminator both ways.
            losses = self.loss_func(y_pred, 1-y_true)

            d_loss = losses.mean()
            loss = (1-self.hparams.lmb_isnew-self.hparams.lmb_ref) * mlm_loss + \
                self.hparams.lmb_isnew * d_loss + \
                self.hparams.lmb_ref * diff_from_frozen * 0.1  # this is in O(100) while the others are O(1)

            self.log(f'generator/{name}_loss', loss)
            self.log(f'generator/{name}_mlm_loss', mlm_loss)
            self.log(f'generator/{name}_frozen_diff_loss', diff_from_frozen)
            self.log(f'generator/{name}_discriminator_loss', d_loss)

        return {'loss': loss,
                'losses': losses.detach().cpu(),
                'mlm_loss': mlm_loss,
                'text': batch['text'],
                'y_true': y_true.cpu(),
                'y_proba': y_proba,
                'y_score': y_true.cpu().detach() * y_proba + (1-y_true.cpu().detach()) * (1-y_proba),
                'optimizer_idx': optimizer_idx}

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        return self.step(batch, optimizer_idx, 'train')

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        outs = []
        for i in range(len(self.optimizers())):
            outs.append(self.step(batch, i, 'val'))
        # print(f"outs[1] keys: {outs[1].keys()}")
        return outs[1]  # generator

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        if self.test_mlm:
            return self.validation_step(batch, batch_idx, optimizer_idx)
        CUI1_embedding = self.forward(batch['CUI1'])
        CUI2_embedding = self.forward(batch['CUI2'])
        pred_similarity = nn.CosineSimilarity()(CUI1_embedding, CUI2_embedding)
        true_similarity = batch['true_similarity']
        return pred_similarity, true_similarity

    def test_epoch_end(self, outputs) -> None:
        if self.test_mlm:
            self.log('test/val_mlm_loss_avg', torch.Tensor([output['mlm_loss'] for output in outputs]).mean())
        else:
            rows = torch.cat([torch.stack(output) for output in outputs], axis=1).T.cpu().numpy()
            df = pd.DataFrame(rows, columns=['pred_similarity', 'true_similarity'])
            df.to_csv(os.path.join(SAVE_PATH, f'test_similarities_{self.hparams.name}.csv'))
            df = df.sort_values(['true_similarity'], ascending=False).reset_index()
            true_rank = list(df.index)
            pred_rank = list(df.sort_values(['pred_similarity'], ascending=False).index)
            correlation, pvalue = stats.spearmanr(true_rank, pred_rank)
            self.log('test/correlation', correlation)
            self.log('test/pvalue', pvalue)
            if self.hparams.max_epochs > 0:
                self.bert_model.save_pretrained(
                    os.path.join(SAVE_PATH, f'bert_{self.hparams.name}_epoch{self.hparams.max_epochs-1}'))

    def training_epoch_end(self, outputs):
        self.on_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.on_end(outputs, 'val')

    def on_end(self, outputs, name):
        # outputs is a list (len=number of batches) of dicts (as returned from the step methods).
        if name == 'train':
            outputs = outputs[0]  # TODO: WHY? only generator outputs. TODO: really?
        losses = torch.cat([output['losses'] for output in outputs])
        y_true = torch.cat([output['y_true'] for output in outputs])
        y_proba = torch.cat([output['y_proba'] for output in outputs])
        y_score = torch.cat([output['y_score'] for output in outputs])

        self.log(f'debug/{name}_loss_histogram', wandb.Histogram(losses))
        self.log(f'debug/{name}_probability_histogram', wandb.Histogram(y_proba))
        self.log(f'debug/{name}_score_histogram', wandb.Histogram(y_score))
        self.log(f'debug/{name}_loss', losses.mean())
        self.log(f'debug/{name}_accuracy', (1.*((1.*(y_proba >= 0.5)) == y_true)).mean())
        self.log(f'debug/{name}_1_accuracy', (1.*(y_proba[y_true == 1] >= 0.5)).mean())
        self.log(f'debug/{name}_0_accuracy', (1.*(y_proba[y_true == 0] < 0.5)).mean())

        if name == 'val':
            texts = np.concatenate([output['text'] for output in outputs])

            df = pd.DataFrame({'text': texts, 'y_true': y_true, 'y_score': y_score, 'y_proba': y_proba, 'loss': losses})
            df = df[(df['loss'] <= df['loss'].quantile(0.05)) | (df['loss'] >= df['loss'].quantile(0.95))]

            self.log(f'debug/{name}_table', wandb.Table(dataframe=df))

    def configure_optimizers(self):
        # Discriminator step paramteres - bert model (sometimes), transformer and classifier.
        grouped_parameters0 = [
            # {'params': chain(*[x.parameters() for x in list(self.bert_model.children())[:1]])},
            # {'params': chain(*[x.parameters() for x in list(self.bert_model.children())[1:]])},
            {'params': self.cls.parameters()},
            {'params': self.sentence_transformer_encoder.parameters()},
            {'params': self.classifier.parameters()}
        ]
        if self.hparams.lmb_isnew > 0:
            grouped_parameters0.append({'params': self.bert_model.parameters()})
        optimizer0 = torch.optim.Adam(grouped_parameters0, lr=self.hparams.learning_rate)
        # Generator step parameters - only the bert model.
        grouped_parameters1 = [
            # {'params': chain(*[x.parameters() for x in list(self.bert_model.children())[:1]])},
            # {'params': chain(*[x.parameters() for x in list(self.bert_model.children())[1:]])},
            {'params': self.bert_model.parameters()},
            # {'params': self.cls.parameters()},
            # {'params': self.sentence_transformer_encoder.parameters()},
            # {'params': self.classifier.parameters()}
        ]
        optimizer1 = torch.optim.Adam(grouped_parameters1, lr=self.hparams.learning_rate)
        return [optimizer0, optimizer1]

    def generate_cui_embeddings(self):
        df = pd.read_csv(os.path.join(DATA_PATH, 'cui_and_name_for_com_class.csv'), index_col=0)
        str_embs = []
        for i in range(0, len(df), 32):
            embedded = self.forward(df.iloc[i:i + 32]['name'].values.tolist()).detach().cpu().numpy()
            for emb in embedded:
                str_embs.append(",".join([str(x) for x in emb]))
        df['emb'] = str_embs
        df[['cui', 'emb']].to_csv(os.path.join(SAVE_PATH, f'cui_{self.hparams.name}_ep{self.hparams.max_epochs}_emb.tsv'),
                                  sep='\t', header=False, index=False)


hparams = config.parser.parse_args(['--name', 'medical_bert_specialized20', #'GAN_new0.3_ref0.3_concat_1sideloss',
                                    '--first_start_year', '2010',
                                    '--first_end_year', '2013',
                                    '--second_start_year', '2016',
                                    '--second_end_year', '2018',
                                    '--test_start_year', '2018',
                                    '--test_end_year', '2018',
                                    '--batch_size', '16',
                                    '--lr', '2e-5',
                                    '--lmb_isnew', '0',  # '0.3'
                                    '--lmb_ref', '0',  # '0.3'
                                    '--agg_sentences', 'concat',
                                    '--max_epochs', '20',  #'10',
                                    '--test_size', '0.3',
                                    '--serve_type', '0',  # Full abstract
                                    # '--serve_type', '2',  # single sentence as text
                                    # '--serve_type', '4',  # 3 sentences as text
                                    # '--serve_type', '5',  # 3 sentences as BOW
                                    # '--overlap_sentences', # with single sentence, no overlap will take every 3rd sentence.
                                    '--abstract_weighting_mode', 'normal',
                                    '--pubmed_version', '2020',
                                    '--only_aact_data',
                                    '--bert_pretrained_path', os.path.join(SAVE_PATH, 'bert_tiny_uncased_2010_2018_v2020_epoch39'),
                                    # bert_GAN_new0.3_ref0.3_concat_1sideloss_epoch19
                                    #'--bert_pretrained_path', os.path.join(SAVE_PATH, 'bert_GAN_new0.3_ref0.1_0.3_concat_epoch19'),
                                    # '--bert_pretrained_path', 'google/bert_uncased_L-2_H-128_A-2',
                                    '--bert_tokenizer', 'google/bert_uncased_L-2_H-128_A-2',
                                    # '--debug',
                                    # '--test_pairs_file', 'test_similarities_CUI_names_bert_pos_fp_2020_2020.csv',
                                    '--test_pairs_file', 'test_similarities_CUI_names_bert_2020_2020_new_sample.csv',
                                    # '--test_pairs_file', 'test_similarities_CUI_names_bert_2018_2018.csv',
                                    ])
hparams.gpus = 1
#hparams.name = f'GAN_lmb{hparams.lmb_isnew}'
if __name__ == '__main__':
    dm = PubMedModule(hparams)
    model = GAN(hparams)
    logger = WandbLogger(name=hparams.name, save_dir=hparams.log_path,
                         version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                         project='Experimental', config=hparams)
    #lr_logger = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.max_epochs,
                         logger=logger,
                         log_every_n_steps=20,
                         accumulate_grad_batches=1,
                         # callbacks=[lr_logger],
                         num_sanity_val_steps=0,
                         # gradient_clip_val=0.3
                         )
    #if hparams.max_epochs == 0:
        #model.set_output_validation_flag()
        #trainer.predict(model, dataloaders=[dm.val_dataloader()])
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    model.test_mlm = True
    dm2 = PubMedModule(hparams, test_mlm=True)
    trainer.test(model, datamodule=dm2)
    #model.generate_cui_embeddings()
    # model.set_output_validation_flag()
    #trainer.predict(model, datamodule=dm)
