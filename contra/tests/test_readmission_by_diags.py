import sys

sys.path.append('/home/shunita/fairemb/')
import os
import pandas as pd
import numpy as np
import random
import torch
from torch import nn
import pytz
import wandb
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from datetime import datetime
from ast import literal_eval
from collections import defaultdict
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer, AutoModel
from contra.constants import SAVE_PATH, LOG_PATH, DATA_PATH, LOS_TEST_PATH_V4
from contra.models.Transformer1D import Encoder1DLayer
from contra.datasets.test_by_diags_dataset import ReadmissionbyDiagsModule, ReadmissionbyEmbDiagsModule
from contra.tests.bert_on_diags_base import BertOnDiagsBase, EmbOnDiagsBase
from contra.constants import READMIT_TEST_PATH
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from contra.utils.delong_auc import delong_roc_test

MIMIC3_CUSTOM_TASKS_PATH = '/home/shunita/mimic3/custom_tasks/data'
MIMIC_PATH = "/home/shunita/mimic3/physionet.org/files/mimiciii/1.4/"

DESCS_AND_MODELS = [('BERT10-18_40eps', os.path.join(SAVE_PATH, 'bert_tiny_uncased_2010_2018_v2020_epoch39')),  # 0
                    ('tinybert_non_medical', 'google/bert_uncased_L-2_H-128_A-2'),  # 1
                    ('GAN20', os.path.join(SAVE_PATH, 'bert_GAN_new0.3_ref0.1_0.3_concat_epoch19')),  # 2
                    ('BERT2020_40eps', os.path.join(SAVE_PATH, 'bert_tiny_uncased_2020_2020_v2020_epoch39')),  # 3
                    ('BERT18_20eps', os.path.join(SAVE_PATH, 'bert_tiny_uncased_2018_2018_v2020_epoch19')),  # 4
                    ]
LR = 1e-5
BATCH_SIZE = 32
RUN_MODEL_INDEX = 0

# setting random seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def print_metrics(ytrue, ypred):
    auc = roc_auc_score(ytrue, ypred)
    acc = accuracy_score(ytrue, ypred.round())
    print(f"AUC: {auc}, Accuracy: {acc}")


def print_aucs_for_readmission(result_df):
    test_df = pd.read_csv(READMIT_TEST_PATH, index_col=0)
    test_df = test_df.merge(result_df, left_index=True, right_on='sample_id')
    desc = 'pred_prob'
    print("all records:")
    print_metrics(test_df['READMISSION'], test_df[desc])
    fem_subset = test_df[test_df.GENDER == 'F']
    print("female records:")
    print_metrics(fem_subset['READMISSION'], fem_subset[desc])
    male_subset = test_df[test_df.GENDER == 'M']
    print("male records:")
    print_metrics(male_subset['READMISSION'], male_subset[desc])


def print_aucs_for_los(result_df):
    test_df = pd.read_csv(LOS_TEST_PATH_V4, index_col=0)
    test_df['LOS5'] = test_df['LOS'] > 5

    print(f"{sum(test_df['LOS5'])/len(test_df)} positive (>5 days)")
    test_df = test_df.merge(result_df, left_index=True, right_on='sample_id')
    desc = 'pred_prob'
    print("all records:")
    print_metrics(test_df['LOS5'], test_df[desc])

    fem_subset = test_df[test_df.GENDER == 'F']
    print(f"female records: {sum(fem_subset['LOS5']) / len(fem_subset)} positive (>5 days)")
    print_metrics(fem_subset['LOS5'], fem_subset[desc])

    male_subset = test_df[test_df.GENDER == 'M']
    print(f"male records: {sum(male_subset['LOS5']) / len(male_subset)} positive (>5 days)")
    print_metrics(male_subset['LOS5'], male_subset[desc])


def delong_on_df(df, true_field, pred1_field, pred2_field):
    return 10**delong_roc_test(df[true_field], df[pred1_field], df[pred2_field])


def join_results(result_files_and_descs, output_fname, descs_for_auc_comparison=[]):
    test_df = pd.read_csv(READMIT_TEST_PATH, index_col=0)
    for desc, fname in result_files_and_descs:
        df = pd.read_csv(fname, index_col=0)
        df = df.rename({'pred_prob': desc}, axis=1)
        if 'sample_id' not in test_df.columns:
            test_df = test_df.merge(df, left_index=True, right_on='sample_id')
        else:
            test_df = test_df.merge(df, on='sample_id')
        test_df[f'{desc}_BCE_loss'] = -(test_df['READMISSION']*np.log(test_df[desc]) +
                                        (1 - test_df['READMISSION'])*np.log(1-test_df[desc]))
        print(f"{desc}\n~~~~~~~~~~~~~~~~")
        print_metrics(test_df['READMISSION'], test_df[desc])
        fem_subset = test_df[test_df.GENDER == 'F']
        print("Female metrics:")
        print_metrics(fem_subset['READMISSION'], fem_subset[desc])
        male_subset = test_df[test_df.GENDER == 'M']
        print("Male metrics:")
        print_metrics(male_subset['READMISSION'], male_subset[desc])
    if descs_for_auc_comparison is None and len(result_files_and_descs) == 2:
        descs_for_auc_comparison = [x[0] for x in result_files_and_descs]
    if len(descs_for_auc_comparison) == 2:
        desc1, desc2 = descs_for_auc_comparison
        print(f"comparing {desc1}, {desc2}:")
        print(f"All records: {delong_on_df(test_df, 'READMISSION', desc1, desc2)}")
        print(f"Female records ({len(test_df[test_df.GENDER == 'F'])}): {delong_on_df(test_df[test_df.GENDER == 'F'], 'READMISSION', desc1, desc2)}")
        print(f"Male records({len(test_df[test_df.GENDER == 'M'])}) :{delong_on_df(test_df[test_df.GENDER == 'M'], 'READMISSION', desc1, desc2)}")
    if output_fname is not None:
        test_df.to_csv(output_fname)


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Classifier, self).__init__()
        # self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        # self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim//2)
        # self.linear3 = torch.nn.Linear(hidden_dim//2, 1)

        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, 1)

        # self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim//2)
        # self.linear3 = torch.nn.Linear(hidden_dim//2, hidden_dim//4)
        # self.linear4 = torch.nn.Linear(hidden_dim//4, 1)
        self.activation = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        out1 = self.activation(self.linear1(x))
        return self.linear2(out1)
        # out2 = self.activation(self.linear2(out1))
        # return self.linear3(out2)
        # out3 = self.activation(self.linear3(out2))
        # return self.linear4(out3)


class BertOnDiagsWithClassifier(BertOnDiagsBase):
    def __init__(self, bert_model, diag_to_title, procedure_to_title, lr, name, use_procedures=False, use_drugs=False,
                 use_lstm=False, additionals=0, label_field='readmitted'):
        super(BertOnDiagsWithClassifier, self).__init__(bert_model, diag_to_title, procedure_to_title, lr, name, use_lstm)
        self.label_field = label_field
        if self.label_field == 'readmitted':
            self.print_aucs = print_aucs_for_readmission
        elif self.label_field == 'los':
            self.print_aucs = print_aucs_for_los

        self.emb_size = self.bert_model.get_input_embeddings().embedding_dim
        self.additionals = additionals
        cls_input_size = self.emb_size * 2 + additionals  # diags and previous diags
        self.use_procedures = use_procedures
        if self.use_procedures:
            cls_input_size += self.emb_size
        self.use_drugs = use_drugs
        if self.use_drugs:
            cls_input_size += self.emb_size


        self.classifier = Classifier(cls_input_size, 100)

        # this is for the elaborate forward_transformer
        self.activation = nn.ReLU()

        # self.heads_num = int(self.emb_size / 64)
        # self.sentence_transformer_encoder = Encoder1DLayer(d_model=self.emb_size, n_head=self.heads_num)
        # # Embedding for abstract-level CLS token:
        # self.cls = nn.Embedding(1, self.emb_size)
        # self.classifier = nn.Linear(self.emb_size, 1)
        #
        # if self.use_procedures:
        #     self.sentence_transformer_encoder2 = Encoder1DLayer(d_model=self.emb_size, n_head=self.heads_num)
        #     # Embedding for abstract-level CLS token:
        #     self.cls2 = nn.Embedding(1, self.emb_size)
        #     self.classifier2 = nn.Linear(self.emb_size, 1)
        #
        # if self.use_drugs:
        #     self.sentence_transformer_encoder3 = Encoder1DLayer(d_model=self.emb_size, n_head=self.heads_num)
        #     # Embedding for abstract-level CLS token:
        #     self.cls3 = nn.Embedding(1, self.emb_size)
        #     self.classifier3 = nn.Linear(self.emb_size, 1)
        #
        # final_classifier_size = 1 + sum([self.use_procedures, self.use_drugs])
        # if final_classifier_size > 1:
        #     self.final_classifier = nn.Linear(final_classifier_size, 1)

        # TODO: more layers? concat instead of mean? the most diags per person is 39
        # self.classifier = Classifier(emb_size, 20)
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward_generic(self, sequence, lookup_dict, sent_transformer, cls_instance, classifier_instance):
        # TODO: use diag_idfs from the batch somehow
        indexes, texts, max_len = self.code_list_to_text_list([list(literal_eval(str_repr)) for str_repr in sequence], lookup_dict)
        inputs = self.bert_tokenizer.batch_encode_plus(texts, padding=True, truncation=True,
                                                       max_length=70,
                                                       add_special_tokens=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # each title is embedded - we take the CLS token embedding
        sent_embedding = self.bert_model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0]
        sample_embedding, mask = [], []
        for start, end in indexes:
            if (start, end) == (-1, -1):
                sample_embedding.append(torch.zeros(max_len+1, self.emb_size, device=self.device))
                cur_mask = torch.ones(max_len+1, device=self.device).bool()
                cur_mask[:] = False
                mask.append(cur_mask)
                continue
            # embedding of CLS token.
            cls = cls_instance(torch.LongTensor([0]).to(self.device))
            cur_sent_embedding = sent_embedding[start:end]
            padding = torch.zeros(max_len - len(cur_sent_embedding), self.emb_size, device=self.device)
            sample_embedding.append(torch.cat([cls, cur_sent_embedding, padding], dim=0))

            cur_mask = torch.ones(max_len + 1, device=self.device).bool()
            cur_mask[len(cur_sent_embedding) + 1:] = False
            mask.append(cur_mask)
        sample_embedding = torch.stack(sample_embedding)
        mask = torch.stack(mask)
        # at this point sample_embedding holds a row for each sentence, and inside the embedded words + embedded 0 (padding).
        # the mask holds True for real words or False for padding.
        # only take the output for the cls - which represents the entire abstract.
        aggregated = sent_transformer(sample_embedding, slf_attn_mask=mask.unsqueeze(1))[0][:, 0, :]
        y_pred = classifier_instance(aggregated)
        return y_pred

    # def forward(self, x):
    #     indexes, texts, max_len = self.code_list_to_text_list([literal_eval(str_repr) for str_repr in x])
    #     inputs = self.bert_tokenizer.batch_encode_plus(texts, padding=True, truncation=True,
    #                                                    max_length=70,
    #                                                    add_special_tokens=True, return_tensors="pt")
    #     inputs = {k: v.to(self.device) for k, v in inputs.items()}
    #     # each title is embedded - we take the CLS token embedding
    #     sent_embedding = self.bert_model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0]
    #     sample_embedding, mask = [], []
    #     for start, end in indexes:
    #         # embedding of CLS token.
    #         cls = self.cls(torch.LongTensor([0]).to(self.device))
    #         cur_sent_embedding = sent_embedding[start:end]
    #         padding = torch.zeros(max_len - len(cur_sent_embedding), self.emb_size, device=self.device)
    #         sample_embedding.append(torch.cat([cls, cur_sent_embedding, padding], dim=0))
    #
    #         cur_mask = torch.ones(max_len + 1, device=self.device).bool()
    #         cur_mask[len(cur_sent_embedding) + 1:] = False
    #         mask.append(cur_mask)
    #     sample_embedding = torch.stack(sample_embedding)
    #     mask = torch.stack(mask)
    #     # at this point sample_embedding holds a row for each sentence, and inside the embedded words + embedded 0 (padding).
    #     # the mask holds True for real words or False for padding.
    #     # only take the output for the cls - which represents the entire abstract.
    #     aggregated = self.sentence_transformer_encoder(sample_embedding, slf_attn_mask=mask.unsqueeze(1))[0][:, 0, :]
    #
    #     y_pred = self.classifier(aggregated)
    #
    #     return y_pred

    def forward(self, batch, name):
        agg = 'lstm' if self.use_lstm else 'sum'
        sample_diag_embeddings = self.embed_diags(batch['diags'], agg)
        sample_prev_diag_embeddings = self.embed_diags(batch['prev_diags'], agg)
        sample_embeddings = torch.cat([sample_prev_diag_embeddings, sample_diag_embeddings], dim=1)
        if self.use_procedures:
            sample_proc_embeddings = self.embed_procs(batch['procedures'], agg)
            sample_embeddings = torch.cat([sample_embeddings, sample_proc_embeddings], dim=1)
        if self.use_drugs:
            sample_drug_embeddings = self.embed_drugs(batch['drugs'], agg)
            sample_embeddings = torch.cat([sample_embeddings, sample_drug_embeddings], dim=1)

        if self.additionals > 0:
            sample_embeddings = torch.cat([sample_embeddings, batch['additionals']], dim=1)

        ypred = self.classifier(sample_embeddings)
        return ypred

    def y_pred_to_probabilities(self, y_pred):
        return torch.sigmoid(y_pred)

    def forward_transformer(self, batch, name):
        ytrue = batch['readmitted'].to(torch.float32)
        ypred_by_diags_logit = self.forward_generic(batch['diags'],
                                                    self.diag_to_title,
                                                    self.sentence_transformer_encoder,
                                                    self.cls,
                                                    self.classifier)
        diag_losses = self.loss_func(ypred_by_diags_logit.squeeze(1), ytrue)
        self.log(f'classification/{name}_diags_BCE_loss', diag_losses.mean())

        inputs_to_final_classifier = [self.activation(ypred_by_diags_logit)]
        if self.use_procedures:
            ypred_by_procedures_logit = self.forward_generic(batch['procedures'],
                                                             self.procedure_to_title,
                                                             self.sentence_transformer_encoder2,
                                                             self.cls2,
                                                             self.classifier2)
            inputs_to_final_classifier.append(self.activation(ypred_by_procedures_logit))
            proc_losses = self.loss_func(ypred_by_procedures_logit.squeeze(1), ytrue)
            self.log(f'classification/{name}_procs_BCE_loss', proc_losses.mean())
        if self.use_drugs:
            ypred_by_drugs_logit = self.forward_generic(batch['drugs'], None, self.sentence_transformer_encoder3,
                                                        self.cls3, self.classifier3)
            inputs_to_final_classifier.append(self.activation(ypred_by_drugs_logit))
            drug_losses = self.loss_func(ypred_by_drugs_logit.squeeze(1), ytrue)
            self.log(f'classification/{name}_drugs_BCE_loss', drug_losses.mean())
        if len(inputs_to_final_classifier) > 1:
            # print(f"shape of ypred_by_diags: {ypred_by_diags_logit.shape}, ypred_by_procs: {ypred_by_procedures_logit.shape}")
            # ypred_logit = self.final_classifier(torch.cat(inputs_to_final_classifier, dim=1))
            ypred_logit = self.final_classifier(torch.cat(inputs_to_final_classifier, dim=1))
            # print(f"shape of final ypred: {ypred_logit.shape}")
        else:
            ypred_logit = ypred_by_diags_logit
        return ypred_logit

    def step(self, batch, name):
        ypred_logit = self.forward(batch, name).squeeze(1)
        losses = self.loss_func(ypred_logit, batch[self.label_field].to(torch.float32))  # calculates loss per sample
        loss = losses.mean()
        self.log(f'classification/{name}_BCE_loss', loss)

        return {'loss': loss}

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        ypred = self.y_pred_to_probabilities(self.forward(batch, 'test'))
        return batch['sample_id'], ypred

    def test_epoch_end(self, outputs) -> None:
        records = []
        for batch in outputs:
            batch_size = len(batch[0])
            # tuple returned by test_step
            for i in range(batch_size):
                records.append({
                    'sample_id': batch[0][i].cpu().numpy(),
                    'pred_prob': batch[1][i].cpu().numpy().squeeze(),
                    })
        df = pd.DataFrame.from_records(records)
        df.to_csv(os.path.join(SAVE_PATH, f'readmission_test_{self.name}.csv'))
        df = pd.read_csv(os.path.join(SAVE_PATH, f'readmission_test_{self.name}.csv'), index_col=0)
        self.print_aucs(df)


    def configure_optimizers(self):
        grouped_parameters = [
            #{'params': self.bert_model.pooler.dense.parameters()},
            # {'params': self.cls.parameters()},
            # {'params': self.sentence_transformer_encoder.parameters()},
            {'params': self.classifier.parameters()}
        ]
        if self.use_lstm:
            grouped_parameters.append({'params': self.lstm.parameters()})
        # if self.use_procedures:
        #     grouped_parameters.extend([
        #         {'params': self.cls2.parameters()},
        #         {'params': self.sentence_transformer_encoder2.parameters()},
        #         {'params': self.classifier2.parameters()}
        #     ])
        # if self.use_drugs:
        #     grouped_parameters.extend([
        #         {'params': self.cls3.parameters()},
        #         {'params': self.sentence_transformer_encoder3.parameters()},
        #         {'params': self.classifier3.parameters()}
        #     ])
        optimizer = torch.optim.Adam(grouped_parameters, lr=self.learning_rate)
        return [optimizer]


class EmbOnDiagsWithClassifier(EmbOnDiagsBase):
    def __init__(self, emb_path, lr, name, use_procedures=False, use_lstm=False, additionals=0,
                 label_field='readmitted', agg_prev_diags=None, agg_diags=None):
        # Can't use the drugs data because they are not CUIs
        super(EmbOnDiagsWithClassifier, self).__init__(emb_path, lr, name, use_lstm)
        self.label_field = label_field
        if self.label_field == 'readmitted':
            self.print_aucs = print_aucs_for_readmission
        elif self.label_field == 'los':
            self.print_aucs = print_aucs_for_los
        default_agg = 'lstm' if self.use_lstm else 'sum'
        self.agg_prev_diags = agg_prev_diags
        if self.agg_prev_diags is None:
            self.agg_prev_diags = default_agg
        self.agg_diags = agg_diags
        if self.agg_diags is None:
            self.agg_diags = default_agg

        self.emb_size = list(self.emb.values())[0].shape[0]
        cls_input_size = 2*self.emb_size  # diags + prev_diags
        self.use_procedures = use_procedures
        if self.use_procedures:
            cls_input_size += self.emb_size
        self.additionals = additionals
        cls_input_size += self.additionals

        # self.heads_num = max(1, int(self.emb_size / 64))
        # self.sentence_transformer_encoder = Encoder1DLayer(d_model=self.emb_size, n_head=self.heads_num)
        # # Embedding for patient-level CLS token:
        # self.cls = nn.Embedding(1, self.emb_size)
        # self.classifier = nn.Linear(self.emb_size, 1)

        # TODO: more layers? concat instead of mean? the most diags per person is 39
        print(f"cls input size: {cls_input_size}")
        self.classifier = Classifier(cls_input_size, 100)
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward_transformer(self, x):
        max_len = 0
        sample_embeddings = []
        for admission_str in x:
            admission = literal_eval(admission_str)
            max_len = max(max_len, len(admission))
            embs = torch.Tensor([self.emb[code] for code in admission]).to(self.device)
            sample_embeddings.append(embs)
        padded_samples, mask = [], []
        for embs in sample_embeddings:
            # embedding of CLS token.
            cls = self.cls(torch.LongTensor([0]).to(self.device))
            padding = torch.zeros(max_len - len(embs), self.emb_size, device=self.device)
            padded_samples.append(torch.cat([cls, embs, padding], dim=0))

            cur_mask = torch.ones(max_len + 1, device=self.device).bool()
            cur_mask[len(embs) + 1:] = False
            mask.append(cur_mask)
        padded_samples = torch.stack(padded_samples)
        mask = torch.stack(mask)
        # at this point sample_embedding holds a row for each sentence, and inside the embedded words + embedded 0 (padding).
        # the mask holds True for real words or False for padding.
        # only take the output for the cls - which represents the entire abstract.
        aggregated = self.sentence_transformer_encoder(padded_samples, slf_attn_mask=mask.unsqueeze(1))[0][:, 0, :]
        y_pred = self.classifier(aggregated)
        return y_pred

    def forward(self, batch):
        # agg = 'lstm' if self.use_lstm else 'sum'
        # sample_diag_embeddings = self.embed_diags(batch['diags'], agg, batch['diag_idfs'])
        sample_diag_embeddings = self.embed_codes(batch['diags'], self.agg_diags)  # no idf weighting
        sample_prev_diag_embeddings = self.embed_codes(batch['prev_diags'], self.agg_prev_diags)
        sample_embeddings = torch.cat([sample_prev_diag_embeddings, sample_diag_embeddings], dim=1)
        if self.use_procedures:
            sample_proc_embeddings = self.embed_codes(batch['procedures'], agg)
            sample_embeddings = torch.cat([sample_embeddings, sample_proc_embeddings], dim=1)

        if self.additionals > 0:
            sample_embeddings = torch.cat([sample_embeddings, batch['additionals']], dim=1)

        ypred = self.classifier(sample_embeddings)
        return ypred

    def y_pred_to_probabilities(self, y_pred):
        return torch.sigmoid(y_pred)

    def step(self, batch, name):
        ypred_logit = self.forward(batch).squeeze(1)
        losses = self.loss_func(ypred_logit, batch[self.label_field].to(torch.float32))  # calculates loss per sample
        loss = losses.mean()
        self.log(f'classification/{name}_BCE_loss', loss)
        return {'loss': loss}

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        return batch['sample_id'], self.y_pred_to_probabilities(self.forward(batch)).squeeze(1)

    def test_epoch_end(self, outputs) -> None:
        records = []
        for batch in outputs:
            batch_size = len(batch[0])
            # tuple returned by test_step
            for i in range(batch_size):
                records.append({
                    'sample_id': batch[0][i].cpu().numpy(),
                    'pred_prob': batch[1][i].cpu().numpy(),
                    })
        df = pd.DataFrame.from_records(records)
        df.to_csv(os.path.join(SAVE_PATH, f'readmission_test_{self.name}.csv'))
        df = pd.read_csv(os.path.join(SAVE_PATH, f'readmission_test_{self.name}.csv'), index_col=0)
        self.print_aucs(df)

    def configure_optimizers(self):
        grouped_parameters = [
            #{'params': self.bert_model.pooler.dense.parameters()},
            #{'params': self.cls.parameters()},
            #{'params': self.sentence_transformer_encoder.parameters()},
            {'params': self.classifier.parameters()}
        ]
        if self.use_lstm:
            grouped_parameters.append({'params': self.lstm.parameters()})
        optimizer = torch.optim.Adam(grouped_parameters, lr=self.learning_rate)
        return [optimizer]

#
# def desc_to_name(desc):
#     return f'{desc}_cls_diags_lstm_2L'

def desc_to_name(desc):
    return f'{desc}_cls_diags_drugs_procs_lstm_2L'


def find_best_threshold_for_model(res_df, model_fields, label_field='READMISSION'):
    fem = res_df[res_df.GENDER == 'F']
    male = res_df[res_df.GENDER == 'M']
    for model_field in model_fields:
        print(model_field)
        print("_________________")

        # find a single best threshold of the model
        fpr, tpr, thresholds = roc_curve(res_df[label_field], res_df[model_field])
        gmeans = (tpr * (1 - fpr)) ** 0.5
        ix = np.argmax(gmeans)
        thresh = thresholds[ix]
        print('Best Threshold={}, G-Mean={:.3f}'.format(thresholds[ix], gmeans[ix]))
        for desc, res in [("female", fem), ("male", male), ("all", res_df)]:
            print(desc)
            # calculate tpr and fpr for the found threshold
            ypred = res[model_field] > thresh
            ytrue = res[label_field] == 1 # convert to boolean
            tpr_thresh = np.sum(ypred & ytrue)/np.sum(ytrue)
            fpr_thresh = 1 - np.sum((~ypred) & (~ytrue))/np.sum(~ytrue)
            print('thresh = {}, tpr: {} fpr: {}'.format(thresh, tpr_thresh, fpr_thresh))


if __name__ == '__main__':
    desc, model_path = DESCS_AND_MODELS[RUN_MODEL_INDEX]
    cui_embeddings = [7, 8, 9, 10]
    if RUN_MODEL_INDEX in cui_embeddings:
        embs_with_missing_diags = [DESCS_AND_MODELS[i][1] for i in cui_embeddings]
        dm = ReadmissionbyEmbDiagsModule(embs_with_missing_diags, batch_size=128)
        model = EmbOnDiagsWithClassifier(model_path, lr=LR, name=desc, use_procedures=False, use_lstm=False, additionals=48)
    else:
        # Here the keys are str
        diag_dict = pd.read_csv(os.path.join(MIMIC_PATH, 'D_ICD_DIAGNOSES.csv'), index_col=0)
        diag_dict = diag_dict.set_index('ICD9_CODE')['LONG_TITLE'].to_dict()  # icd9 code to description
        # Here the keys are int
        proc_dict = pd.read_csv(os.path.join(MIMIC_PATH, 'D_ICD_PROCEDURES.csv'), index_col=0)
        proc_dict = proc_dict.set_index('ICD9_CODE')['LONG_TITLE'].to_dict()
        dm = ReadmissionbyDiagsModule(diag_dict, proc_dict, batch_size=BATCH_SIZE)
        model = BertOnDiagsWithClassifier(model_path, diag_dict, proc_dict, lr=LR, name=desc_to_name(desc),
                                          use_procedures=True, use_drugs=True, use_lstm=True, additionals=0) # additionals=110

    logger = WandbLogger(name=desc_to_name(desc), save_dir=LOG_PATH,
                         version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                         project='FairEmbedding_test',
                         config={'lr': LR, 'batch_size': BATCH_SIZE}
                         )
    trainer = pl.Trainer(gpus=1,
                         max_epochs=4,
                         logger=logger,
                         log_every_n_steps=20,
                         accumulate_grad_batches=1,
                         #num_sanity_val_steps=2,
                         )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    # to_show = [DESCS_AND_MODELS[i] for i in (1, 3, 5, 6)]
    # join_results([(desc, os.path.join(SAVE_PATH, f'readmission_test_{desc_to_name(desc)}.csv')) for desc, path in to_show],
    #              output_fname=os.path.join(SAVE_PATH, 'readmission_test_with_attrs_with2020.csv'))

