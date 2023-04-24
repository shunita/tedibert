import pytorch_lightning as pl
pl.trainer.seed_everything(42)

import os
import sys
sys.path.append(os.path.expanduser('~/fairemb/'))

from pytorch_lightning.callbacks import ModelCheckpoint
from ast import literal_eval
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
import numpy as np
import pytz
import wandb
from datetime import datetime

from contra.constants import CLINICAL_NOTES_AND_DIAGS, LOG_PATH, SAVE_PATH
from contra.datasets.clinical_notes_diags_dataset import ClinicalNotesModule, get_labels
from contra.tests.descs_and_models import DESCS_AND_MODELS, cui_embeddings
from sklearn.metrics import precision_score, recall_score, f1_score

RUN_MODEL_INDEX = 3
BATCH_SIZE = 16
LR = 1e-5
NUM_LABELS = 10
USE_ICD9_CAT = False
MAX_EPOCHS = 60
USE_LSTM = True
FOCUS_ON_DIAGS = False
START_FROM_CKPT = os.path.join(SAVE_PATH, 'notes/exp_6', 'epoch=39-step=50839.ckpt')


def row_precision(row):
    if row['ytrue_size'] == 0:
        return 0
    return row['intersect_size']/row['ytrue_size']


def row_recall(row):
    if row['ypred_size'] == 0:
        return 0
    return row['intersect_size']/row['ypred_size']


def row_accuracy(row):
    if row['union_size'] == 0:
        return 0
    return row['intersect_size']/row['union_size']


def row_f1(row):
    if row['ypred_size'] == 0 and row['ytrue_size'] == 0:
        return 0
    return 2*row['intersect_size']/(row['ypred_size'] +  row['ytrue_size'])


def calculate_metrics(df_path_or_df):
    # expected shape of ypred and ytrue - N * num_labels
    if type(df_path_or_df) == str:
        df1 = pd.read_csv(df_path_or_df, index_col=0)
        df1['pred_prob'] = df1['pred_prob'].apply(literal_eval)
        df1['labels'] = df1['labels'].apply(literal_eval)
    else:
        df1 = df_path_or_df
    ytrue = np.stack(df1['labels'].values)
    ypred = np.stack(df1['pred_prob'].values).round()
    df1['intersect_size'] = np.sum(ypred * ytrue, axis=1)
    df1['union_size'] = np.sum(np.clip(ypred+ytrue, 0, 1), axis=1)
    df1['ytrue_size'] = np.sum(ytrue, axis=1)
    df1['ypred_size'] = np.sum(ypred, axis=1)
    df1['row_precision'] = df1.apply(row_precision, axis=1)
    df1['row_recall'] = df1.apply(row_recall, axis=1)
    df1['row_f1'] = df1.apply(row_f1, axis=1)
    df1['row_accuracy'] = df1.apply(row_accuracy, axis=1)
    res = {}
    for gender in ['M', 'F']:
        # Calculate the multilabel version according to the formulas in https://arxiv.org/pdf/1802.02311v2.pdf section 3.4
        df = df1[df1['gender'] == gender]
        print(f"\ngender: {gender}, #rows: {len(df)}")
        if len(df) == 0:
            continue
        prec_multi = df['row_precision'].mean()
        recall_multi = df['row_recall'].mean()
        f1_multi = df['row_f1'].mean()
        accuracy_multi = df['row_accuracy'].mean()
        print("Multilabel metrics: precision, recall, f1, accuracy")
        # print(f"precision: {prec_multi}, recall: {recall_multi}, f1: {f1_multi}, accuracy: {accuracy_multi}")
        print(f"{prec_multi}, {recall_multi}, {f1_multi},{accuracy_multi}")
        res[f'prec_multi_{gender}'] = prec_multi
        res[f'recall_multi_{gender}'] = recall_multi
        res[f'f1_multi_{gender}'] = f1_multi
        res[f'acc_multi_{gender}'] = accuracy_multi

        ytrue = np.stack(df['labels'].values)
        ypred = np.stack(df['pred_prob'].values).round()

        # print(f"ypred shape: {ypred.shape} ytrue shape: {ytrue.shape}")
        print("Single label metrics:")
        labels = ['Metric'] + get_labels(NUM_LABELS, USE_ICD9_CAT)
        prec = precision_score(ytrue, ypred, average=None)
        recall = recall_score(ytrue, ypred, average=None)
        f1 = f1_score(ytrue, ypred, average=None)
        print(",".join([str(x) for x in labels]))
        print(",".join(['Precision'] + [str(x) for x in prec]))
        print(",".join(['Recall'] + [str(x) for x in recall]))
        print(",".join(['F1'] + [str(x) for x in f1]))
    return res


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels):
        super(Classifier, self).__init__()

        # self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        # self.linear2 = torch.nn.Linear(hidden_dim, num_labels)
        # self.activation = torch.nn.ReLU(inplace=False)

        self.linear1 = torch.nn.Linear(input_dim, num_labels)

    def forward(self, x):
        # out1 = self.activation(self.linear1(x))
        # return self.linear2(out1)
        return self.linear1(x)


class BertOnNotes(pl.LightningModule):
    def __init__(self, bert_model, lr, name, num_labels, save_dir, use_lstm=False):
        super(BertOnNotes, self).__init__()
        if bert_model == 'emilyalsentzer/Bio_ClinicalBERT':
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
        else:
            self.bert_tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
        self.bert_model = AutoModel.from_pretrained(bert_model)
        self.emb_size = self.bert_model.get_input_embeddings().embedding_dim
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = torch.nn.LSTM(self.emb_size, self.emb_size, 1, bidirectional=False, batch_first=True)
        # define the classifier
        self.classifier = Classifier(self.emb_size, self.emb_size, num_labels)
        self.learning_rate = lr
        self.name = name
        self.save_dir = save_dir
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')

    def collect_sentences_in_batch(self, list_of_sent_lists):
        indexes = []
        texts = []
        index = 0
        max_len = 0
        for sent_list in list_of_sent_lists:
            # sample is a list of sentences
            indexes.append((index, index + len(sent_list)))
            index += len(sent_list)
            texts.extend(sent_list)
            if max_len < len(sent_list):
                max_len = len(sent_list)
        return indexes, texts, max_len

    def embed_sentences(self, list_of_str_sent_lists, agg):
        # list_of_str_sent_lists - a list of string representations of lists of sentences (one list per patient admission).
        # for example: ["['first sent', 'second sent']", "['another sent', 'a fourth sent']"]
        indexes, texts, max_len = self.collect_sentences_in_batch([literal_eval(x) for x in list_of_str_sent_lists])
        if len(texts) > 0:
            inputs = self.bert_tokenizer.batch_encode_plus(texts, padding=True, truncation=True,
                                                       max_length=55,  # TODO: find the right max length
                                                       add_special_tokens=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # each title is embedded - we take the CLS token embedding
            outputs = self.bert_model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0]
        sample_embeddings = []
        # Aggregate the title embeddings into a single embedding (for each patient)
        for start, end in indexes:
            if agg == 'mean':
                sample_emb = torch.mean(outputs[start:end], dim=0)
            elif agg == 'sum':
                sample_emb = torch.sum(outputs[start:end], dim=0)
            elif agg == 'lstm':
                # operate on batches of size 1. Not the fastest way.
                # output is of shape: 1 (batch size) * num_diags * emb_size
                # TODO: clean hidden state?? when?
                out, (hidden, cell) = self.lstm(outputs[start:end].unsqueeze(dim=0))
                sample_emb = hidden.squeeze()
            elif agg == 'first':
                sample_emb = outputs[start]
            else:
                raise (f"Unsupported agg method: {agg} in embed_diags")
            sample_embeddings.append(sample_emb)
        sample_embeddings = torch.stack(sample_embeddings)
        return sample_embeddings

    def forward(self, batch, name):
        agg = 'lstm' if self.use_lstm else 'sum'
        sample_embeddings = self.embed_sentences(batch['sentences'], agg)
        ypred = self.classifier(sample_embeddings)
        return ypred

    def step(self, batch, name):
        # TODO: check shapes of all the vars
        ypred_logit = self.forward(batch, name).squeeze(1)
        # print(f"shape of ypred_logit: {ypred_logit.shape}")  # [batch_size, num_labels]
        losses = self.loss_func(ypred_logit, batch['labels'].to(torch.float32))  # calculates loss per sample
        # print(f"shape of losses: {losses.shape}")  # [batch_size, num_labels]
        loss = losses.mean()
        # print(f"shape of loss: {loss.shape}") # scalar
        self.log(f'classification/{name}_BCE_loss', loss)
        if name == 'val':
            ypred = self.y_pred_to_probabilities(self.forward(batch, 'mid_val'))
            #return {'loss': loss, 'hadm_id': batch['hadm_id'], 'pred_prob': ypred, 'gender': batch['gender'], 'labels': batch['labels']}
            return batch['hadm_id'], ypred, batch['labels'], batch['gender']
        #return {'loss': loss}
        return loss

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        return self.step(batch, 'train')

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        return self.step(batch, 'val')

    def validation_epoch_end(self, outputs) -> None:
        # sample_ids = np.concatenate([batch['hadm_id'].cpu().numpy() for batch in outputs])
        # pred_prob = np.concatenate([batch['pred_prob'].cpu().numpy().squeeze() for batch in outputs])
        # labels = np.concatenate([batch['labels'].cpu().numpy().squeeze() for batch in outputs])
        # genders = np.concatenate([batch['gender'] for batch in outputs])
        # print(f"sample ids shape: {sample_ids.shape}, pred prob shape: {pred_prob.shape}, labels shape: {labels.shape}, genders shape: {genders.shape}")
        # df = pd.DataFrame.from_dict({'hadm_id': sample_ids, 'pred_prob': pred_prob, 'labels': labels,
        #                              'gender': genders}, orient='columns')
        df = self.build_result_dataframe(outputs)
        try:
            res = calculate_metrics(df)

            self.log(f'classification/precision_F', res['prec_multi_F'])
            self.log(f'classification/precision_M', res['prec_multi_M'])
            self.log(f'classification/recall_F', res['recall_multi_F'])
            self.log(f'classification/recall_M', res['recall_multi_M'])
            self.log(f'classification/f1_F', res['f1_multi_F'])
            self.log(f'classification/f1_M', res['f1_multi_M'])
            self.log(f'classification/acc_F', res['acc_multi_F'])
            self.log(f'classification/acc_M', res['acc_multi_M'])
        except:
            print("Could not calculate metrics, probably only one class in batch.")

    def y_pred_to_probabilities(self, y_pred):
        return torch.sigmoid(y_pred)

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        ypred = self.y_pred_to_probabilities(self.forward(batch, 'test'))
        return batch['hadm_id'], ypred, batch['labels'], batch['gender']

    def build_result_dataframe(self, outputs):
        records = []
        for batch in outputs:
            # print(batch)
            batch_size = len(batch[0])
            # tuple returned by test_step
            for i in range(batch_size):
                records.append({
                    'hadm_id': batch[0][i].cpu().numpy(),
                    'pred_prob': list(batch[1][i].cpu().numpy().squeeze()),
                    'labels': list(batch[2][i].cpu().numpy().squeeze()),
                    'gender': batch[3][i],
                    })
        df = pd.DataFrame.from_records(records)
        # print(df.head())
        return df

    def test_epoch_end(self, outputs) -> None:
        df = self.build_result_dataframe(outputs)
        df.to_csv(os.path.join(self.save_dir, f'notes_test_{self.name}.csv'))
        calculate_metrics(os.path.join(self.save_dir, f'notes_test_{self.name}.csv'))
        # if CROSS_VAL is not None:
        #     self.print_aucs(df, self.name)
        # else:
        #     self.print_aucs(df)

    def configure_optimizers(self):
        grouped_parameters = [
            {'params': self.classifier.parameters()}
        ]
        if self.use_lstm:
            grouped_parameters.append({'params': self.lstm.parameters()})
        optimizer = torch.optim.Adam(grouped_parameters, lr=self.learning_rate)
        return [optimizer]


if __name__ == '__main__':
    model_name, model_path = DESCS_AND_MODELS[RUN_MODEL_INDEX]
    code_or_cat = "cats" if USE_ICD9_CAT else "codes"
    lstm_desc = '_lstm' if USE_LSTM else ''
    desc = f'notes_top{NUM_LABELS}_{code_or_cat}{lstm_desc}_{model_name}'

    next_exp_index = max([int(x.split('_')[-1]) for x in os.listdir(os.path.join(SAVE_PATH, 'notes'))]) + 1
    save_dir = os.path.join(SAVE_PATH, 'notes', f'exp_{next_exp_index}')
    os.makedirs(save_dir)

    if RUN_MODEL_INDEX in cui_embeddings:
        print("test_clinical_code_from_notes does not support CUI embeddings.")
        sys.exit()
    dm = ClinicalNotesModule(data_file=CLINICAL_NOTES_AND_DIAGS, topX=NUM_LABELS, use_icd9_cat=USE_ICD9_CAT,
                             batch_size=BATCH_SIZE, focus_on_diags=FOCUS_ON_DIAGS)
    model = BertOnNotes(model_path, lr=LR, name=desc, num_labels=NUM_LABELS, save_dir=save_dir, use_lstm=USE_LSTM)
    logger = WandbLogger(name=desc, save_dir=LOG_PATH,
                         version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                         project='FairEmbedding_test',
                         config={'lr': LR, 'batch_size': BATCH_SIZE}
                         )
    ckpt_callback = ModelCheckpoint(dirpath=save_dir, every_n_epochs=1, save_top_k=-1)
    trainer = pl.Trainer(gpus=1,
                         max_epochs=MAX_EPOCHS,
                         logger=logger,
                         log_every_n_steps=20,
                         accumulate_grad_batches=1,
                         default_root_dir=save_dir,
                         # num_sanity_val_steps=2,
                         callbacks=[ckpt_callback]
                         )
    trainer.fit(model, datamodule=dm, ckpt_path=START_FROM_CKPT)
    trainer.test(model, datamodule=dm)
