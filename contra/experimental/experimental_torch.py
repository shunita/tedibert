import sys

sys.path.append('/home/shunita/fairemb/')
import os
import pytz
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiplicativeLR
from gensim.models import KeyedVectors
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer, AutoModel, BertConfig, AutoConfig
from contra import config
from contra.datasets.pubmed_bow_dataset import PubMedExpModule
from contra.experimental.exp_utils import count_old_new_appearances, texts_to_BOW
from contra.constants import EXP_PATH, SAVE_PATH, FULL_PUMBED_2020_PATH
from contra.common.utils import mean_pooling
from contra.utils.pubmed_utils import populate_idf_dict_bert
from itertools import chain
from sklearn.linear_model import LogisticRegression
from pytorch_lightning.callbacks import LearningRateMonitor
from contra.models.w2v_on_years import PretrainedW2V
from contra.utils.text_utils import TextUtils


class ExperimentalModule(pl.LightningModule):
    def __init__(self, hparams):
        super(ExperimentalModule, self).__init__()
        self.hparams = hparams

    def forward(self, batch):
        pass

    def sample_weights(self):
        pass

    def ypred_to_probabilities(self, ypred):
        return ypred

    def step(self, batch, name):
        ypred = self.forward(batch).squeeze(1)
        ytrue = batch['is_new'].float()
        loss = self.loss_func(ypred, ytrue)
        self.log(f'{name}_loss', loss)
        probs = self.ypred_to_probabilities(ypred).cpu().detach()
        if not all(ytrue) and any(ytrue):
            # Calc auc only if batch has more than one class.
            self.log(f'{name}_auc', roc_auc_score(ytrue.cpu().detach(), probs))
        self.log(f'{name}_accuracy', accuracy_score(ytrue.cpu().detach(), probs.round()))
        self.sample_weights()
        return loss

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        return self.step(batch, 'train')

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        return self.step(batch, 'val')


class LogRegBOW(ExperimentalModule):
    # Logistic Regression as a pytorch module
    def __init__(self, vocab_size, hparams):
        super(LogRegBOW, self).__init__(hparams)
        self.linear = nn.Linear(vocab_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        y_pred = self.linear(batch['text'].float())
        return y_pred

    def ypred_to_probabilities(self, ypred):
        return torch.sigmoid(ypred)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.linear.parameters(), lr=self.hparams.learning_rate)
        return [optimizer]

    def batched_forward(self, data, batch_size=32):
        res = []
        for i in range(0, len(data), batch_size):
            end_index = min(len(data), i+batch_size)
            res.append(torch.sigmoid(self.forward(data[i:end_index])))
        return torch.cat(res, axis=0)


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(SimpleClassifier, self).__init__()
        act_func = nn.ReLU(inplace=False)
        layers = [nn.Linear(in_dim, hid_dim),
                  act_func,
                  nn.Linear(hid_dim, hid_dim//2),
                  act_func,
                  nn.Linear(hid_dim//2, hid_dim // 4),
                  act_func,
                  nn.Linear(hid_dim//4, out_dim),
                  nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor):
        return self.model(input)


class W2VClassifier(ExperimentalModule):
    def __init__(self, emb_size, hparams):
        super(W2VClassifier, self).__init__(hparams)
        self.w2v = PretrainedW2V(idf_path=os.path.join(FULL_PUMBED_2020_PATH, 'idf_dict2010_13+16_18_train.pickle'),
                                 vectors_path=os.path.join(SAVE_PATH, 'word2vec_2010_13+16_18_train_1000.wordvectors'),
                                 ndocs=8780)  # abstracts in train
        # emb_size = 300
        self.tu = TextUtils()
        self.linear = nn.Linear(emb_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def word_tokenize_sample(self, sample):
        return self.tu.word_tokenize(sample)

    def forward(self, batch):
        embedded = self.w2v.embed_batch([self.word_tokenize_sample(sample) for sample in batch['text']], device=self.device)
        y_pred = self.linear(embedded)
        return y_pred

    def ypred_to_probabilities(self, ypred):
        return torch.sigmoid(ypred)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.linear.parameters(), lr=self.hparams.learning_rate)
        return [optimizer]


class W2V3SentClassifier(W2VClassifier):
    def __init__(self, emb_size, hparams):
        super(W2V3SentClassifier, self).__init__(emb_size, hparams)

    def word_tokenize_sample(self, sample):
        wordlist = []
        for sent in sample.split('<BREAK>'):
            wordlist.extend(self.tu.word_tokenize(sent))
        return wordlist


class W2VEmbeddings(nn.Module):
    def __init__(self, pretrained_vectors_path, freeze=True):
        super(W2VEmbeddings, self).__init__()
        self.wv = KeyedVectors.load(pretrained_vectors_path, mmap='r')
        self.vocab_size = len(self.wv.vocab)
        weights = torch.FloatTensor(self.wv.vectors.copy())
        weights = torch.cat([weights, torch.zeros(1, weights.shape[1])], dim=0)
        self.emb_layer = nn.Embedding.from_pretrained(weights, freeze=freeze, padding_idx=self.vocab_size)
        self.emb_layer.weight.requires_grad = not freeze
        self.unknown_words = set()

    def set_device(self, device):
        self.device = device

    def word_to_index(self, word):
        if word in self.wv.vocab:
            return self.wv.vocab[word].index
        self.unknown_words.add(word)
        return self.vocab_size

    def tokenize(self, word_list, pad_to_length):
        # turn words to indexes
        indexes = [self.word_to_index(word) for word in word_list]
        for i in range(len(indexes), pad_to_length):
            indexes.append(self.vocab_size)
        return indexes

    def batch_tokenize(self, list_of_word_lists):
        pad_len = max([len(wordlist) for wordlist in list_of_word_lists])
        tokenized = torch.as_tensor(np.stack([self.tokenize(wordlist, pad_len) for wordlist in list_of_word_lists]),
                                    device=self.device)
        # print(f"batch_tokenize: {tokenized.device}")
        return tokenized

    def forward(self, input: torch.Tensor):
        tokenized = self.batch_tokenize(input)
        # print(f"tokenized device: {tokenized.device}")
        embedded = self.emb_layer(tokenized)
        return embedded


class W2VLSTMClassifier(ExperimentalModule):
    def __init__(self, emb_size, hidden_size, hparams):
        super(W2VLSTMClassifier, self).__init__(hparams)
        self.w2v = W2VEmbeddings(os.path.join(SAVE_PATH, f'word2vec_2010_13+16_18_train_{emb_size}.wordvectors'),
                                 freeze=True)
        self.tu = TextUtils()
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def word_tokenize_sample(self, sample):
        return self.tu.word_tokenize(sample)

    def forward(self, batch):
        self.w2v.set_device(self.device)
        embedded = self.w2v([self.word_tokenize_sample(sample) for sample in batch['text']])
        output, (hn, cn) = self.lstm(embedded)
        # hn shape: (#layers * #directions) X batch size X hidden dim
        rep = torch.mean(hn, dim=0)  # result is of shape batch_size * hidden_dim
        y_pred = self.linear(rep)
        return y_pred

    # def sample_weights(self):
    #     print(f'linear params: {self.linear.weight[:10]}')
    #     print(f'emb_layer params: {[w[:10] for w in self.w2v.emb_layer.weight[:10]]}')

    def ypred_to_probabilities(self, ypred):
        return torch.sigmoid(ypred)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(chain(self.w2v.emb_layer.parameters(), self.lstm.parameters(), self.linear.parameters()),
                                     lr=self.hparams.learning_rate)
        return [optimizer]


class W2VLSTM3SentClassifier(W2VLSTMClassifier):
    def __init__(self, emb_size, hidden_size, hparams):
        super(W2VLSTM3SentClassifier, self).__init__(emb_size, hidden_size, hparams)

    def word_tokenize_sample(self, sample):
        # sample is three sentences, as text, seprarated by '<BREAK>'.
        wordlist = []
        for sent in sample.split('<BREAK>'):
            wordlist.extend(self.tu.word_tokenize(sent))
        return wordlist


class Bert3SentClassifier(ExperimentalModule):
    def __init__(self, hparams, mode='mean'):
        super(Bert3SentClassifier, self).__init__(hparams)
        self.mode = mode
        self.bert_tokenizer = AutoTokenizer.from_pretrained(hparams.bert_tokenizer)
        bert_config = AutoConfig.from_pretrained(hparams.bert_pretrained_path,
                                                 # hidden_dropout_prob=0.2,
                                                 # attention_probs_dropout_prob=0.2
                                                 )
        self.bert_model = AutoModel.from_config(bert_config)
        # move the model to evaluation mode
        self.bert_model.eval()

        # Freeze some parts of the model (unfrozen: only the pooler)
        # modules = [self.bert_model.embeddings]  #, *self.bert_model.encoder.layer]
        # for module in modules:
        #     for param in module.parameters():
        #         param.requires_grad = False

        self.sentence_embedding_size = self.bert_model.get_input_embeddings().embedding_dim
        self.max_len = 70
        classifier_size = self.sentence_embedding_size
        if self.mode == 'concat':
            classifier_size = classifier_size*3
        # self.classifier = nn.Linear(classifier_size, 1)
        self.classifier = SimpleClassifier(classifier_size, 1024, 1)
        self.loss_func = torch.nn.BCELoss()

    # def ypred_to_probabilities(self, ypred):
    #     return torch.sigmoid(ypred)

    # def sample_weights(self):
    #     print(f'classifier params: {list(self.classifier.model.parameters())[:10]}')

    def forward(self, x):
        # embed the text
        # x['text'] is a tensor where each item is a list of sentences.
        indexes = []
        all_sentences = []
        index = 0
        for sample in x['text']:
            sample_as_list = sample.split('<BREAK>')
            # sample is a list of sentences
            indexes.append((index, index + len(sample_as_list)))
            index += len(sample_as_list)
            all_sentences.extend(sample_as_list)

        inputs = self.bert_tokenizer.batch_encode_plus(all_sentences, padding=True, truncation=True,
                                                       max_length=self.max_len, add_special_tokens=True,
                                                       return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.bert_model(**inputs, output_hidden_states=False)
        sent_embedding = outputs.pooler_output  # this is not the same as last_hidden_state[0][0]!
        # sent_embedding = outputs.last_hidden_state[:, 0]
        # print(f'sent_embedding shape: {sent_embedding.shape}')
        # print(f"first sent_embedding shape:{sent_embedding[0].shape} 10 items: {sent_embedding[0][:10]})")
        sample_embedding = []
        if self.mode == 'ensemble':
            sent_pred = self.ypred_to_probabilities(self.classifier(sent_embedding))
            # sent_pred = self.classifier(sent_embedding)
            y_pred = []
        for start, end in indexes:
            if self.mode == 'concat':
                sample_vectors = [sent_embedding[i] for i in range(start, end)]
                # padding because not all samples have 3 sentences.
                for i in range(end-start, 3):
                    sample_vectors.append(torch.zeros(sent_embedding.shape[1], device=self.device))
                sample_embedding.append(torch.cat(sample_vectors))
            elif self.mode == 'mean':
                sample_embedding.append(torch.mean(sent_embedding[start:end], dim=0))
            elif self.mode == 'ensemble':
                y_pred.append(torch.mean(sent_pred[start:end]))
        if self.mode == 'ensemble':
            y_pred = torch.stack(y_pred)
        else:
            sample_embedding = torch.stack(sample_embedding)
            y_pred = self.ypred_to_probabilities(self.classifier(sample_embedding))
            # print(f"sample embedding: {sample_embedding[:5]}")
            # y_pred = self.classifier(sample_embedding)
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            # chain(self.bert_model.parameters(), self.classifier.parameters()),
            self.classifier.parameters(),
            lr=self.hparams.learning_rate,
            # weight_decay=1e-4
        )
        # def lr_factor(epoch):
        #     if epoch <= 6:
        #         return 1
        #     if epoch >= 7:
        #         return 0.75
        #     return 1
        #
        # scheduler = MultiplicativeLR(optimizer, lr_lambda=lr_factor)

        # lmbda = lambda epoch: 1.0
        # scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
        # scheduler = ReduceLROnPlateau(optimizer, patience=2, verbose=True)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': scheduler,
            # 'monitor': 'val_loss'
        }

class Bert3SentPreEmbeddedClassifier(ExperimentalModule):
    def __init__(self, hparams):
        super(Bert3SentPreEmbeddedClassifier, self).__init__(hparams)
        self.max_len = 70
        self.classifier = SimpleClassifier(128, 1024, 1)
        self.loss_func = torch.nn.BCELoss()

    # def ypred_to_probabilities(self, ypred):
    #     return torch.sigmoid(ypred)

    # def sample_weights(self):
    #     print(f'classifier params: {list(self.classifier.model.parameters())[:10]}')

    def forward(self, x):
        # x['text'] is a tensor where each item is an embedding of 2-3 sentences.
        sample_embedding = x['text']
        y_pred = self.ypred_to_probabilities(self.classifier(sample_embedding))
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams.learning_rate,
        )
        return {
            'optimizer': optimizer,
        }


class Bert3SentTransformerClassifier(Bert3SentClassifier):
    def __init__(self, hparams):
        super(Bert3SentTransformerClassifier, self).__init__(hparams, mode='mean')
        # Tranformer Aggregation
        # number of heads - following the convention from https://arxiv.org/abs/1908.08962
        self.heads_num = int(self.sentence_embedding_size/64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.sentence_embedding_size,
                                                   nhead=self.heads_num)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        # embed the text
        # x['text'] is a tensor where each item is a list of sentences.
        indexes = []
        all_sentences = []
        index = 0
        for sample in x['text']:
            sample_as_list = sample.split('<BREAK>')
            # sample is a list of sentences
            indexes.append((index, index + len(sample_as_list)))
            index += len(sample_as_list)
            all_sentences.extend(sample_as_list)

        inputs = self.bert_tokenizer.batch_encode_plus(all_sentences, padding=True, truncation=True,
                                                       max_length=self.max_len, add_special_tokens=True,
                                                       return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.bert_model(**inputs, output_hidden_states=False)
        # sent_embedding = outputs.pooler_output  # this is not the same as last_hidden_state[0][0]!
        sent_embedding = outputs.last_hidden_state  # batch_size*tokens*emb_dim
        mask = inputs['attention_mask'] == 0  # tensor with 'True' where the paddings are
        # After the transpose the input shape is: tokens * batch size * emb dim.
        # The transformer output shape is tokens * batch size * emb dim.
        # We take the first token [CLS] aggregation.
        aggregated = self.transformer_encoder(sent_embedding.transpose(0, 1), src_key_padding_mask=mask)[0]
        sample_embedding = []
        for start, end in indexes:
            sample_embedding.append(torch.mean(aggregated[start:end], dim=0))
        sample_embedding = torch.stack(sample_embedding)
        y_pred = torch.sigmoid(self.classifier(sample_embedding))
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            chain(self.transformer_encoder.parameters(), self.classifier.parameters()),
            lr=self.hparams.learning_rate)

        def lr_factor(epoch):
            if epoch <= 2:
                return 1
            if epoch == 5:
                return 0.1
            if epoch == 10:
                return 0.5
            return 1

        scheduler = MultiplicativeLR(optimizer, lr_lambda=lr_factor)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            # 'monitor': 'val_loss'
        }


class Bert3SentWordMeanClassifier(Bert3SentClassifier):
    def __init__(self, hparams):
        super(Bert3SentWordMeanClassifier, self).__init__(hparams)
        self.token_id_to_idf = populate_idf_dict_bert(hparams.bert_tokenizer)

    def forward(self, x):
        # embed the text
        # x['text'] is a tensor where each item is a list of sentences.
        indexes = []
        all_sentences = []
        index = 0
        for sample in x['text']:
            sample_as_list = sample.split('<BREAK>')
            # sample is a list of sentences
            indexes.append((index, index + len(sample_as_list)))
            index += len(sample_as_list)
            all_sentences.extend(sample_as_list)

        inputs = self.bert_tokenizer.batch_encode_plus(all_sentences, padding=True, truncation=True,
                                                       max_length=self.max_len, add_special_tokens=True,
                                                       return_tensors="pt")
        idf_per_token = inputs['input_ids'].clone().apply_(
            lambda tid: self.token_id_to_idf[tid] if tid in self.token_id_to_idf else 1.0).to(self.device)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.bert_model(**inputs, output_hidden_states=False)

        # sent_embedding = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        sent_embedding = mean_pooling(outputs.last_hidden_state, idf_per_token)
        sample_embedding = []
        for start, end in indexes:
            sample_embedding.append(torch.mean(sent_embedding[start:end], dim=0))
        sample_embedding = torch.stack(sample_embedding)
        y_pred = torch.sigmoid(self.classifier(sample_embedding))
        return y_pred


class SentenceBertWithContextBOWClassifier(pl.LightningModule):
    # 3 LogReg classifiers structure.
    def __init__(self, vocab_size, hparams):
        super(SentenceBertWithContextBOWClassifier, self).__init__(hparams)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(hparams.bert_tokenizer)
        self.bert_model = AutoModel.from_pretrained(hparams.bert_pretrained_path)
        self.sentence_embedding_size = self.bert_model.get_input_embeddings().embedding_dim
        self.max_len = 70
        self.context_classifier = nn.Linear(vocab_size, 1)
        self.sentence_classifier = nn.Linear(self.sentence_embedding_size, 1)
        self.final_classifier = nn.Linear(2, 1)
        self.BCELoss = torch.nn.BCELoss()

    def forward(self, x):
        context_pred = torch.sigmoid(self.context_classifier(x['context'].float())).to(self.device)
        # embed the text
        inputs = self.bert_tokenizer.batch_encode_plus(x['text'], padding=True, truncation=True,
                                                       max_length=self.max_len, add_special_tokens=True,
                                                       return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.bert_model(**inputs, output_hidden_states=False)
        sent_embedding = outputs.pooler_output
        #y_pred = torch.sigmoid(self.sentence_classifier(torch.cat([sent_embedding, context_pred], dim=1)))
        sent_pred = torch.sigmoid(self.sentence_classifier(sent_embedding))
        y_pred = torch.sigmoid(self.final_classifier(torch.cat([context_pred, sent_pred], dim=1)))
        return y_pred, context_pred, sent_pred

    def step(self, batch, name):
        ypred, context_pred, sent_pred = self.forward(batch)
        ypred, context_pred, sent_pred = ypred.squeeze(), context_pred.squeeze(), sent_pred.squeeze()
        ytrue = batch['is_new'].float()
        context_loss = self.BCELoss(context_pred, ytrue)
        sentence_loss = self.BCELoss(sent_pred, ytrue)
        loss = self.BCELoss(ypred, ytrue)
        self.log(f'{name}_context_loss', context_loss)
        self.log(f'{name}_sentence_loss', sentence_loss)
        self.log(f'{name}_loss', loss)
        if name == 'train':
            self.log(f'{name}_context_weight', self.final_classifier.weight.squeeze()[0])
            self.log(f'{name}_sent_weight', self.final_classifier.weight.squeeze()[1])
        if not all(ytrue) and any(ytrue):
            # Calc auc only if batch has more than one class.
            self.log(f'{name}_auc', roc_auc_score(ytrue.cpu().detach(), ypred.cpu().detach()))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(chain(self.context_classifier.parameters(),
                                           self.sentence_classifier.parameters(),
                                           self.final_classifier.parameters()),
                                     lr=self.hparams.learning_rate)
        return [optimizer]


def get_top_important_words(bow_vector, model_weights, index_to_word, n=5):
    sentence_weights = np.multiply(bow_vector, model_weights)
    top_indices = sentence_weights.argsort()[-n:][::-1]
    return ' '.join(['({}, {:.2f})'.format(index_to_word[i], sentence_weights[i]) for i in top_indices])

def train_abstract_model(dm):
    print("Training abstract model now")
    hparams = config.parser.parse_args(['--name', 'abs helper',
                                        '--first_start_year', '2011',
                                        '--first_end_year', '2013',
                                        '--second_start_year', '2016',
                                        '--second_end_year', '2018',
                                        '--batch_size', '32',
                                        '--lr', '3e-4',
                                        '--max_epochs', '20',
                                        '--test_size', '0.3',
                                        '--abstract_weighting_mode', 'normal',
                                        '--pubmed_version', '2020',
                                        '--only_aact_data',
                                        ])
    hparams.gpus = 0
    dm.train = None
    dm.serve_type = 0
    # dm = PubMedBOWModule(hparams)
    # dm.prepare_data()
    dm.setup()
    model = LogRegBOW(len(dm.index_to_word), hparams)
    logger = WandbLogger(name=hparams.name, save_dir=hparams.log_path,
                         version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                         project='Experimental', config=hparams)
    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.max_epochs,
                         logger=logger,
                         log_every_n_steps=10,
                         accumulate_grad_batches=1)
    trainer.fit(model, datamodule=dm)
    w = model.linear.weight.detach().numpy().squeeze()
    # words_and_weights = list(zip(dm.index_to_word, w))
    # word_to_weight = {word: weight for (word, weight) in words_and_weights}
    return model, w  # word_to_weight


def extract_top_features_from_BOW_model(model_weights, index_to_word, words_and_weights_fname):
    words_and_weights_file = os.path.join(EXP_PATH, words_and_weights_fname)
    words_and_weights = list(zip(index_to_word, model_weights))
    words_df = pd.DataFrame(words_and_weights, columns=['word', 'weight'])
    word_to_appearances = count_old_new_appearances(df_with_old_new_label=None)
    words_df['old_appearances'] = pd.Series([word_to_appearances[w][0] for w in dm.index_to_word])
    words_df['new_appearances'] = pd.Series([word_to_appearances[w][1] for w in dm.index_to_word])
    words_df.to_csv(words_and_weights_file)


hparams = config.parser.parse_args(['--name', 'deb 3sent bert sentmean pre-embed',
                                    '--first_start_year', '2010',
                                    '--first_end_year', '2013',
                                    '--second_start_year', '2016',
                                    '--second_end_year', '2018',
                                    '--batch_size', '64',
                                    '--lr', '1e-5',
                                    '--max_epochs', '50',
                                    '--test_size', '0.3',
                                    # '--serve_type', '2',  # single sentence as text
                                    # '--overlap_sentences', # with single sentence, no overlap will take every 3rd sentence.
                                    # '--serve_type', '4',  # 3 sentences as text
                                    # '--serve_type', '5',  # 3 sentences as BOW
                                    '--serve_type', '6',  # 3 sentences embedded by bert sentence mean
                                    '--abstract_weighting_mode', 'normal',
                                    '--pubmed_version', '2020',
                                    '--only_aact_data',
                                    # '--bert_pretrained_path',
                                    # os.path.join(SAVE_PATH, 'bert_tiny_uncased_2010_2018_v2020_epoch39'),
                                    '--bert_pretrained_path', 'google/bert_uncased_L-2_H-128_A-2',
                                    '--bert_tokenizer', 'google/bert_uncased_L-2_H-128_A-2',
                                    # '--debug',
                                    ])
hparams.gpus = 1
if __name__ == '__main__':
    dm = PubMedExpModule(hparams)
    dm.prepare_data()
    dm.setup()
    # model = LogRegBOW(len(dm.index_to_word), hparams)
    # model = W2VClassifier(emb_size=300, hparams=hparams)
    # model = W2V3SentClassifier(emb_size=1000, hparams=hparams)
    # model = Bert3SentClassifier(hparams, mode='mean')

    model = Bert3SentPreEmbeddedClassifier(hparams)

    # model = Bert3SentWordMeanClassifier(hparams)
    # model = Bert3SentTransformerClassifier(hparams)
    # model = W2VLSTM3SentClassifier(emb_size=300, hidden_size=300, hparams=hparams)
    # model = W2VLSTMClassifier(emb_size=300, hidden_size=300, hparams=hparams)
    logger = WandbLogger(name=hparams.name, save_dir=hparams.log_path,
                         version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                         project='Experimental', config=hparams)
    lr_logger = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.max_epochs,
                         logger=logger,
                         log_every_n_steps=20,
                         accumulate_grad_batches=1,
                         # callbacks=[lr_logger],
                         # gradient_clip_val=0.3
                         )
    trainer.fit(model, datamodule=dm)
    # print(f"words without embedding: {len(model.w2v.unknown_words)}")

    sys.exit()
    logger.experiment.finish()
    # Extract weights
    extract_top_features_from_BOW_model(model_weights=model.linear.weight.detach().numpy().squeeze(),
                                        index_to_word=dm.index_to_word,
                                        words_and_weights_fname='BOW_words_and_weights_old_new_by_abstract_after_filter_torch.csv')

    # analyse performance by sample
    sentence_analysis_file = os.path.join(EXP_PATH, 'sentence3_analysis_torch.csv')
    train = dm.train_df
    #bow_train = dm.bow_train.toarray()

    bow_train = torch.stack([torch.tensor(dm.train[i]['text']) for i in range(len(dm.train))])
    print(f"shape of bow_train: {bow_train.shape}, len(train)={len(train)}")
    ytrain_pred = model.batched_forward(bow_train).detach().numpy().squeeze()
    print(f"len(train)={len(train)}, shape of pred: {ytrain_pred.shape}")
    train['3sent_model_prob'] = ytrain_pred
    # which words contributed most to the model decision?
    train['top_features'] = [get_top_important_words(bow_train[i].numpy(), w, dm.index_to_word)
                             for i in range(len(bow_train))]

    # ~~~~~~~~~~~~Train a simple model on full abstracts~~~~~~~~~~
    # abstract_model = LogisticRegression()
    # abstract_model.fit(dm.abstract_train, dm.abstract_train_y)
    # abstract_for_each_sample = texts_to_BOW(train['tokenized'], dm.word_to_index)
    # train['abstract_model_prob_on_abstract'] = abstract_model.predict_proba(abstract_for_each_sample)[:, 1]
    # train['abstract_model_prob_on_3sent'] = abstract_model.predict_proba(bow_train)[:, 1]
    # word_to_abs_model_weight = {word: weight for word, weight in
    #                             zip(dm.index_to_word, abstract_model.coef_.squeeze())}

    # ~~~~~~~~~~~~Train a torch model on full abstracts~~~~~~~~~~
    # TODO: which part of the abstract did the sentence come from?
    # abstract_for_each_sample = texts_to_BOW(train['tokenized'], dm.word_to_index)
    # abstract_model, abstract_model_weights = train_abstract_model(dm)
    # train['abstract_model_prob_on_abstract'] = abstract_model.batched_forward(torch.tensor(abstract_for_each_sample.toarray())).detach().numpy().squeeze()
    # train['abstract_model_prob_on_3sent'] = abstract_model.batched_forward(bow_train).detach().numpy().squeeze()
    # train['abstract_model_top_features'] = [get_top_important_words(bow_train[i].numpy(),
    #                                                                 abstract_model_weights,
    #                                                                 dm.index_to_word)
    #                                         for i in range(len(bow_train))]

    train[['text', 'year', 'label', '3sent_model_prob', 'top_features',
           'abstract_model_prob_on_abstract', 'abstract_model_prob_on_3sent', 'abstract_model_top_features']].to_csv(
        sentence_analysis_file)



