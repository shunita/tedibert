from ast import literal_eval
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn

from contra.utils.code_mapper import read_emb, CodeMapper


class BertOnDiagsBase(pl.LightningModule):
    def __init__(self, bert_model, diag_to_title, proc_to_title, lr, name, use_lstm=False):
        super(BertOnDiagsBase, self).__init__()
        self.diag_to_title = diag_to_title
        self.proc_to_title = proc_to_title
        self.bert_tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
        self.bert_model = AutoModel.from_pretrained(bert_model)
        self.emb_size = self.bert_model.get_input_embeddings().embedding_dim
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = torch.nn.LSTM(self.emb_size, self.emb_size, 1, bidirectional=False, batch_first=True)
        self.learning_rate = lr
        self.name = name

    def code_list_to_text_list(self, codes_list, lookup_dict):
        '''This will also work on drugs - without a lookup dict'''
        indexes = []
        texts = []
        index = 0
        max_len = 0
        for codes in codes_list:
            if len(codes) == 0:
                indexes.append((-1, -1))
                continue
            # sample is a list of sentences
            if lookup_dict is not None:
                titles = [lookup_dict[d] for d in codes if d in lookup_dict]
            else:
                titles = codes
            indexes.append((index, index + len(titles)))
            index += len(titles)
            texts.extend(titles)
            if max_len < len(titles):
                max_len = len(titles)
        return indexes, texts, max_len

    def embed_codes(self, x, agg, lookup_dict):
        # x - a list of lists of codes (one per patient admission).
        # First make each diag list into a sentence (description/title) list.
        indexes, texts, max_len = self.code_list_to_text_list([literal_eval(str_repr) for str_repr in x], lookup_dict)
        if len(texts) > 0:
            inputs = self.bert_tokenizer.batch_encode_plus(texts, padding=True, truncation=True,
                                                       max_length=70,
                                                       add_special_tokens=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # each title is embedded - we take the CLS token embedding
            outputs = self.bert_model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0]
        sample_embeddings = []
        # Aggregate the title embeddings into a single embedding (for each patient)
        for start, end in indexes:
            if (start, end) == (-1, -1):  # empty list
                sample_embeddings.append(torch.zeros(self.emb_size, device=self.device))
                continue
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

    def embed_diags(self, x, agg='mean'):
        return self.embed_codes(x, agg, self.diag_to_title)

    def embed_procs(self, x, agg='mean'):
        return self.embed_codes(x, agg, self.proc_to_title)

    def embed_drugs(self, x, agg='mean'):
        return self.embed_codes(x, agg, None)

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        return self.step(batch, 'train')

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        return self.step(batch, 'val')


class EmbOnDiagsBase(pl.LightningModule):
    def __init__(self, emb_path, lr, name, use_lstm):
        super(EmbOnDiagsBase, self).__init__()
        self.code_mapper = CodeMapper()
        self.emb = read_emb(emb_path)
        self.learning_rate = lr
        self.name = name
        self.emb_size = list(self.emb.values())[0].shape[0]
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = torch.nn.LSTM(self.emb_size, self.emb_size, 1, bidirectional=False, batch_first=True)

    def embed_codes(self, x, agg='mean', weights=None):
        # x - a list of strings, each representing a list of diags as cuis (one per patient admission).
        # weights - a list of lists of diag weights (idfs)
        sample_embeddings = []
        for i, admission_str in enumerate(x):
            admission = literal_eval(admission_str)

            if weights is not None:
                sample_weights = literal_eval(weights[i])
                embs = []
                for j, code in enumerate(admission):
                    embs.append(self.emb[code]*sample_weights[j])
                embs = torch.Tensor(embs).to(self.device)
            else:
                embs = torch.Tensor([self.emb[code] for code in admission]).to(self.device)
            if len(embs) == 0:
                sample_embeddings.append(torch.zeros(self.emb_size, device=self.device))
            elif agg == 'mean':
                sample_embeddings.append(torch.mean(embs, axis=0))
            elif agg == 'sum':
                sample_embeddings.append(torch.sum(embs, axis=0))
            elif agg == 'lstm':
                # operate on batches of size 1. Not the fastest way.
                # output is of shape: 1 (batch size) * num_diags * emb_size
                # TODO: clean hidden state?? when?
                # print(f"shape of embs: {embs.shape}, after unsqueeze: {embs.unsqueeze(dim=0).shape}")
                out, (hidden, cell) = self.lstm(embs.unsqueeze(dim=0))
                sample_embeddings.append(hidden.squeeze())
            elif agg == 'first':
                sample_embeddings.append(embs[0])
            else:
                raise (f"Unsupported agg method: {agg} in embed_diags")
            # print(f"shape of last added item: {sample_embeddings[-1].shape}")
        sample_embeddings = torch.stack(sample_embeddings)
        return sample_embeddings

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        return self.step(batch, 'train')

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        return self.step(batch, 'val')
