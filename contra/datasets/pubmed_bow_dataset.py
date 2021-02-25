import numpy as np
from torch.utils.data import Dataset, DataLoader
from contra.datasets import PubMedModule
from contra.experimental.exp_utils import get_vocab, texts_to_BOW, get_binary_labels_from_df
from contra.utils.text_utils import TextUtils
tu = TextUtils()


class PubMedBOWModule(PubMedModule):
    def __init__(self, hparams):
        super(PubMedBOWModule, self).__init__(hparams)

    def setup(self, stage=None):
        if self.train is not None:
            return
        # self.train_df, self.val_df were filled by prepare_data
        self.train_df['tokenized'] = self.train_df['text'].apply(tu.word_tokenize)
        self.val_df['tokenized'] = self.val_df['text'].apply(tu.word_tokenize)
        self.train_df, ytrain = get_binary_labels_from_df(self.train_df)
        self.val_df, yval = get_binary_labels_from_df(self.val_df)
        self.word_to_index, self.index_to_word = get_vocab(self.train_df['tokenized'])
        self.bow_train = texts_to_BOW(self.train_df['tokenized'], self.word_to_index)
        self.bow_val = texts_to_BOW(self.val_df['tokenized'], self.word_to_index)
        self.train = PubMedBOWDataset(self.bow_train, ytrain)
        self.val = PubMedBOWDataset(self.bow_val, yval)

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=True, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=8)


class PubMedBOWDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.X[index].toarray().squeeze().astype(np.double)
        return {'text': x, 'is_new': self.y[index]}
