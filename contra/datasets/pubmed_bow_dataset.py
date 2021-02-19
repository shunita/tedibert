from torch.utils.data import Dataset, DataLoader
from contra.datasets import PubMedModule
from contra.experimental.exp_utils import get_vocab, texts_to_BOW


class PubMedBOWModule(PubMedModule):
    def __init__(self, hparams):
        super(PubMedBOWModule, self).__init__(hparams)

    def setup(self, stage=None):
        # self.train_df, self.val_df were filled by prepare_data
        self.vocab = get_vocab(self.train_df['text'])
        bow_train = texts_to_BOW(self.train_df['text'], self.vocab)
        bow_val = texts_to_BOW(self.val_df['text'], self.vocab)
        self.train = PubMedBOWDataset(bow_train, self.train_df['is_new'].values)
        self.val = PubMedBOWDataset(bow_val, self.val_df['is_new'].values)

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=8)


class PubMedBOWDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {'text': self.X[index], 'is_new': self.y[index]}
