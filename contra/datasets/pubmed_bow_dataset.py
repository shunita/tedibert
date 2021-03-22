import numpy as np
from torch.utils.data import Dataset, DataLoader
from contra.datasets import PubMedModule
from contra.experimental.exp_utils import get_vocab, texts_to_BOW, text_seq_to_BOW, fill_binary_year_label
from contra.utils.text_utils import TextUtils
tu = TextUtils()


class PubMedExpModule(PubMedModule):
    def __init__(self, hparams):
        '''
        :param hparams:
        :param filter_by_containing_words:
        :param serve_type: 0 - full abstract as text
                           1 - full abstract as BOW
                           2 - single sentence as text
                           3 - single sentence as BOW
                           4 - three sentences as text
                           5 - three sentences as BOW
        '''
        super(PubMedExpModule, self).__init__(hparams)

        self.num_top_words = 2
        self.serve_type = hparams.serve_type
        self.index_to_word = None
        self.word_to_index = None

    def filter_by_containing_words(self, wordlist):
        # check if at least <num_top_words> of the words in the sentence (wordlist)
        # is one of the words that we care about.
        found = 0
        for word in wordlist:
            if word in self.containing_words:
                found += 1
        return found >= self.num_top_words

    def setup(self, stage=None):
        if self.train is not None:
            return
        # self.train_df, self.val_df were filled by prepare_data
        # The next two lines fill the 'label' field.
        self.train_df = fill_binary_year_label(self.train_df, self.first_time_range, self.second_time_range)
        self.val_df = fill_binary_year_label(self.val_df, self.first_time_range, self.second_time_range)
        # The text field is either one sentence (options 2,3,4,5) or the full abstract (0,1).
        self.train_df['tokenized'] = self.train_df['text'].apply(tu.word_tokenize)
        self.val_df['tokenized'] = self.val_df['text'].apply(tu.word_tokenize)
        # build a BOW vocab based on the train.
        if self.index_to_word is None:
            abstracts = self.train_df['title_and_abstract'].copy().drop_duplicates(inplace=False).apply(tu.word_tokenize)
            self.word_to_index, self.index_to_word = get_vocab(abstracts)
        ytrain = self.train_df['label'].values
        yval = self.val_df['label'].values

        if self.serve_type in [0, 2]:  # abstract or a single sentence as text
            self.train = PubMedExpTextDataset(self.train_df, ytrain)
            self.val = PubMedExpTextDataset(self.val_df, yval)

        if self.serve_type in [1, 3]:  # abstract or a single sentence as BOW
            self.bow_train = texts_to_BOW(self.train_df['tokenized'], self.word_to_index)
            self.bow_val = texts_to_BOW(self.val_df['tokenized'], self.word_to_index)
            self.train = PubMedExpBOWDataset(self.bow_train, ytrain)
            self.val = PubMedExpBOWDataset(self.bow_val, yval)

        elif self.serve_type == 4:
            self.train = PubMedExp3SentTextDataset(self.train_df, ytrain)
            self.val = PubMedExp3SentTextDataset(self.val_df, yval)
        elif self.serve_type == 5:
            self.train = PubMedExp3SentBOWDataset(self.train_df, ytrain, self.word_to_index)
            self.val = PubMedExp3SentBOWDataset(self.val_df, yval, self.word_to_index)

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=8)


class PubMedExpBOWDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.X[index].toarray().squeeze().astype(np.double)
        return {'text': x, 'is_new': self.y[index]}

class PubMedExpTextDataset(Dataset):
    def __init__(self, df, y):
        self.df = df
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.df.iloc[index]['text']
        return {'text': x, 'is_new': self.y[index]}


class PubMedExp3SentDatasetBase(Dataset):
    def __init__(self, df, y, context_field_name='sentences', sentence_field_name='text'):
        self.df = df
        self.y = y
        # self.word_to_index = word_to_index
        self.context_field_name = context_field_name
        self.sentence_field_name = sentence_field_name

    def __len__(self):
        return len(self.y)

    def create_text_field(self, sent_list):
        return None

    def __getitem__(self, index):
        line = self.df.iloc[index]
        sents = line[self.context_field_name]
        sentence_position = line['pos']
        context_positions = []
        if sentence_position > 0:
            context_positions.append(sentence_position - 1)
        context_positions.append(sentence_position)
        if sentence_position + 1 < len(sents):
            context_positions.append(sentence_position + 1)
        return {'text': self.create_text_field([sents[i] for i in context_positions]),
                'is_new': self.y[index]}


class PubMedExp3SentBOWDataset(PubMedExp3SentDatasetBase):
    def __init__(self, df, y, word_to_index, context_field_name='sentences', sentence_field_name='text'):
        super(PubMedExp3SentBOWDataset, self).__init__(df, y, context_field_name, sentence_field_name)
        self.word_to_index = word_to_index

    def create_text_field(self, sent_list):
        context = ' '.join(sent_list)
        return text_seq_to_BOW(tu.word_tokenize(context), self.word_to_index).astype(np.double)


class PubMedExp3SentTextDataset(PubMedExp3SentDatasetBase):
    def __init__(self, df, y, context_field_name='sentences', sentence_field_name='text'):
        super(PubMedExp3SentTextDataset, self).__init__(df, y, context_field_name, sentence_field_name)

    def create_text_field(self, sent_list):
        sent_list_filtered_by_words = [' '.join(tu.word_tokenize(sent)) for sent in sent_list]
        return '<BREAK>'.join(sent_list_filtered_by_words)
