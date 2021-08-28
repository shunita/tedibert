import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
from datetime import datetime
from contra.constants import SAVE_PATH, LOG_PATH


def date_type(dt):
    return datetime.strptime(dt, '%Y/%m/%d')


parser = argparse.ArgumentParser(add_help=False)

# Data choices - years
parser.add_argument('--first_start_year', default=2010, type=int,
                    help='first year of the first range')
parser.add_argument('--first_end_year', default=2013, type=int,
                    help='last year of the first range')
parser.add_argument('--second_start_year', default=2018, type=int,
                    help='first year of the second range')
parser.add_argument('--second_end_year', default=2018, type=int,
                    help='last year of the second range')
parser.add_argument('--test_start_year', type=int,
                    help='first year of the test range')
parser.add_argument('--test_end_year', type=int,
                    help='last year of the test range')
parser.add_argument('--test_pairs_file', type=str,
                    help='a prepared test file with pairs of concepts for test.')
parser.add_argument('--abstract_pairs_file', type=str,
                    help='a data file with abstract pairs, matched by bow similarity.')

parser.add_argument('--start_year', default=2018, type=int,
                    help='start year for BERT pretraining')
parser.add_argument('--end_year', default=2018, type=int,
                    help='end year for BERT pretraining')

# Data choices - abstracts       
parser.add_argument('--abstract_weighting_mode', type=str, default='normal', 
                    help='how to use the abstracts when training the initial embedding.'
                    '\n"normal" - use all abstracts.'
                    '\n"subsample" - use the same number of abstracts from each year in the training.'
                    '\n"newer" - higher weight to abstracts from later years.')
parser.add_argument('--pubmed_version', type=int, default=2019, help='2019 or 2020')
parser.add_argument('--only_aact_data', dest='only_aact_data', action='store_true')

# Data choices - parsing
# parser.add_argument('--by_sentence', dest='by_sentence', action='store_true')
parser.add_argument('--serve_type', type=int, dest='serve_type',
                    help='0 - full abstract as text\n'
                    '1 - full abstract as BOW\n'
                    '2 - single sentence as text\n'
                    '3 - single sentence as BOW\n'
                    '4 - three sentences as text\n'
                    '5 - three sentences as BOW'
                    )
parser.add_argument('--overlap_sentences', dest='overlap_sentences', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')

# Bert
parser.add_argument('--num_frozen_layers', type=int, default=0, help='how many layers to freeze in the bert model')
parser.add_argument('--bert_pretrained_path', type=str, default='bert-base-cased',
                    help='path to a saved checkpoint to start bert training from.')
parser.add_argument('--bert_tokenizer', type=str, default='bert-base-cased', help='which tokenizer to use for bert')
parser.add_argument('--bert_save_prefix', type=str, help='prefix for saving bert checkpoints')

# Model parameters
parser.add_argument('--min_num_participants', default=1, type=int, help='minimum number of total participants')
parser.add_argument('--test_size', default=0.2, type=float, help='train_test_split ratio')
parser.add_argument('--name', type=str, help='a special name for the current model running')
parser.add_argument('--save_path', metavar='DIR', default=SAVE_PATH, type=str, help='path to save model')
parser.add_argument('--log_path', default=LOG_PATH, type=str, help='tensorboard log path')
parser.add_argument('--gpus', default=None, nargs='+', help='gpus parameter used for pytorch_lightning') # default='0'
parser.add_argument('--emb_algorithm', default='bert', type=str, 
                    help='inital embedding algorithm - either "bert", "w2v" or "doc2vec"(not yet fully supported).')
parser.add_argument('--initial_emb_size', default=768, type=int, 
                    help='initial embedding size. 768 for bert, or 300 for w2v, or other.')
parser.add_argument('--dim', type=int, default=128, metavar='d', dest='embedding_size',
                    help='the desired embedding size')
parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='learning_rate')
parser.add_argument('--max_epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch_size', default=32, type=int, help='batch size for training')

# Model Architecture
parser.add_argument('--bn', action='store_true',
                    help='Should batch normalization be used in discriminator/ratio prediction')
parser.add_argument('--activation', default='relu', 
                    help='Activation function to use in discriminator/ratio prediction. '
                         'Currently supported: "relu" or "swish".')
parser.add_argument('--regularize', default=0, type=float,
                    help='weight decay parameter for discriminator/ratio prediction.')
parser.add_argument('--agg_sentences', default='transformer',
                    help='How should sentence embeddings from bert be aggregated into an abstract embedding?'
                         'Supported options: "transformer" or "concat".')

parser.add_argument('--lmb_isnew', default=1, type=float, help='discriminator weight in the loss function')
parser.add_argument('--lmb_ratio', default=1, type=float, help='ratio prediction weight in the loss function')
parser.add_argument('--lmb_ref', default=1, type=float, help='diff from reference bert weight in the loss function')