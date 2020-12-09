import argparse
from datetime import datetime

from contra.constants import SAVE_PATH, LOG_PATH


parser = argparse.ArgumentParser(add_help=False)

date_type = lambda dt: datetime.strptime(dt, '%Y/%m/%d')
parser.add_argument('--pivot_datetime', default="2016/01/01", type=date_type,
                    help='pivot year for new/old split')
parser.add_argument('--start_year', default=2018, type=int,
                    help='start year for BERT pretraining')
parser.add_argument('--end_year', default=2018, type=int,
                    help='end year for BERT pretraining')
parser.add_argument('--min_num_participants', default=1, type=int, help='minimum number of total participants')
parser.add_argument('--train_test_split', default=0.8, type=float, help='train_test_split ratio')

parser.add_argument('--name', type=str, help='a special name for the current model running')
parser.add_argument('--save_path', metavar='DIR', default=SAVE_PATH, type=str, help='path to save model')
parser.add_argument('--log_path', default=LOG_PATH, type=str, help='tensorboard log path')
parser.add_argument('--gpus', default='0', help='gpus parameter used for pytorch_lightning') # default='0'

parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='learning_rate')
parser.add_argument('--dim', type=int, default=128, metavar='d', dest='embedding_size',
                    help='the desired embedding size')
parser.add_argument('--max_epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lmb_isnew', default=1, type=int, help='lambda for the weighting of the discriminator')
parser.add_argument('--lmb_ratio', default=1, type=int, help='lambda for the weighting of the discriminator')
