import os

PROJECT_DIR_PATH = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(PROJECT_DIR_PATH)

DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
SAVE_PATH = os.path.join(PROJECT_ROOT, 'saved_models')
CACHE_PATH = os.path.join(PROJECT_ROOT, 'cache')
LOG_PATH = os.path.join(PROJECT_ROOT, 'logdir')

FULL_PUMBED_2018_PATH = os.path.join(os.path.expanduser('~'), 'pubmed_2018')
PUBMED_SHARDS = 50

FULL_PUMBED_2019_PATH = os.path.join(os.path.expanduser('~'), 'pubmed_2019_by_years')
FULL_PUMBED_2020_PATH = os.path.join(os.path.expanduser('~'), 'pubmed_2020_by_years')
DEFAULT_PUBMED_VERSION = 2019