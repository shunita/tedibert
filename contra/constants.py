import os

PROJECT_DIR_PATH = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(PROJECT_DIR_PATH)

DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
SAVE_PATH = os.path.join(PROJECT_ROOT, 'saved_models')
CACHE_PATH = os.path.join(PROJECT_ROOT, 'cache')
LOG_PATH = os.path.join(PROJECT_ROOT, 'logdir')
EXP_PATH = os.path.join(PROJECT_ROOT, 'exp_results')


FULL_PUMBED_2018_PATH = os.path.join(os.path.expanduser('~'), 'pubmed_2018')
PUBMED_SHARDS = 50

FULL_PUMBED_2019_PATH = os.path.join(os.path.expanduser('~'), 'pubmed_2019_by_years')
FULL_PUMBED_2020_PATH = os.path.join(os.path.expanduser('~'), 'pubmed_2020_by_years')
FULL_PUMBED_2022_PATH = os.path.join(os.path.expanduser('~'), 'pubmed_2022_by_years')
DEFAULT_PUBMED_VERSION = 2020

# LOS_TEST_PATH = '/home/shunita/mimic3/custom_tasks/data/los_by_diag.csv'
# LOS_TEST_PATH_V2 = '/home/shunita/mimic3/custom_tasks/data/los_by_diag_v2.csv'
# LOS_TEST_PATH_V3 = '/home/shunita/mimic3/custom_tasks/data/los_by_diag_v3.csv'

# for upsampling females from GAN or SMOTE use the second or third line.
LOS_TEST_PATH_V4 = '/home/shunita/mimic3/custom_tasks/data/los_by_diag_v4.csv'
# LOS_TEST_PATH_V4 = 'data/los_by_diags_sampled_medgan.csv'
# LOS_TEST_PATH_V4 = 'data/los_by_diags_with_smote.csv'
# LOS_TEST_PATH_V4 = 'data/los_by_diags_10CV.csv'

# for upsampling females from GAN or SMOTE use the second or third line.
READMIT_TEST_PATH = '/home/shunita/mimic3/custom_tasks/data/stays_readmission_plus_measurements.csv'
# READMIT_TEST_PATH = 'data/readmission_by_diags_sampled_medgan_v2.csv'
# READMIT_TEST_PATH = 'data/readmission_by_diags_with_smote.csv'
# READMIT_TEST_PATH = 'data/readmission_by_diags_10CV.csv'

CLINICAL_NOTES_AND_DIAGS = os.path.expanduser('~/mimic3/clinical-notes-diagnosis-dl-nlp/local_data/clinical_notes_and_diags.csv')