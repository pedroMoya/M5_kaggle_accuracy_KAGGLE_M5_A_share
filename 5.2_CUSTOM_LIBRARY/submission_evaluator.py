# submission evaluator
import os
import logging
import logging.handlers as handlers
import json
import itertools as it
import pandas as pd
import numpy as np
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
tf.keras.backend.set_floatx('float32')
from tensorflow.keras import layers
from tensorflow.keras.experimental import PeepholeLSTMCell
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import losses, models
from tensorflow.keras import metrics
from tensorflow.keras import callbacks as cb
from tensorflow.keras import backend as kb
from sklearn.metrics import mean_squared_error

# open local settings
with open('./settings.json') as local_json_file:
    local_submodule_settings = json.loads(local_json_file.read())
    local_json_file.close()

# log setup
current_script_name = os.path.basename(__file__).split('.')[0]
log_path_filename = ''.join([local_submodule_settings['log_path'], current_script_name, '.log'])
logging.basicConfig(filename=log_path_filename, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logHandler = handlers.RotatingFileHandler(log_path_filename, maxBytes=10485760, backupCount=5)
logger.addHandler(logHandler)

# set random seed for reproducibility --> done in _2_train.py module
np.random.seed(42)
tf.random.set_seed(42)

# clear session
kb.clear_session()


class submission_tester:

    def evaluate_external_submit(self, local_settings, local_raw_unit_sales, local_mse=None):
        try:
            print('evaluation of submission (read and evaluate TESTsubmission.csv file in 9.3_OTHERS_INPUTS folder)')
            # open csv file
            external_submit = pd.read_csv(''.join([local_settings['others_inputs_path'], 'TESTsubmission.csv']))
            external_submit = external_submit.iloc[1:, 1:].values
            ground_truth_submit = pd.read_csv(''.join([local_settings['raw_data_path'], 'TESTsubmission.csv']))


        except Exception as submodule_error:
            print('internal submission evaluation submodule_error: ', submodule_error)
            logger.info('error in evaluation of external submit TESTsubmission.csv')
            logger.error(str(submodule_error), exc_info=True)
            return False

    def evaluate_internal_submit(self, local_forecasts, local_settings, local_raw_unit_sales, local_mse=None):
        try:
            print('evaluation of submission (corresponding to local script processes)')

        except Exception as submodule_error:
            print('internal (local) submission evaluation submodule_error: ', submodule_error)
            logger.info('error in evaluation of local submission')
            logger.error(str(submodule_error), exc_info=True)
            return False
        return True
