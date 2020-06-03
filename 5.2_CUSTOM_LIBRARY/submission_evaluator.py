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
from tensorflow.keras.metrics import mean_absolute_percentage_error

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


# functions definitions


def test_forecasts(local_y_pred, local_y_truth):
    print('y_truth shape:', local_y_truth.shape, 'y_pred_shape:', type(local_y_truth))
    print('y_truth type:', local_y_pred.shape, 'y_pred type:', type(local_y_pred))
    # evaluation of models forecasts according to day-wise comparison
    # forecaster(y_truth) <=> y_pred
    print('\nmodels evaluation\nusing MEAN SQUARED ERROR, '
          'MODIFIED-MEAN ABSOLUTE PERCENTAGE ERROR and MEAN ABSOLUTE ERROR')
    print('{:^19s}{:^19s}'.format('time_serie', 'error_metric_MSE'))
    time_series_error_metrics = []
    for local_time_serie in range(local_y_truth.shape[0]):
        y_truth_ts = local_y_truth[local_time_serie, :]
        y_pred_ts = local_y_pred[local_time_serie, :]
        error_metric_mse = mean_squared_error(y_truth_ts, y_pred_ts)
        time_series_error_metrics.append([local_time_serie, error_metric_mse])
    print('model evaluation subprocess ended successfully')
    return np.array(time_series_error_metrics)


class submission_tester:

    def evaluate_external_submit(self, local_forecast_horizon_days, local_settings):
        try:
            print('evaluation of submission (read and evaluate TESTsubmission.csv file in 9.3_OTHERS_INPUTS folder)')
            # open csv files
            external_submit = pd.read_csv(''.join([local_settings['others_inputs_path'], 'TESTsubmission.csv']))
            external_submit = external_submit.iloc[:, :].values
            ground_truth_submit = pd.read_csv(''.join([local_settings['raw_data_path'], 'sales_train_evaluation.csv']))
            ground_truth_submit = ground_truth_submit.iloc[:, 6:].values
            y_pred = external_submit[:, -local_forecast_horizon_days:]
            y_truth = ground_truth_submit[:, -local_forecast_horizon_days:]
            print(y_pred[271, :])
            print(y_truth[271, :])
            print(y_pred[21013, :])
            print(y_truth[21013, :])
            forecasts_evaluation = test_forecasts(y_pred, y_truth)
            forecasts_evaluation = np.array(forecasts_evaluation)
            np.savetxt(''.join([local_settings['others_outputs_path'], 'external_submission_evaluation.csv']),
                       forecasts_evaluation, fmt='%10.15f', delimiter=',', newline='\n')
            print('evaluation of external submission file done, results saved to file')
            return True

        except Exception as submodule_error:
            print('external file submission evaluation submodule_error: ', submodule_error)
            logger.info('error in evaluation of external submit TESTsubmission.csv')
            logger.error(str(submodule_error), exc_info=True)
            return False

    def evaluate_internal_submit(self, local_forecast_horizon_days, local_forecasts, local_settings):
        try:
            print('evaluation of submission (corresponding to local script processes)')
            # open csv files
            internal_submit = local_forecasts
            ground_truth_submit = pd.read_csv(''.join([local_settings['raw_data_path'], 'sales_train_evaluation.csv']))
            ground_truth_submit = ground_truth_submit.iloc[:, 6:].values
            y_pred = internal_submit[:, -local_forecast_horizon_days:]
            y_truth = ground_truth_submit[:, -local_forecast_horizon_days:]
            forecasts_evaluation = test_forecasts(y_pred, y_truth)
            forecasts_evaluation = np.array(forecasts_evaluation)
            np.savetxt(''.join([local_settings['others_outputs_path'], 'internal_submission_evaluation.csv']),
                       forecasts_evaluation, fmt='%10.15f', delimiter=',', newline='\n')
            print('evaluation of internal submission file done, results saved to file')
            return True

        except Exception as submodule_error:
            print('internal (local) submission evaluation submodule_error: ', submodule_error)
            logger.info('error in evaluation of local submission')
            logger.error(str(submodule_error), exc_info=True)
            return False
        return True
