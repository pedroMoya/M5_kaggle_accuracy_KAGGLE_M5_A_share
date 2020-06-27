# convert acc_frequencies forecast to unit_sales_forecast (from 29 to 28 data)
import os
import sys
import logging
import logging.handlers as handlers
import json
import itertools as it
import pandas as pd
import numpy as np
from sklearn.linear_model import RANSACRegressor, ARDRegression
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import backend as kb

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


class make_acc_freq_to_unit_sales_forecast:

    def make_forecast(self, local_settings, time_series_trained):
        try:
            # data stored are accumulated_absolute_frequencies
            print('starting time_series accumulated_frequency individual time_serie make forecast submodule')
            acc_freq_forecast = np.load(''.join([local_settings['train_data_path'],
                                                 'point_forecast_NN_from_acc_freq_training.npy']))
            local_nof_time_series = acc_freq_forecast.shape[0]
            local_forecast_horizon_days = local_settings['forecast_horizon_days']

            # passing from predicted accumulated absolute frequencies to sales for day
            y_pred = np.zeros(shape=(local_nof_time_series, local_forecast_horizon_days), dtype=np.dtype('float32'))
            for day in range(local_forecast_horizon_days):
                next_acc_freq = acc_freq_forecast[:, day + 1: day + 2]
                next_acc_freq = next_acc_freq.reshape(next_acc_freq.shape[0], 1)
                y_pred[:, day: day + 1] = np.add(next_acc_freq, -acc_freq_forecast[:, day: day + 1]).clip(0)
            print('y_pred shape:', y_pred.shape)

            # building forecast output, fill with zeros time_series not processed
            nof_time_series = local_settings['number_of_time_series']
            nof_days = y_pred.shape[1]
            local_forecasts = np.zeros(shape=(nof_time_series * 2, nof_days), dtype=np.dtype('float32'))
            local_forecasts[:nof_time_series, :] = y_pred

            # dealing with Validation stage or Evaluation stage
            if local_settings['competition_stage'] == 'submitting_after_June_1th_using_1941days':
                local_raw_data_filename = 'sales_train_evaluation.csv'
                local_raw_data_sales = pd.read_csv(''.join([local_settings['raw_data_path'], local_raw_data_filename]))
                local_raw_unit_sales = local_raw_data_sales.iloc[:, 6:].values
                local_forecasts[:nof_time_series, :] = local_raw_unit_sales[:, -local_forecast_horizon_days:]
                local_forecasts[nof_time_series:, :] = y_pred

            print('accumulated_frequency individual ts based based forecast done\n')
            print('accumulated_frequency individual ts based submodule has finished')
        except Exception as submodule_error:
            print('accumulated_frequency individual ts based submodule_error: ', submodule_error)
            logger.info('error in accumulated_frequency individual based time_series forecast')
            logger.error(str(submodule_error), exc_info=True)
            return False, []
        return True, y_pred
