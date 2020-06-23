# sort and frequency oriented  time series forecast submodule
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


def from_accumulated_frequency_to_forecast(local_accumulated_frequency_data, local_days_in_focus_frame,
                                           local_aff_forecast_horizon_days):
    # ---------------kernel----------------------------------
    nof_frequencies_registered = local_accumulated_frequency_data.shape[1]
    preprocess_structure = np.zeros(shape=(local_accumulated_frequency_data.shape[0],
                                           2 + nof_frequencies_registered
                                           + local_aff_forecast_horizon_days), dtype=np.dtype('float32'))
    preprocess_structure[:, 2: 2 + nof_frequencies_registered] = local_accumulated_frequency_data
    for local_time_serie in range(preprocess_structure.shape[0]):
        preprocess_structure[local_time_serie, 0] = \
            np.mean(preprocess_structure[local_time_serie, 2: 2 + nof_frequencies_registered])
        preprocess_structure[local_time_serie, 1] = local_time_serie
    # sort in base to mean unit_sales_accumulated_absolute_frequencies
    preprocess_structure = preprocess_structure.sort(axis=0)
    # preprocess_structure[:, 2 + nof_frequencies_registered:] = \
    #     neural_network.run(preprocess_structure[:, 2: 2 + nof_frequencies_registered], local_aff_forecast_horizon_days)
    # return sort to original time_serie raw data order
    # eliminate mean column
    preprocess_structure = preprocess_structure[:, 1:]
    preprocess_structure = preprocess_structure.sort(axis=0)
    # eliminate time_serie_index column and previous data
    local_y = preprocess_structure[:, 1 + nof_frequencies_registered:]
    return local_y
    # ---------------kernel----------------------------------


def make_forecast(local_array, local_mf_forecast_horizon_days, local_days_in_focus_frame):
    local_forecast = []
    days = np.array([day for day in range(local_days_in_focus_frame)])
    days = np.divide(days, np.amax(days))
    x_y_data = np.zeros(shape=(days.shape[0], 2), dtype=np.dtype('float32'))
    x_y_data[:, 0] = days
    for local_time_serie in range(local_array.shape[0]):
        x_y_data[:, 1] = local_array[local_time_serie, :]
        x = x_y_data[:, 0].reshape(-1, 1)
        y = x_y_data[:, 1].reshape(-1, )
        y_max = np.amax(y)
        y = np.divide(y, y_max * (y_max != 0) + 1 * (y_max == 0))
        regression = RANSACRegressor(base_estimator=ARDRegression(), min_samples=29, max_trials=2000, random_state=0,
                                     loss='squared_loss', residual_threshold=2.0).fit(x, y)
        score = regression.score(x, y)
        print('time_serie, score of RANdom SAmple Consensus algorithm', local_time_serie, score)
        forecast_days = np.add(days, local_mf_forecast_horizon_days)[-local_mf_forecast_horizon_days:].reshape(-1, 1)
        local_forecast_ts = regression.predict(forecast_days)
        local_forecast.append(local_forecast_ts)
    local_forecast = np.array(local_forecast)
    local_array_max = np.amax(local_array, axis=1)
    local_forecast = np.multiply(local_forecast, local_array_max.reshape(local_array_max.shape[0], 1))
    print('local_forecast shape:', local_forecast.shape)
    return local_forecast


class accumulated_frequency_distribution_based_engine:

    def accumulate_and_distribute(self, local_settings, local_raw_unit_sales, regression_technique):
        try:
            # this use absolute accumulated frequency
            print('starting time_series accumulated_frequency distribution forecast submodule')
            with open(''.join([local_settings['hyperparameters_path'],
                               'accumulated_frequency_model_hyperparameters.json'])) as local_ad_json_file:
                local_accumulate_and_distribute_hyperparameters = json.loads(local_ad_json_file.read())
                local_ad_json_file.close()
            local_days_in_focus = local_accumulate_and_distribute_hyperparameters['days_in_focus_frame']
            local_nof_time_series = local_raw_unit_sales.shape[0]
            local_forecast_horizon_days = local_settings['forecast_horizon_days']

            # ---------------kernel----------------------------------
            local_unit_sales_data = local_raw_unit_sales[:, -local_days_in_focus:]
            accumulated_frequency_array = \
                np.zeros(shape=(local_nof_time_series, local_days_in_focus),
                         dtype=np.dtype('float32'))
            for local_day in range(local_days_in_focus):
                if local_day != 0:
                    accumulated_frequency_array[:, local_day] = \
                        np.add(accumulated_frequency_array[:, local_day - 1], local_unit_sales_data[:, local_day])
                else:
                    accumulated_frequency_array[:, 0] = local_unit_sales_data[:, 0]
            if regression_technique == 'in_block_neural_network' and \
                    local_settings['repeat_training_in_block'] == 'True':
                acc_freq_forecast = from_accumulated_frequency_to_forecast(
                    accumulated_frequency_array[:, :local_days_in_focus],
                    local_days_in_focus, local_forecast_horizon_days)
            elif regression_technique == 'RANSACRegressor':
                acc_freq_forecast = make_forecast(accumulated_frequency_array[:, :local_days_in_focus],
                                                  local_forecast_horizon_days + 1, local_days_in_focus)
            else:
                print('regression technique indicated in hyperparameters was not understood'
                      '(accumulated_frequency_distribution based submodule)')
                return False, []

            # passing from predicted accumulated absolute frequencies to sales for day
            y_pred = np.zeros(shape=(local_nof_time_series, local_forecast_horizon_days), dtype=np.dtype('float32'))
            for day in range(local_forecast_horizon_days):
                next_acc_freq = acc_freq_forecast[:, day + 1: day + 2]
                # another options is use np.add(np.array(y_pred).sum(), -previous_acc_freq)
                y_pred[:, day: day + 1] = np.add(next_acc_freq, -acc_freq_forecast[:, day: day + 1])
            print('y_pred shape:', y_pred.shape)
            print(y_pred)
            # ---------------kernel----------------------------------

            print('accumulated_frequency_distribution based forecast done\n')
            print('accumulated_frequency_distribution submodule has finished')
        except Exception as submodule_error:
            print('accumulated_frequency_distribution based submodule_error: ', submodule_error)
            logger.info('error in accumulated_frequency_distribution time_series forecast')
            logger.error(str(submodule_error), exc_info=True)
            return False, []
        return True, y_pred
