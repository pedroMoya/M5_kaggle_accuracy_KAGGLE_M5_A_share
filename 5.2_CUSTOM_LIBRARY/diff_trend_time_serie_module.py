# stochastic simulation time series module
import os
import logging
import logging.handlers as handlers
import json
import itertools as it
import pandas as pd
import numpy as np
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


def next_coefficients(local_time_serie_data, local_days_in_focus_frame,
                             local_forecast_horizon_days, local_nof_features_for_training):
    coefficients = 0

    # ---------------kernel----------------------------------

    # ---------------kernel----------------------------------
    return coefficients


def next_values(local_priming_block, local_diff_coefficients):
    values = 0

    # ---------------kernel----------------------------------

    # ---------------kernel----------------------------------
    return values


class difference_trends_insight:

    def run_diff_trends_ts_analyser(self, local_settings, local_raw_unit_sales,  local_time_series_not_improved):
        try:
            # opening diff_trends hyperparameters
            with open(''.join([local_settings['hyperparameters_path'],
                               'diff_trends_model_hyperparameters.json'])) \
                    as local_r_json_file:
                diff_trends_hyperparameters = json.loads(local_r_json_file.read())
                local_r_json_file.close()
            local_forecast_horizon_days = local_settings['forecast_horizon_days']
            diffs_array = np.add(local_raw_unit_sales[1:], -local_raw_unit_sales[:-1])
            days_in_block = diff_trends_hyperparameters['block_length']
            diff_coefficients = next_coefficients(diffs_array, days_in_block, diff_trends_hyperparameters)
            new_forecast = np.zeros((local_settings['number_of_time_series'], local_forecast_horizon_days))
            priming_block = local_raw_unit_sales[:, -days_in_block:]
            new_forecast = next_values(priming_block, diff_coefficients)
            # save results
            np.save(''.join([local_settings['others_outputs_path'],
                             'diff_pattern_based_forecasts']), new_forecast)
            np.savetxt(''.join([local_settings['others_outputs_path'], 'forecasts_diff_based_.csv']),
                       new_forecast, fmt='%10.15f', delimiter=',', newline='\n')
            print('diff_trend time_series submodule has finished')
        except Exception as submodule_error:
            print('time_series diff_trend time_series submodule_error: ', submodule_error)
            logger.info('error in diff_trend time_series sub_module')
            logger.error(str(submodule_error), exc_info=True)
            return False
        return True
