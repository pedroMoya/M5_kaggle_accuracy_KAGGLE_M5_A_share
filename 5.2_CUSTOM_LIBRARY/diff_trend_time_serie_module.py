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


def next_diff_coefficients(local_diffs_array, local_days_in_focus_frame, local_diff_trends_hyperparameters):
    # this core is where stochastic_simulation again takes its place
    # ---------------kernel----------------------------------
    nof_iterations = local_diff_trends_hyperparameters['nof_iterations']
    local_block_length = local_diff_trends_hyperparameters['block_length']
    local_diffs_array = local_diffs_array[:, -local_days_in_focus_frame:]
    local_nof_ts = local_diffs_array.shape[0]
    random_occurrence_array = []
    for iteration in range(nof_iterations):
        # generating random triggers
        random_occurrence_trigger_augment = np.random.rand(local_nof_ts, local_block_length)
        random_occurrence_trigger_decrease = np.random.rand(local_nof_ts, local_block_length)
        random_occurrence_trigger_no_change = np.random.rand(local_nof_ts, local_block_length)
        # calculating probabilities
        prob_occurrence_augment = np.array([[np.divide((local_diffs_array > 0).sum(axis=1), local_days_in_focus_frame)]
                                           * local_days_in_focus_frame]).reshape(local_nof_ts, local_block_length)
        prob_occurrence_decrease = np.array([[np.divide((local_diffs_array < 0).sum(axis=1), local_days_in_focus_frame)]
                                             * local_days_in_focus_frame]).reshape(local_nof_ts, local_block_length)
        prob_occurrence_no_change = np.array([[np.divide((local_diffs_array == 0).sum(axis=1),
                                                         local_days_in_focus_frame)] *
                                              local_days_in_focus_frame]).reshape(local_nof_ts, local_block_length)
        # stochastic event realization
        random_event = local_diffs_array
        for local_time_serie in range(local_nof_ts):
            random_event[local_time_serie, random_occurrence_trigger_augment[local_time_serie, :] >
                         prob_occurrence_augment[local_time_serie, :]] = \
                np.median(local_diffs_array[local_time_serie, :] > 0, axis=0)
            random_event[local_time_serie, random_occurrence_trigger_decrease[local_time_serie, :] >
                         prob_occurrence_decrease[local_time_serie, :]] = \
                np.median(local_diffs_array[local_time_serie, :] < 0, axis=0)
            random_event[local_time_serie, random_occurrence_trigger_no_change[local_time_serie, :] >
                         prob_occurrence_no_change[local_time_serie, :]] = 0
            random_event[local_time_serie, random_occurrence_trigger_no_change[local_time_serie, :] >
                         prob_occurrence_no_change[local_time_serie, :]] = 0
        random_occurrence_array.append(random_event)
    random_occurrence_array = np.array(random_occurrence_array)
    diff_coefficients = np.mean(random_occurrence_array, axis=0)
    return diff_coefficients
    # ---------------kernel----------------------------------


def next_values(local_priming_block, local_diff_coefficients, local_days_in_block, local_nv_forecast_horizon_days):
    # ---------------kernel----------------------------------
    local_nof_time_series = local_priming_block.shape[0]
    forecast_triangle = np.zeros(shape=(local_nof_time_series, local_days_in_block, local_days_in_block))
    for outer_primer_idx in range(local_days_in_block):
        priming_sale = local_priming_block[:, outer_primer_idx]
        for inner_stride_idx in range(local_days_in_block):
            forecast_triangle[:, inner_stride_idx, outer_primer_idx] = np.add(
                priming_sale, local_diff_coefficients[:, inner_stride_idx])
            priming_sale = np.add(priming_sale, local_diff_coefficients[:, inner_stride_idx])
    new_values = np.zeros(shape=(local_nof_time_series, local_days_in_block))
    sum_values = np.sum(forecast_triangle, axis=2)
    for stair_in_triangle in range(local_nv_forecast_horizon_days):
        new_values[:, stair_in_triangle: stair_in_triangle + 1] = \
            np.divide(sum_values, local_nv_forecast_horizon_days - stair_in_triangle)
    return new_values
    # ---------------kernel----------------------------------


class difference_trends_insight:

    def run_diff_trends_ts_analyser(self, local_settings, local_raw_unit_sales):
        try:
            # opening diff_trends hyperparameters
            with open(''.join([local_settings['hyperparameters_path'],
                               'diff_trends_model_hyperparameters.json'])) \
                    as local_r_json_file:
                diff_trends_hyperparameters = json.loads(local_r_json_file.read())
                local_r_json_file.close()
            local_forecast_horizon_days = local_settings['forecast_horizon_days']
            diffs_array = np.add(local_raw_unit_sales[:, 1:], -local_raw_unit_sales[:, :-1])
            days_in_block = diff_trends_hyperparameters['block_length']
            diff_coefficients = next_diff_coefficients(diffs_array, days_in_block, diff_trends_hyperparameters)
            priming_block = local_raw_unit_sales[:, -days_in_block:]
            new_forecast = next_values(priming_block, diff_coefficients, days_in_block, local_forecast_horizon_days)

            # save results
            np.save(''.join([local_settings['others_outputs_path'],
                             'diff_pattern_based_forecasts']), new_forecast)
            np.savetxt(''.join([local_settings['others_outputs_path'], 'forecasts_diff_based_.csv']),
                       new_forecast, fmt='%10.15f', delimiter=',', newline='\n')
            print('diff_trend time_series submodule has finished, successfully (called by _2_train_module)')
        except Exception as submodule_error:
            print('time_series diff_trend time_series submodule_error: ', submodule_error)
            logger.info('error in diff_trend time_series sub_module')
            logger.error(str(submodule_error), exc_info=True)
            return "error"
        return new_forecast
