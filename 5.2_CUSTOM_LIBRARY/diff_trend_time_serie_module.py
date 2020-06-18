# stochastic simulation of difference-trends in time_series module
import os
import sys
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

# load custom libraries
sys.path.insert(1, local_submodule_settings['custom_library_path'])
from mini_module_submission_generator import save_submission

# set random seed for reproducibility --> done in _2_train.py module
np.random.seed(42)
tf.random.set_seed(42)

# clear session
kb.clear_session()


def next_diff_coefficients(local_diffs_array, local_block_length, local_diff_trends_hyperparameters):
    # this core is where stochastic_simulation -again- takes its place
    # ---------------kernel----------------------------------
    nof_iterations = local_diff_trends_hyperparameters['nof_iterations']
    local_diffs_array = local_diffs_array[:, -local_block_length:]
    local_nof_ts = local_diffs_array.shape[0]
    random_occurrences_array = []
    random_event = np.zeros(shape=local_diffs_array.shape, dtype=np.dtype('float32'))

    # calculating probabilities
    prob_occurrence_augment = np.array([[np.divide((local_diffs_array > 0).sum(axis=1), local_block_length)]
                                        * local_block_length]).reshape(local_nof_ts, local_block_length)
    prob_occurrence_decrease = np.array([[np.divide((local_diffs_array < 0).sum(axis=1), local_block_length)]
                                         * local_block_length]).reshape(local_nof_ts, local_block_length)

    # median of diff>0 and diff<0
    median_array_pos, median_array_neg = [], []
    for local_ts_diff in range(local_nof_ts):
        array_pos = local_diffs_array[local_ts_diff, :][local_diffs_array[local_ts_diff, :] > 0]
        if array_pos.size != 0:
            median_array_pos.append(np.median(array_pos))
        else:
            median_array_pos.append(0.)
        array_neg = local_diffs_array[local_ts_diff, :][local_diffs_array[local_ts_diff, :] < 0]
        if array_neg.size != 0:
            median_array_neg.append(np.median(array_neg))
        else:
            median_array_neg.append(0.)
    median_array_pos, median_array_neg = np.array(median_array_pos), np.array(median_array_neg)
    median_array_neg.reshape(median_array_neg.shape[0])
    median_array_pos.reshape(median_array_pos.shape[0])
    random_occurrences_array = np.zeros(shape=(nof_iterations, local_nof_ts, local_block_length),
                                        dtype=np.dtype('float32'))

    # runs along time_series
    print('starting iterations in diff_trends stochastic simulation, nof iteration:', nof_iterations)
    for iteration in range(nof_iterations):
        for local_time_serie in range(local_nof_ts):
            # by now, simply
            # the order in assigns indicates the hierarchy, first lines lower hierarchies, last lines upper hierarchies
            # using a zeros array as base, low the compute load (the _no_change_ comments lines above an here below)
            # random_event[local_time_serie, random_occurrence_trigger_no_change[local_time_serie, :] >
            #              prob_occurrence_no_change[local_time_serie, :]] = 0
            # in this particular scenario, have tested mean an median, with better results (lower MSE) with median
            # generating random triggers
            random_occurrence_trigger_augment = np.random.rand(1, local_block_length)
            random_occurrence_trigger_decrease = np.random.rand(1, local_block_length)
            random_occurrence_trigger_no_change = np.random.rand(1, local_block_length)
            # normalization
            random_occurrence_trigger_total = np.add(random_occurrence_trigger_augment,
                                                     random_occurrence_trigger_decrease)
            random_occurrence_trigger_total = np.add(random_occurrence_trigger_total,
                                                     random_occurrence_trigger_no_change)
            random_occurrence_trigger_augment = np.divide(random_occurrence_trigger_augment,
                                                          random_occurrence_trigger_total)
            random_occurrence_trigger_decrease = np.divide(random_occurrence_trigger_decrease,
                                                           random_occurrence_trigger_total)
            # random_occurrence_trigger_no_change = np.divide(random_occurrence_trigger_no_change,
            #                                                 random_occurrence_trigger_total)
            # prob_occurrence_no_change = np.array([[np.divide((local_diffs_array == 0).sum(axis=1),
            #                                                  local_block_length)] *
            #                                       local_block_length]).reshape(local_nof_ts, local_block_length)

            # stochastic event realization
            random_event[local_time_serie, random_occurrence_trigger_decrease[0, :] >
                         prob_occurrence_decrease[0, :]] =\
                median_array_neg[local_time_serie].astype(np.dtype('float32'))
            random_event[local_time_serie, random_occurrence_trigger_augment[0, :] >
                         prob_occurrence_augment[0, :]] =\
                median_array_pos[local_time_serie].astype(np.dtype('float32'))
        random_occurrences_array[iteration, :, :] = random_event
        random_event[:, :] = 0.
        # this piece above can be recoded in another overlapping or/and weighting way
    print('random_occurrences array processes finished\n')

    # consolidating and calculating the necessary statistics
    diff_coefficients_median = np.median(random_occurrences_array, axis=(0, 2))
    diff_coefficients_std = np.std(random_occurrences_array, axis=(0, 2))
    # instantiating a normal distribution of the events, only positive values are considered
    diff_coefficients_list = []
    for local_time_serie in range(local_nof_ts):
        diff_coefficients = np.random.normal(diff_coefficients_median[local_time_serie],
                                             diff_coefficients_std[local_time_serie],
                                             local_diffs_array[local_time_serie, :].shape)\
            .clip(diff_coefficients_median[local_time_serie] - 2 * diff_coefficients_std[local_time_serie],
                  diff_coefficients_median[local_time_serie] + 2 * diff_coefficients_std[local_time_serie])
        diff_coefficients_list.append(diff_coefficients)
    diff_coefficients = np.array(diff_coefficients_list)
    return diff_coefficients
    # ---------------kernel----------------------------------


def next_values(local_priming_block, local_diff_coefficients, local_days_in_block,
                local_nv_forecast_horizon_days):
    # ---------------kernel----------------------------------
    local_nof_time_series = local_priming_block.shape[0]
    forecast_structure = np.zeros(shape=(local_nof_time_series, local_days_in_block),
                                  dtype=np.dtype('float32'))
    for outer_stride_idx in range(local_days_in_block - 1):
        for inner_stride_idx in range(outer_stride_idx, local_days_in_block):
            priming_sale = \
                local_priming_block[:, inner_stride_idx: inner_stride_idx + 1]
            local_array = np.add(
                 local_diff_coefficients[:, inner_stride_idx: inner_stride_idx + 1], priming_sale)
            forecast_structure[:, inner_stride_idx: inner_stride_idx + 1] = \
                np.add(forecast_structure[:, inner_stride_idx: inner_stride_idx + 1], local_array)
    # some explanations about this process....
    # each stair strides fill 28 days from his starting point, (first starting point day -28, finishing day 27)
    # (day 0: first forecast_horizon_day & day 27: last forecast_horizon_day)
    # STRUCTURE --> (DAY, NUMBER OF BLOCKS IN THIS DAY)
    # (-28, 1)-(-27, 2)..(-14, 15)....(-1, 28)------(0, 28)..(7, 28)-(14, 28)-(15, 28)..(26, 28)..(27, 28)
    #   ^pre-Forecast    ^middle-preF ^last-preF    ^firstForecast    ^middleForecast              ^last_Forecast
    #   ^first_block(1)                                                                           ^last_block(56)
    #  the last 28 blocks lengths are 28, 27, .... until 1 day length the last_block
    forecast_structure = forecast_structure[:, -local_nv_forecast_horizon_days:]
    diff_values = np.divide(forecast_structure, local_days_in_block).clip(0)
    print('subprocess next_diff_values successfully completed')
    return diff_values
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
            # subtract sales_day(n) - sales_day(n-1)
            diffs_array = np.add(local_raw_unit_sales[:, 1:], -local_raw_unit_sales[:, :-1]).astype(np.dtype('float32'))
            days_in_block = diff_trends_hyperparameters['block_length']
            # call to function that returns an array with  random generated coefficients for apply in diffs_arrays
            diff_coefficients = next_diff_coefficients(diffs_array, days_in_block, diff_trends_hyperparameters)
            priming_block = local_raw_unit_sales[:, -days_in_block:].astype(np.dtype('float32'))
            new_forecast = next_values(priming_block, diff_coefficients, days_in_block,
                                       local_forecast_horizon_days)

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
        # saving submission as diff_trends_forecasts.csv
        save_submission_diff_trends_stochastic_model = save_submission()
        save_submission_review = save_submission_diff_trends_stochastic_model.save('diff_trends_model_forecasts.csv',
                                                                                   new_forecast, local_settings)
        if save_submission_review:
            print('submission of diff_trends stochastic model to csv successful')
        else:
            print('error at submission of diff_trends stochastic model to csv')
        return new_forecast
