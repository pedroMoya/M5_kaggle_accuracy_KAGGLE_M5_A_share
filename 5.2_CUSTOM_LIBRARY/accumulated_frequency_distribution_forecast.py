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

# importing custom libraries sub-dependencies
sys.path.insert(1, local_submodule_settings['custom_library_path'])
from in_block_neural_network_engine import in_block_neural_network

# set random seed for reproducibility --> done in _2_train.py module
np.random.seed(42)
tf.random.set_seed(42)

# clear session
kb.clear_session()


# functions definitions


def predict_accumulated_frequencies(local_acc_freq_data, local_nof_acc_frequencies, local_paf_settings,
                                    local_in_block_forecaster):
    print('data_presorted_acc_freq in_block model compiled, weights loaded\n')
    # running model to make predictions
    print('making predictions of acc_freq with the model trained..')
    # reshaping to correct input_shape
    local_acc_freq_data = local_acc_freq_data.reshape((1, local_acc_freq_data.shape[1], local_acc_freq_data.shape[0]))
    local_x_input = local_acc_freq_data[:, -local_nof_acc_frequencies:, :]
    # making the predictions
    local_y_pred_normalized = local_in_block_forecaster.predict(local_x_input)
    # reshaping output
    local_y_pred = local_y_pred_normalized.reshape((local_y_pred_normalized.shape[2], local_y_pred_normalized.shape[1]))
    print('prediction of accumulated_frequencies function finish with success')
    return True, local_y_pred


def forecast_from_saved_nn_model(local_preprocess_structure, local_fourth_model_forecaster, local_fsnn_settings):
    # making predictions
    local_forecast_horizon_days = local_fsnn_settings['forecast_horizon_days']
    nof_acc_frequencies = local_forecast_horizon_days + 1
    predict_acc_frequencies_review, predict_freq_array = predict_accumulated_frequencies(
        local_preprocess_structure, nof_acc_frequencies, local_fsnn_settings, local_fourth_model_forecaster)
    if predict_acc_frequencies_review:
        print('success at making predictions of accumulated frequencies')
    else:
        print('error at making predictions of accumulated frequencies')
        return False
    print('freq_accumulated based neural_network predictions has end (model previously saved)')
    print(predict_freq_array.shape)
    print(predict_freq_array)
    return predict_freq_array


def from_accumulated_frequency_to_forecast(local_accumulated_frequency_data, local_faff_settings):
    # ---------------kernel----------------------------------
    print('input:')
    print(local_accumulated_frequency_data)
    nof_frequencies_registered = local_accumulated_frequency_data.shape[1]
    nof_forecast_freq = local_faff_settings['forecast_horizon_days'] + 1
    preprocess_structure = np.zeros(shape=(local_accumulated_frequency_data.shape[0],
                                           local_accumulated_frequency_data.shape[1] + 2 + nof_forecast_freq),
                                    dtype=np.dtype('float32'))
    preprocess_structure[:, 2: 2 + nof_frequencies_registered] = local_accumulated_frequency_data
    for local_time_serie in range(preprocess_structure.shape[0]):
        preprocess_structure[local_time_serie, 0] = \
            np.mean(preprocess_structure[local_time_serie, 2: 2 + nof_frequencies_registered])
        preprocess_structure[local_time_serie, 1] = local_time_serie

    # simple normalization
    local_max_array = np.amax(preprocess_structure, axis=1)
    local_max_array[local_max_array == 0] = 1
    local_max_array = local_max_array.reshape(local_max_array.shape[0], 1)
    preprocess_structure = np.divide(preprocess_structure, local_max_array)

    # sort in base to mean unit_sales_accumulated_absolute_frequencies
    preprocess_structure = preprocess_structure[preprocess_structure[:, 0].argsort()]

    # setting repeat_training_in_block True is of higher priority level than save_fresh_forecast_from_fourth_model
    if local_faff_settings['save_fresh_forecast_from_fourth_model'] == 'True' and \
            local_faff_settings['repeat_training_in_block'] == 'False':
        # load previously saved model
        fourth_model_forecaster = models.load_model(''.join([local_faff_settings['models_path'],
                                                             '_acc_freq_in_block_nn_model_.h5']))
        fourth_model_forecaster.load_weights(''.join([local_faff_settings['models_path'],
                                                      '_weights_acc_freq_in_block_nn_model_.h5']))
        preprocess_structure[:, 2 + nof_frequencies_registered:] = \
            forecast_from_saved_nn_model(preprocess_structure[:, 2: 2 + nof_frequencies_registered],
                                         fourth_model_forecaster, local_faff_settings)
    else:
        preprocess_structure[:, 2 + nof_frequencies_registered:] = \
            in_block_neural_network.train_nn_model(preprocess_structure[:, 2: 2 + nof_frequencies_registered])

    # return sort to original time_serie raw data order
    # eliminate mean column
    preprocess_structure = preprocess_structure[:, 1:]
    preprocess_structure = preprocess_structure[preprocess_structure[:, 0].argsort()]

    # simple denormalization
    preprocess_structure = np.multiply(preprocess_structure, local_max_array)

    # eliminate time_serie_index column, previous data, and clip lower values to zero
    local_y = preprocess_structure[:, 1 + nof_frequencies_registered:].clip(0)

    # this output is still considered a preprocess, according that works with acc_frequencies and not unit_sales_by_day
    return local_y
    # ---------------kernel----------------------------------


def make_forecast(local_array, local_mf_forecast_horizon_days, local_days_in_focus_frame):
    local_forecast = []
    # simple normalization
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
    # simple denormalization
    local_array_max = np.amax(local_array, axis=1)
    local_forecast = np.multiply(local_forecast, local_array_max.reshape(local_array_max.shape[0], 1))
    print('local_forecast shape:', local_forecast.shape)
    return local_forecast


# why and how?
# in a few lines: 1) accumulated frequencies deals correctly with zeros
# 2) normalization (very simple) pass from absolute acc_frequencies to relative acc_freq
# 3) RANSACRegression accounts linear components
# 4) ANNs deals with no-linear components
# 5) previous stochastic simulations take in consideration the underlying random process (other approach: chaotic..)
# 6) amplification, two focus_days (short and large length), was discovered with exploration metaheuristic
# 7) the sort by row_mean for in_block NN training allows fast and better predictions, because structure is important
# 8) the all of above indicates in overall one think (in my opinion): mastering patterns rules this domain


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
            if regression_technique == 'RANSACRegressor':
                accumulated_frequency_array = \
                    np.zeros(shape=(local_nof_time_series, local_days_in_focus), dtype=np.dtype('float32'))
                local_unit_sales_data = local_raw_unit_sales[:, -local_days_in_focus:]
            elif regression_technique == 'in_block_neural_network':
                with open(''.join([local_settings['hyperparameters_path'],
                                   'freq_acc_in_block_model_hyperparameters.json'])) as local_js_file:
                    local_ibnn_hyperparameters = json.loads(local_js_file.read())
                    local_js_file.close()
                local_days_in_focus = local_ibnn_hyperparameters['days_in_focus_frame']
                # local_unit_sales_data = local_raw_unit_sales[:, -local_days_in_focus:]
                local_unit_sales_data = local_raw_unit_sales
                accumulated_frequency_array = \
                    np.zeros(shape=local_unit_sales_data.shape, dtype=np.dtype('float32'))
            else:
                print('regression technique do not identified')
                return False, []
            if regression_technique == 'in_block_neural_network':
                for local_day in range(accumulated_frequency_array.shape[1]):
                    if local_day != 0:
                        accumulated_frequency_array[:, local_day] = \
                            np.add(accumulated_frequency_array[:, local_day - 1], local_unit_sales_data[:, local_day])
                    else:
                        accumulated_frequency_array[:, 0] = local_unit_sales_data[:, 0]
                acc_freq_forecast = from_accumulated_frequency_to_forecast(accumulated_frequency_array, local_settings)
            elif regression_technique == 'RANSACRegressor' and \
                    local_accumulate_and_distribute_hyperparameters['repeat_RANSAC_regression'] == 'True':
                for local_day in range(local_days_in_focus):
                    if local_day != 0:
                        accumulated_frequency_array[:, local_day] = \
                            np.add(accumulated_frequency_array[:, local_day - 1], local_unit_sales_data[:, local_day])
                    else:
                        accumulated_frequency_array[:, 0] = local_unit_sales_data[:, 0]
                acc_freq_forecast = make_forecast(accumulated_frequency_array[:, :local_days_in_focus],
                                                  local_forecast_horizon_days + 1, local_days_in_focus)
            elif local_accumulate_and_distribute_hyperparameters['repeat_RANSAC_regression'] == 'False' or\
                    local_settings['repeat_training_in_block'] == 'False':
                print('not repeating training according settings specifications')
                if regression_technique == 'in_block_neural_network' and \
                        local_settings['save_fresh_forecast_from_fourth_model'] == 'False':
                    y_pred = np.load(''.join([local_settings['train_data_path'], 'fourth_model_forecast_data.npy']))
                    y_pred = y_pred[:local_nof_time_series, :]
                elif regression_technique == 'RANSACRegressor':
                    y_pred = np.load(''.join([local_settings['train_data_path'], 'third_model_forecast_data.npy']))
                    y_pred = y_pred[:local_nof_time_series, :]
                else:
                    print('error: it was not possible to load previous training because de regression technique '
                          'was not in settings')
                    return False, []
                return True, y_pred
            else:
                print('regression technique indicated in hyperparameters was not understood'
                      '(accumulated_frequency_distribution based submodule)')
                return False, []

            # passing from predicted accumulated absolute frequencies to sales for day
            y_pred = np.zeros(shape=(local_nof_time_series, local_forecast_horizon_days), dtype=np.dtype('float32'))
            print(acc_freq_forecast.shape)
            for day in range(local_forecast_horizon_days):
                next_acc_freq = acc_freq_forecast[:, day + 1: day + 2]
                next_acc_freq = next_acc_freq.reshape(next_acc_freq.shape[0], 1)
                y_pred[:, day: day + 1] = np.add(next_acc_freq, -acc_freq_forecast[:, day: day + 1]).clip(0)
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
