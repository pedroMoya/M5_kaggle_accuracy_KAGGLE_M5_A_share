# use accumulated_freq approach and nearest_neighbor regression for making forecasts
import os
import sys
import logging
import logging.handlers as handlers
import json
import itertools as it
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import neighbors

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
from stochastic_model_obtain_results import stochastic_simulation_results_analysis
from save_forecast_and_make_submission import save_forecast_and_submission

# functions definitions


def acc_freq_nearest_neighbor_regressor(local_afnnr_settings, local_afnnr_hyperparameters, local_acc_freq_array):
    try:
        print('training acc_freq approach nearest_neighbor_regression model')
        nof_time_series = local_afnnr_settings['number_of_time_series']
        nof_days = local_afnnr_hyperparameters['days_in_focus_frame']
        local_afnnr_forecast_horizon_days = local_afnnr_settings['forecast_horizon_days'] + 1
        local_y_pred = np.zeros(shape=(nof_time_series, local_afnnr_forecast_horizon_days),
                                dtype=np.dtype('float32'))
        # applying nearest neighbor regression
        n_neighbors = local_afnnr_hyperparameters['n_neighbors']
        nof_samples = local_afnnr_hyperparameters['nof_samples']
        for local_time_serie in range(nof_time_series):
            # creating training data
            local_x_train, local_y_train = [], []
            for sample in range(0, nof_samples):
                local_x_train.append(
                    local_acc_freq_array[local_time_serie: local_time_serie + 1,
                    -sample - 1 - local_afnnr_forecast_horizon_days + nof_days: -sample - 1 + nof_days])
                local_y_train.append(local_acc_freq_array[local_time_serie: local_time_serie + 1,
                                     -sample - local_afnnr_forecast_horizon_days + nof_days:
                                     -sample + nof_days])
            local_x_train = np.array(local_x_train)
            local_x_train = local_x_train.reshape(local_x_train.shape[0], local_x_train.shape[2])
            local_y_train = np.array(local_y_train)
            local_y_train = local_y_train.reshape(local_y_train.shape[0], local_y_train.shape[2])
            local_x_input = local_y_train[-1].reshape(1, -1)
            # regression based in nearest_neighbor
            for i, weights in enumerate(['uniform', 'distance']):
                knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
                local_y_output = knn.fit(local_x_train, local_y_train).predict(local_x_input)
                local_y_pred[local_time_serie, :] = local_y_output
    except Exception as acc_freq_nearest_neighbor_error:
        print('acc_freq nearest neighbor regression function error: ',
              acc_freq_nearest_neighbor_error)
        logger.info('error in acc_freq nearest neighbor regression function')
        logger.error(str(acc_freq_nearest_neighbor_error), exc_info=True)
        return False
    return local_y_pred


# classes definitions


class generate_forecast_and_calculate_mse:

    def execute(self, local_settings, local_raw_unit_sales, local_raw_unit_sales_ground_truth):
        try:
            print('executing acc_freq and nearest_neighbor regression (ninth model)')
            # opening hyperparameters
            with open(''.join([local_settings['hyperparameters_path'],
                               'acc_freq_and_nearest_neighbor_model_hyperparameters.json'])) \
                    as local_r_json_file:
                local_grc_model_hyperparameters = json.loads(local_r_json_file.read())
                local_r_json_file.close()
            local_forecast_name = 'ninth_model_acc_freq_and_nearest_neighbor_regression_forecast'
            local_nof_time_series = local_settings['number_of_time_series']
            local_forecast_horizon_days = local_settings['forecast_horizon_days']
            local_forecasts = np.zeros(shape=(local_nof_time_series * 2, local_forecast_horizon_days),
                                       dtype=np.dtype('float32'))

            # generate accumulated_frequencies (transform)
            local_days_in_focus = local_grc_model_hyperparameters['days_in_focus_frame']
            accumulated_frequency_array = \
                np.zeros(shape=(local_nof_time_series, local_days_in_focus), dtype=np.dtype('float32'))
            local_unit_sales_data = local_raw_unit_sales[:, -local_days_in_focus:]
            for local_day in range(local_days_in_focus):
                if local_day != 0:
                    accumulated_frequency_array[:, local_day] = \
                        np.add(accumulated_frequency_array[:, local_day - 1], local_unit_sales_data[:, local_day])
                else:
                    accumulated_frequency_array[:, 0] = local_unit_sales_data[:, 0]

            # apply regression to acc_freq
            acc_freq_and_nearest_neighbor_forecast_aggregated = \
                acc_freq_nearest_neighbor_regressor(local_settings,  local_grc_model_hyperparameters,
                                                    accumulated_frequency_array)

            # de-transform to unit_sales forecasts
            # passing from predicted accumulated absolute frequencies to sales for day
            y_pred = np.zeros(shape=(local_nof_time_series, local_forecast_horizon_days), dtype=np.dtype('float32'))
            print(acc_freq_and_nearest_neighbor_forecast_aggregated.shape)
            for day in range(local_forecast_horizon_days):
                next_acc_freq = acc_freq_and_nearest_neighbor_forecast_aggregated[:, day + 1: day + 2]
                next_acc_freq = next_acc_freq.reshape(next_acc_freq.shape[0], 1)
                y_pred[:, day: day + 1] = \
                    np.add(next_acc_freq, -acc_freq_and_nearest_neighbor_forecast_aggregated[:, day: day + 1]).clip(0)
            print('y_pred shape:', y_pred.shape)
            print(y_pred)

            # generating correct best fixed forecasts
            local_forecasts[:local_nof_time_series, :] = y_pred
            # dealing with Validation stage or Evaluation stage
            if local_settings['competition_stage'] == 'submitting_after_June_1th_using_1941days':
                local_forecasts[:local_nof_time_series, :] = local_raw_unit_sales[:, -local_forecast_horizon_days:]
                local_forecasts[local_nof_time_series:, :] = y_pred

            # save forecast and submission
            store_and_submit_ninth_model_forecast = save_forecast_and_submission()
            ninth_model_save_review = \
                store_and_submit_ninth_model_forecast.store_and_submit(''.join([local_forecast_name,'_data']),
                                                                       local_settings,
                                                                       y_pred)
            if ninth_model_save_review:
                print('acc_freq nearest neighbor regression generation successfully completed')
            else:
                print('error at acc_freq nearest neighbor regression forecast generation')

            # calculate mse and save results
            calculate_mse = stochastic_simulation_results_analysis()
            calculate_mse_time_series_not_improved = \
                calculate_mse.evaluate_stochastic_simulation(local_settings, local_grc_model_hyperparameters,
                                                             local_raw_unit_sales, local_raw_unit_sales_ground_truth,
                                                             local_forecast_name)
            print('number of ts not improved with acc_freq nearest neighbor regression ',
                  len(calculate_mse_time_series_not_improved))

        except Exception as submodule_error:
            print('acc_freq nearest neighbor regression forecast generation submodule_error: ', submodule_error)
            logger.info('error in acc_freq nearest neighbor regression generation submodule')
            logger.error(str(submodule_error), exc_info=True)
            return False
        return True
