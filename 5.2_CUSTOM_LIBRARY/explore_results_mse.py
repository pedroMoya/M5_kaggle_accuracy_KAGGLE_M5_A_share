# saving forecasts and making forecast
import os
import sys
import logging
import logging.handlers as handlers
import json
import pandas as pd
import numpy as np

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
from save_forecast_and_make_submission import save_forecast_and_submission


class explore_results_and_generate_submission:

    def run(self, submission_name, local_ergs_settings):
        try:
            # first check the stage, if evaluation stage, this means that no MSE are available, warning about this
            if local_ergs_settings['competition_stage'] != 'submitting_after_June_1th_using_1913days':
                print('settings indicate that the final stage is now in progress')
                print('so there not available real MSE for comparison')
                print('the last saved data will be used and allow to continue..')
                print(''.join(['\x1b[0;2;41m',
                               'but be careful with this submission and consider other way to make the final submit',
                              '\x1b[0m']))

            # loading the forecasts
            first_model_forecast = np.load(''.join([local_ergs_settings['train_data_path'],
                                                    'first_model_forecast_data.npy']))
            second_model_forecast = np.load(''.join([local_ergs_settings['train_data_path'],
                                                    'second_model_forecast_data.npy']))
            lstm_model_forecast = np.load(''.join([local_ergs_settings['train_data_path'],
                                                   'LSTM_model_forecast_data.npy']))
            RANSAC_model_forecast = np.load(''.join([local_ergs_settings['train_data_path'],
                                                   'RANSAC_model_forecast_data.npy']))

            # loading the results
            first_model_result = np.load(''.join([local_ergs_settings['models_evaluation_path'],
                                                  'time_series_forecast_results_stochastic_simulation_mse.npy']))
            second_model_result = np.load(''.join([local_ergs_settings['models_evaluation_path'],
                                                   'time_series_second_model_forecast_mse.npy']))
            lstm_model_result = np.load(''.join([local_ergs_settings['models_evaluation_path'],
                                                 'LSTM_model_forecast_result_mse.npy']))
            RANSAC_model_result = np.load(''.join([local_ergs_settings['models_evaluation_path'],
                                                 'LSTM_model_forecast_result_mse.npy']))

            # -----------------------kernel------------------------------------------
            nof_ts = local_ergs_settings['number_of_time_series']
            local_forecast_horizon_days = local_ergs_settings['forecast_horizon_days']
            best_y_pred = np.zeros(shape=(nof_ts, local_forecast_horizon_days), dtype=np.dtype('float32'))
            count_best_first_model, count_best_second_model, count_best_lstm = 0, 0, 0
            for time_serie_index in range(nof_ts):
                first_model_mse = first_model_result[time_serie_index][2]
                second_model_mse = second_model_result[time_serie_index][2]
                lstm_model_mse = lstm_model_result[time_serie_index][1]
                if first_model_mse < second_model_mse and first_model_mse < lstm_model_mse:
                    best_y_pred[time_serie_index, :] = first_model_forecast[time_serie_index, :]
                    count_best_first_model += 1
                elif second_model_mse < first_model_mse and second_model_mse < lstm_model_mse:
                    best_y_pred[time_serie_index, :] = second_model_forecast[time_serie_index, :]
                    count_best_second_model += 1
                else:
                    best_y_pred[time_serie_index, :] = lstm_model_forecast[time_serie_index, :]
                    count_best_lstm += 1
            print('it was used ', count_best_first_model, ' ts forecasts from first model')
            print('it was used ', count_best_second_model, ' ts forecasts from second model')
            print('it was used ', count_best_lstm, ' ts forecasts from LSTM model')

            # saving best mse_based between different models forecast and submission
            store_and_submit_best_model_forecast = save_forecast_and_submission()
            mse_based_best_model_save_review = \
                store_and_submit_best_model_forecast.store_and_submit(submission_name, local_ergs_settings,
                                                                      best_y_pred)
            if mse_based_best_model_save_review:
                print('first_model forecast data and submission done')
            else:
                print('error at storing first model forecast data or submission')
        except Exception as submodule_error:
            print('explore_results and generate_submission submodule_error: ', submodule_error)
            logger.info('error in explore_results and generate_submission submodule')
            logger.error(str(submodule_error), exc_info=True)
            return False
        return True
