# analyzing each result (validation stage) and selecting the best,
# if there no best (based in threshold) then applies a smart_reshift based in historic sales
import os
import sys
import datetime
import logging
import logging.handlers as handlers
import json
import itertools as it
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
from save_forecast_and_make_submission import save_forecast_and_submission
from stochastic_model_obtain_results import stochastic_simulation_results_analysis

# functions definitions


def make_smart_reshift(local_array, local_md_forecast_horizon_days):
    local_md_days_in_historical_data = len(local_array)
    local_array_forecast = np.zeros(shape=(1, local_md_forecast_horizon_days), dtype=np.dtype('float32'))


    return local_array_forecast


# classes definitions


class explore_results_focused_reshift_and_generate_submission:

    def run(self, submission_name, local_ergs_settings):
        try:
            print('\nstarting the smart mse_based and reshift forecast selection approach')
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
            third_model_forecast = np.load(''.join([local_ergs_settings['train_data_path'],
                                                   'third_model_forecast_data.npy']))
            fourth_model_forecast = np.load(''.join([local_ergs_settings['train_data_path'],
                                                     'fourth_model_forecast_data.npy']))
            # this forecast has the shape=30490, 28
            fifth_model_forecast_30490_28 = np.load(''.join([local_ergs_settings['train_data_path'],
                                                             'fifth_model_forecast_data.npy']))
            fifth_model_forecast = np.zeros(shape=(60980, 28), dtype=np.dtype('float32'))
            fifth_model_forecast[0: 30490, :] = fifth_model_forecast_30490_28
            sixth_model_forecast = np.load(''.join([local_ergs_settings['train_data_path'],
                                                    'sixth_model_forecast_data.npy']))
            seventh_model_forecast = np.load(''.join([local_ergs_settings['train_data_path'],
                                                     'seventh_model_forecast_data.npy']))
            eighth_model_forecast = np.load(''.join([local_ergs_settings['train_data_path'],
                                                    'eighth_model_nearest_neighbor_forecast_data.npy']))
            ninth_model_forecast = np.load(''.join([local_ergs_settings['train_data_path'],
                                                    'ninth_model_random_average_simulation_forecast_data.npy']))
            eleventh_model_forecast = np.load(''.join([local_ergs_settings['train_data_path'],
                                                       'eleventh_model_forecast_data.npy']))
            
            # day by day comparison
            with open(''.join([local_ergs_settings['hyperparameters_path'],
                               'organic_in_block_time_serie_based_model_hyperparameters.json'])) \
                    as local_r_json_file:
                local_model_ergs_hyperparameters = json.loads(local_r_json_file.read())
                local_r_json_file.close()
            nof_ts = local_ergs_settings['number_of_time_series']
            local_forecast_horizon_days = local_ergs_settings['forecast_horizon_days']
            local_threshold = local_model_ergs_hyperparameters['stochastic_simulation_poor_result_threshold']
            best_lower_error_ts_y_pred = np.zeros(shape=(nof_ts, local_forecast_horizon_days),
                                                  dtype=np.dtype('float32'))
            ts_model_mse = []

            # accessing ground_truth data and rechecking stage of competition
            local_ergs_raw_data_filename = 'sales_train_evaluation.csv'
            local_ergs_raw_unit_sales = pd.read_csv(''.join([local_ergs_settings['raw_data_path'],
                                                             local_ergs_raw_data_filename]))
            print('raw sales data accessed (best_mse and smart_reshift approach model)')
            # extract data and check  dimensions
            local_ergs_raw_unit_sales = local_ergs_raw_unit_sales.iloc[:, 6:].values
            local_max_selling_time = np.shape(local_ergs_raw_unit_sales)[1]
            local_settings_max_selling_time = local_ergs_settings['max_selling_time']
            if local_settings_max_selling_time + 28 <= local_max_selling_time:
                local_ergs_raw_unit_sales_ground_truth = local_ergs_raw_unit_sales
                print('ground_truth data obtained')
                print('length raw data ground truth:', local_ergs_raw_unit_sales_ground_truth.shape[1])
                local_ergs_raw_unit_sales = local_ergs_raw_unit_sales[:, :local_settings_max_selling_time]
                print('length raw data for training:', local_ergs_raw_unit_sales.shape[1])
            elif local_max_selling_time != local_settings_max_selling_time:
                print("settings doesn't match data dimensions, it must be rechecked before continue"
                      "(_day_by_day_best_lower_error_model_module)")
                logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                                     ' data dimensions does not match settings']))
                return False
            else:
                if local_ergs_settings['competition_stage'] != 'submitting_after_June_1th_using_1941days':
                    print(''.join(['\x1b[0;2;41m', 'Warning', '\x1b[0m']))
                    print('please check: forecast horizon days will be included within training data')
                    print('It was expected that the last 28 days were not included..')
                    print('to avoid overfitting')
                elif local_ergs_settings['competition_stage'] == 'submitting_after_June_1th_using_1941days':
                    print(''.join(['\x1b[0;2;41m', 'Straight end of the competition', '\x1b[0m']))
                    print('settings indicate that this is the last stage!')
                    print('caution: take in consideration that evaluations in this point are not useful, '
                          'because will be made using the last data (the same used in training)')

            # calculating witch forecasts improved well and making this best forecast based in previous time_steps
            local_result_ts_model_mse = np.load(''.join([local_ergs_settings['models_evaluation_path'],
                                                         'ts_model_mse.npy']))
            local_ts_forecast_improved = \
                local_result_ts_model_mse[local_result_ts_model_mse[:, 2] < local_threshold][:, 0]
            print('time_series forecast improved (< threshold: ', local_threshold, ') -->',
                  len(local_ts_forecast_improved))
            for local_time_serie in local_ts_forecast_improved:
                local_best_model = local_result_ts_model_mse[int(local_time_serie), 1]
                if local_best_model == 1:
                    best_lower_error_ts_y_pred[int(local_time_serie), :] = \
                        first_model_forecast[int(local_time_serie), :]
                elif local_best_model == 2:
                    best_lower_error_ts_y_pred[int(local_time_serie), :] = \
                        second_model_forecast[int(local_time_serie), :]
                elif local_best_model == 3:
                    best_lower_error_ts_y_pred[int(local_time_serie), :] = \
                        third_model_forecast[int(local_time_serie), :]
                elif local_best_model == 4:
                    best_lower_error_ts_y_pred[int(local_time_serie), :] = \
                        fourth_model_forecast[int(local_time_serie), :]
                elif local_best_model == 5:
                    best_lower_error_ts_y_pred[int(local_time_serie), :] = \
                        fifth_model_forecast[int(local_time_serie), :]
                elif local_best_model == 6:
                    best_lower_error_ts_y_pred[int(local_time_serie), :] = \
                        sixth_model_forecast[int(local_time_serie), :]
                elif local_best_model == 7:
                    best_lower_error_ts_y_pred[int(local_time_serie), :] = \
                        seventh_model_forecast[int(local_time_serie), :]
                elif local_best_model == 8:
                    best_lower_error_ts_y_pred[int(local_time_serie), :] = \
                        eighth_model_forecast[int(local_time_serie), :]
                elif local_best_model == 9:
                    best_lower_error_ts_y_pred[int(local_time_serie), :] = \
                        ninth_model_forecast[int(local_time_serie), :]
                elif local_best_model == 11:
                    best_lower_error_ts_y_pred[int(local_time_serie), :] = \
                        eleventh_model_forecast[int(local_time_serie), :]
                else:
                    print('model number did not understood')
                    print('aborting process')
                    return False

            # applying smart_reshift in the poor results (>threshold) time_series forecasts
            local_ts_forecast_not_improved = \
                local_result_ts_model_mse[local_result_ts_model_mse[:, 2] >= local_threshold][:, 0]
            print('time_series forecast not improved (>= threshold: ', local_threshold, ') -->',
                  len(local_ts_forecast_not_improved))
            for local_time_serie in local_ts_forecast_not_improved:
                local_array_for_smart_reshift_analysis = \
                    local_ergs_raw_unit_sales[int(local_time_serie), - 2 * local_forecast_horizon_days:]
                local_forecast_smart_reshift = make_smart_reshift(local_array_for_smart_reshift_analysis,
                                                                local_forecast_horizon_days)
                best_lower_error_ts_y_pred[int(local_time_serie), :] = local_forecast_smart_reshift
            print('forecasts based in smartReshift finished')

            # saving best mse_based between different models forecast and submission
            # submission name is : best_mse_and_select_smartReshift_model_forecast
            store_and_submit_best_model_forecast = save_forecast_and_submission()
            point_error_based_best_model_save_review = \
                store_and_submit_best_model_forecast.store_and_submit(submission_name, local_ergs_settings,
                                                                      best_lower_error_ts_y_pred)
            if point_error_based_best_model_save_review:
                print('forecast_build_based_in_results and reshift data and submission done')
            else:
                print('error at storing forecast_build_based_in_results and reshift or at submission')

            # evaluating the forecast_build_based_in_results_ts_model
            local_ergs_forecasts_name = 'best_mse_and_smart_reshift_model_forecast'
            zeros_as_forecast = stochastic_simulation_results_analysis()
            zeros_as_forecast_review = \
                zeros_as_forecast.evaluate_stochastic_simulation(local_ergs_settings,
                                                                 local_model_ergs_hyperparameters,
                                                                 local_ergs_raw_unit_sales,
                                                                 local_ergs_raw_unit_sales_ground_truth,
                                                                 local_ergs_forecasts_name)

            # saving errors by time_serie and storing the estimated model forecast
            ts_model_mse = np.array(ts_model_mse)
            np.save(''.join([local_ergs_settings['models_evaluation_path'],
                             'forecast_build_based_in_results_ts_model']), ts_model_mse)
            np.savetxt(''.join([local_ergs_settings['models_evaluation_path'],
                                'forecast_build_based_in_results_ts_model.csv']),
                       ts_model_mse, fmt='%10.15f', delimiter=',', newline='\n')
        except Exception as submodule_error:
            print('build_forecast based in results submodule_error: ', submodule_error)
            logger.info('error in build_forecast based in results  submodule')
            logger.error(str(submodule_error), exc_info=True)
            return False
        return True
