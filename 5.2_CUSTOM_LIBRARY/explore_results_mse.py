# saving forecasts and making forecast
import os
import sys
import datetime
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
from save_forecast_and_make_submission import save_forecast_and_submission
from stochastic_model_obtain_results import stochastic_simulation_results_analysis

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
            third_model_forecast = np.load(''.join([local_ergs_settings['train_data_path'],
                                                   'third_model_forecast_data.npy']))
            fourth_model_forecast = np.load(''.join([local_ergs_settings['train_data_path'],
                                                     'fourth_model_forecast_data.npy']))
            # this forecast has the shape=30490, 28
            fifth_model_forecast_30490_28 = np.load(''.join([local_ergs_settings['train_data_path'],
                                                             'fifth_model_forecast_data.npy']))
            fifth_model_forecast = np.zeros(shape=(60980, 28), dtype=np.dtype('float32'))
            fifth_model_forecast[0: 30490, :] = fifth_model_forecast_30490_28
            # saving fifth model submission as (local_name).csv
            # local_name = 'fifth_model'
            # save_submission_stochastic_simulation = save_submission()
            # save_submission_review = \
            #     save_submission_stochastic_simulation.save(''.join([local_name, '_submission.csv']),
            #                                                fifth_model_forecast,
            #                                                local_ergs_settings)
            # if save_submission_review:
            #     print('fifth model saved in csv format as submission')

            seventh_model_forecast = np.load(''.join([local_ergs_settings['train_data_path'],
                                                     'seventh_model_forecast_data.npy']))
            eighth_model_forecast = np.load(''.join([local_ergs_settings['train_data_path'],
                                                    'eighth_model_nearest_neighbor_forecast_data.npy']))
            ninth_model_forecast = np.load(''.join([local_ergs_settings['train_data_path'],
                                                    'ninth_model_random_average_simulation_forecast_data.npy']))

            # ----------------------------------------------------------------------------------------
            # sixth_model is COMBINATION MODEL
            sixth_model_forecast = np.add(first_model_forecast, second_model_forecast)
            sixth_model_forecast = np.add(sixth_model_forecast, third_model_forecast)
            sixth_model_forecast = np.add(sixth_model_forecast, fifth_model_forecast)
            # sixth_model_forecast = np.add(sixth_model_forecast, fourth_model_forecast)
            sixth_model_forecast = np.add(sixth_model_forecast, seventh_model_forecast)
            sixth_model_forecast = np.add(sixth_model_forecast, eighth_model_forecast)
            sixth_model_forecast = np.add(sixth_model_forecast, ninth_model_forecast)
            sixth_model_forecast = np.divide(sixth_model_forecast, 7.)
            np.save(''.join([local_ergs_settings['train_data_path'], 'sixth_model_forecast_data']),
                    sixth_model_forecast)
            # ----------------------------------------------------------------------------------------

            # loading the results
            first_model_result = np.load(''.join([local_ergs_settings['models_evaluation_path'],
                                                  'time_series_results_first_model_mse.npy']))
            second_model_result = np.load(''.join([local_ergs_settings['models_evaluation_path'],
                                                   'time_series_results_second_model_mse.npy']))
            third_model_result = np.load(''.join([local_ergs_settings['models_evaluation_path'],
                                                 'time_series_results_third_model_mse.npy']))
            fourth_model_result = np.load(''.join([local_ergs_settings['models_evaluation_path'],
                                                   'time_series_results_fourth_model_mse.npy']))
            fifth_model_result = np.load(''.join([local_ergs_settings['models_evaluation_path'],
                                                 'time_series_results_fifth_model_mse.npy']))
            seventh_model_result = np.load(''.join([local_ergs_settings['models_evaluation_path'],
                                                    'time_series_results_seventh_model_mse.npy']))
            eighth_model_result = np.load(''.join([local_ergs_settings['models_evaluation_path'],
                                                  'nearest_neighbor_time_series_results_model_mse.npy']))
            ninth_model_result = np.load(''.join([local_ergs_settings['models_evaluation_path'],
                                                  'time_series_results_ninth_model_mse.npy']))

            if local_ergs_settings['results_mse_with_model_combination_as_forecasts_done'] != 'True':
                print('applying sixth model')
                zeros_as_forecast = stochastic_simulation_results_analysis()
                with open(''.join([local_ergs_settings['hyperparameters_path'],
                                   'organic_in_block_time_serie_based_model_hyperparameters.json'])) \
                        as local_r_json_file:
                    local_model_ergs_hyperparameters = json.loads(local_r_json_file.read())
                    local_r_json_file.close()
                local_ergs_raw_data_filename = 'sales_train_evaluation.csv'
                local_ergs_raw_unit_sales = pd.read_csv(''.join([local_ergs_settings['raw_data_path'],
                                                                 local_ergs_raw_data_filename]))
                print('raw sales data accessed (sixth_model results evaluation)')
                # extract data and check  dimensions
                local_ergs_raw_unit_sales = local_ergs_raw_unit_sales.iloc[:, 6:].values
                local_max_selling_time = np.shape(local_ergs_raw_unit_sales)[1]
                local_settings_max_selling_time = local_ergs_settings['max_selling_time']
                if local_settings_max_selling_time + 28 <= local_max_selling_time:
                    local_ergs_raw_unit_sales_ground_truth = local_ergs_raw_unit_sales
                    print('length raw data ground truth:', local_ergs_raw_unit_sales_ground_truth.shape[1])
                    local_ergs_raw_unit_sales = local_ergs_raw_unit_sales[:, :local_settings_max_selling_time]
                    print('length raw data for training:', local_ergs_raw_unit_sales.shape[1])
                elif local_max_selling_time != local_settings_max_selling_time:
                    print("settings doesn't match data dimensions, it must be rechecked before continue"
                          "(_sixth_model_results_module)")
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
                local_ergs_forecasts_name = 'sixth_model_forecast'
                zeros_as_forecast_review = \
                    zeros_as_forecast.evaluate_stochastic_simulation(local_ergs_settings,
                                                                     local_model_ergs_hyperparameters,
                                                                     local_ergs_raw_unit_sales,
                                                                     local_ergs_raw_unit_sales_ground_truth,
                                                                     local_ergs_forecasts_name)
            sixth_model_result = np.load(''.join([local_ergs_settings['models_evaluation_path'],
                                                  'time_series_results_sixth_model_mse.npy']))
            # saving SIXTH_MODEL_FORECAST combination model
            store_and_submit_sixth_model_forecast = save_forecast_and_submission()
            sixth_model_save_review = \
                store_and_submit_sixth_model_forecast.store_and_submit('sixth_model_combination_', local_ergs_settings,
                                                                       sixth_model_forecast[0: 30490, :])
            if sixth_model_save_review:
                print('sixth model forecast data and submission done')
            else:
                print('error at storing sixth model forecast data or submission')

            # -----------------------kernel------------------------------------------
            nof_ts = local_ergs_settings['number_of_time_series']
            local_forecast_horizon_days = local_ergs_settings['forecast_horizon_days']
            best_y_pred = np.zeros(shape=(nof_ts, local_forecast_horizon_days), dtype=np.dtype('float32'))
            count_best_first_model, count_best_second_model, count_best_third_model, count_best_fourth_model,\
                count_best_fifth_model, count_best_sixth_model, count_best_seventh_model, count_best_eighth_model,\
                count_best_ninth_model = 0, 0, 0, 0, 0, 0, 0, 0, 0
            ts_model_mse = []
            for time_serie_index in range(nof_ts):
                first_model_mse = first_model_result[time_serie_index][2]
                second_model_mse = second_model_result[time_serie_index][2]
                third_model_mse = third_model_result[time_serie_index][2]
                # fourth_model_mse = fourth_model_result[time_serie_index][2]
                fifth_model_mse = fifth_model_result[time_serie_index][1]
                sixth_model_mse = sixth_model_result[time_serie_index][2]
                seventh_model_mse = seventh_model_result[time_serie_index][2]
                # eighth_model_mse = eighth_model_result[time_serie_index][2]
                # ninth_model_mse = ninth_model_result[time_serie_index][2]
                eighth_model_mse = 5000000
                ninth_model_mse = 5000000
                fourth_model_mse = 5000000
                if first_model_mse <= second_model_mse and first_model_mse <= third_model_mse \
                        and first_model_mse <= fourth_model_mse and first_model_mse <= fifth_model_mse\
                        and first_model_mse <= sixth_model_mse and first_model_mse <= seventh_model_mse\
                        and first_model_mse <= eighth_model_mse and first_model_mse <= ninth_model_mse:
                    best_y_pred[time_serie_index, :] = first_model_forecast[time_serie_index, :]
                    count_best_first_model += 1
                    ts_model_mse.append([time_serie_index, int(1), first_model_mse])
                elif second_model_mse <= first_model_mse and second_model_mse <= third_model_mse \
                        and second_model_mse <= fourth_model_mse and second_model_mse <= fifth_model_mse\
                        and second_model_mse <= sixth_model_mse and second_model_mse <= seventh_model_mse\
                        and second_model_mse <= eighth_model_mse and second_model_mse <= ninth_model_mse:
                    best_y_pred[time_serie_index, :] = second_model_forecast[time_serie_index, :]
                    count_best_second_model += 1
                    ts_model_mse.append([time_serie_index, int(2), second_model_mse])
                elif third_model_mse <= first_model_mse and third_model_mse <= second_model_mse \
                        and third_model_mse <= fourth_model_mse and third_model_mse <= fifth_model_mse\
                        and third_model_mse <= sixth_model_mse and third_model_mse <= seventh_model_mse\
                        and third_model_mse <= eighth_model_mse and third_model_mse <= ninth_model_mse:
                    best_y_pred[time_serie_index, :] = third_model_forecast[time_serie_index, :]
                    count_best_third_model += 1
                    ts_model_mse.append([time_serie_index, int(3), third_model_mse])
                elif fourth_model_mse <= first_model_mse and fourth_model_mse <= second_model_mse \
                        and fourth_model_mse <= third_model_mse and fourth_model_mse <= fifth_model_mse\
                        and fourth_model_mse <= sixth_model_mse and fourth_model_mse <= seventh_model_mse\
                        and fourth_model_mse <= eighth_model_mse and fourth_model_mse <= ninth_model_mse:
                    best_y_pred[time_serie_index, :] = fourth_model_forecast[time_serie_index, :]
                    count_best_fourth_model += 1
                    ts_model_mse.append([time_serie_index, int(4), fourth_model_mse])
                elif fifth_model_mse <= first_model_mse and fifth_model_mse <= second_model_mse \
                        and fifth_model_mse <= third_model_mse and fifth_model_mse <= fourth_model_mse\
                        and fifth_model_mse <= sixth_model_mse and fifth_model_mse <= seventh_model_mse\
                        and fifth_model_mse <= eighth_model_mse and fifth_model_mse <= ninth_model_mse:
                    best_y_pred[time_serie_index, :] = fifth_model_forecast[time_serie_index, :]
                    count_best_fifth_model += 1
                    ts_model_mse.append([time_serie_index, int(5), fifth_model_mse])
                elif sixth_model_mse <= first_model_mse and sixth_model_mse <= second_model_mse \
                        and sixth_model_mse <= third_model_mse and sixth_model_mse <= fourth_model_mse\
                        and sixth_model_mse <= fifth_model_mse and sixth_model_mse <= seventh_model_mse\
                        and sixth_model_mse <= eighth_model_mse and sixth_model_mse <= ninth_model_mse:
                    best_y_pred[time_serie_index, :] = sixth_model_forecast[time_serie_index, :]
                    count_best_sixth_model += 1
                    ts_model_mse.append([time_serie_index, int(6), sixth_model_mse])
                elif seventh_model_mse <= first_model_mse and seventh_model_mse <= second_model_mse \
                        and seventh_model_mse <= third_model_mse and seventh_model_mse <= fourth_model_mse\
                        and seventh_model_mse <= fifth_model_mse and seventh_model_mse <= sixth_model_mse\
                        and seventh_model_mse <= eighth_model_mse and seventh_model_mse <= ninth_model_mse:
                    best_y_pred[time_serie_index, :] = seventh_model_forecast[time_serie_index, :]
                    count_best_seventh_model += 1
                    ts_model_mse.append([time_serie_index, int(7), seventh_model_mse])
                elif ninth_model_mse <= first_model_mse and ninth_model_mse <= second_model_mse \
                        and ninth_model_mse <= third_model_mse and ninth_model_mse <= fourth_model_mse\
                        and ninth_model_mse <= fifth_model_mse and ninth_model_mse <= sixth_model_mse\
                        and ninth_model_mse <= eighth_model_mse and ninth_model_mse <= eighth_model_mse:
                    best_y_pred[time_serie_index, :] = ninth_model_forecast[time_serie_index, :]
                    count_best_ninth_model += 1
                    ts_model_mse.append([time_serie_index, int(9), ninth_model_mse])
                else:
                    best_y_pred[time_serie_index, :] = \
                        eighth_model_forecast[time_serie_index, :].astype(np.dtype('float32'))
                    count_best_eighth_model += 1
                    ts_model_mse.append([time_serie_index, int(8), eighth_model_mse])
            print('\nbest_mse_model forecast consolidation:')
            print('it was used ', count_best_first_model, ' ts forecasts from first model')
            print('it was used ', count_best_second_model, ' ts forecasts from second model')
            print('it was used ', count_best_third_model, ' ts forecasts from third model')
            print('it was used ', count_best_fourth_model, ' ts forecasts from fourth model')
            print('it was used ', count_best_fifth_model, ' ts forecasts from fifth model')
            print('it was used ', count_best_sixth_model, ' ts forecasts from sixth model')
            print('it was used ', count_best_seventh_model, ' ts forecasts from seventh model')
            print('it was used ', count_best_eighth_model, ' ts forecasts from eighth model')
            print('it was used ', count_best_ninth_model, ' ts forecasts from ninth model')

            # evaluating the best_mse criteria final_model forecast
            local_ergs_forecasts_name = 'final_best_mse_criteria_model_forecast'
            zeros_as_forecast = stochastic_simulation_results_analysis()
            zeros_as_forecast_review = \
                zeros_as_forecast.evaluate_stochastic_simulation(local_ergs_settings,
                                                                 local_model_ergs_hyperparameters,
                                                                 local_ergs_raw_unit_sales,
                                                                 local_ergs_raw_unit_sales_ground_truth,
                                                                 local_ergs_forecasts_name)

            # saving best mse_based between different models forecast and submission
            store_and_submit_best_model_forecast = save_forecast_and_submission()
            mse_based_best_model_save_review = \
                store_and_submit_best_model_forecast.store_and_submit(submission_name, local_ergs_settings,
                                                                      best_y_pred)
            if mse_based_best_model_save_review:
                print('best mse_based model forecast data and submission done')
            else:
                print('error at storing best mse_based model forecast data or submission')

            # saving mse by time_serie and indicating the best model
            ts_model_mse = np.array(ts_model_mse)
            np.save(''.join([local_ergs_settings['models_evaluation_path'], 'ts_model_mse']), ts_model_mse)
            np.savetxt(''.join([local_ergs_settings['models_evaluation_path'], 'ts_model_mse.csv']),
                       ts_model_mse, fmt='%10.15f', delimiter=',', newline='\n')
        except Exception as submodule_error:
            print('explore_results and generate_submission submodule_error: ', submodule_error)
            logger.info('error in explore_results and generate_submission submodule')
            logger.error(str(submodule_error), exc_info=True)
            return False
        return True
