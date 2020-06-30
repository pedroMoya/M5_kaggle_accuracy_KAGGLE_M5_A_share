# analyzing each point forecast and selecting the best, day by day, saving forecasts and making final forecast
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

class explore_day_by_day_results_and_generate_submission:

    def run(self, submission_name, local_ergs_settings):
        try:
            print('\nstarting the granular day_by_day ts_by_ts point forecast selection approach')
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
            best_mse_model_forecast = np.load(''.join([local_ergs_settings['train_data_path'],
                                                       'mse_based_best_ts_forecast.npy']))
            
            # day by day comparison
            with open(''.join([local_ergs_settings['hyperparameters_path'],
                               'organic_in_block_time_serie_based_model_hyperparameters.json'])) \
                    as local_r_json_file:
                local_model_ergs_hyperparameters = json.loads(local_r_json_file.read())
                local_r_json_file.close()
            nof_ts = local_ergs_settings['number_of_time_series']
            local_forecast_horizon_days = local_ergs_settings['forecast_horizon_days']
            best_lower_error_ts_day_by_day_y_pred = np.zeros(shape=(nof_ts, local_forecast_horizon_days),
                                                             dtype=np.dtype('float32'))
            count_best_first_model, count_best_second_model, count_best_third_model, count_best_fourth_model,\
                count_best_fifth_model, count_best_sixth_model, count_best_seventh_model, count_best_eighth_model,\
                count_best_ninth_model, count_best_mse_model = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ts_model_mse = []

            # accessing ground_truth data and rechecking stage of competition
            local_ergs_raw_data_filename = 'sales_train_evaluation.csv'
            local_ergs_raw_unit_sales = pd.read_csv(''.join([local_ergs_settings['raw_data_path'],
                                                             local_ergs_raw_data_filename]))
            print('raw sales data accessed (day_by_day_approach_best_lower_error_model results evaluation)')
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

            # will only use the last data available
            local_ergs_raw_unit_sales_ground_truth = \
                local_ergs_raw_unit_sales_ground_truth[:, -local_forecast_horizon_days:]

            # very granular approach
            # iterating in each point_forecast, calculating error and selecting best lower error model forecast
            for time_serie_index, day_index in it.product(range(nof_ts), range(local_forecast_horizon_days)):
                # acquiring day_by_day data
                ground_truth_ts_day = local_ergs_raw_unit_sales_ground_truth[time_serie_index, day_index]
                first_model_ts_day = first_model_forecast[time_serie_index, day_index]
                second_model_ts_day = second_model_forecast[time_serie_index, day_index]
                third_model_ts_day = third_model_forecast[time_serie_index, day_index]
                fourth_model_ts_day = fourth_model_forecast[time_serie_index, day_index]
                fifth_model_ts_day = fifth_model_forecast[time_serie_index, day_index]
                sixth_model_ts_day = sixth_model_forecast[time_serie_index, day_index]
                seventh_model_ts_day = seventh_model_forecast[time_serie_index, day_index]
                eighth_model_ts_day = eighth_model_forecast[time_serie_index, day_index].astype(np.dtype('float32'))
                ninth_model_ts_day = ninth_model_forecast[time_serie_index, day_index]
                best_mse_model_ts_day = best_mse_model_forecast[time_serie_index, day_index]

                # calculating error
                first_model_ts_day_error = np.abs(ground_truth_ts_day - first_model_ts_day)
                second_model_ts_day_error = np.abs(ground_truth_ts_day - second_model_ts_day)
                third_model_ts_day_error = np.abs(ground_truth_ts_day - third_model_ts_day)
                fourth_model_ts_day_error = np.abs(ground_truth_ts_day - fourth_model_ts_day)
                fifth_model_ts_day_error = np.abs(ground_truth_ts_day - fifth_model_ts_day)
                sixth_model_ts_day_error = np.abs(ground_truth_ts_day - sixth_model_ts_day)
                seventh_model_ts_day_error = np.abs(ground_truth_ts_day - seventh_model_ts_day)
                eighth_model_ts_day_error = np.abs(ground_truth_ts_day - eighth_model_ts_day)
                ninth_model_ts_day_error = np.abs(ground_truth_ts_day - ninth_model_ts_day)
                best_mse_model_ts_day_error = np.abs(ground_truth_ts_day - best_mse_model_ts_day)

                # selecting best point ts_day forecast
                if first_model_ts_day_error <= second_model_ts_day_error and \
                        first_model_ts_day_error <= third_model_ts_day_error \
                        and first_model_ts_day_error <= fourth_model_ts_day_error \
                        and first_model_ts_day_error <= fifth_model_ts_day_error\
                        and first_model_ts_day_error <= sixth_model_ts_day_error \
                        and first_model_ts_day_error <= seventh_model_ts_day_error\
                        and first_model_ts_day_error <= eighth_model_ts_day_error \
                        and first_model_ts_day_error <= ninth_model_ts_day_error \
                        and first_model_ts_day_error <= best_mse_model_ts_day_error:
                    best_lower_error_ts_day_by_day_y_pred[time_serie_index, day_index] = first_model_ts_day
                    count_best_first_model += 1
                    ts_model_mse.append([time_serie_index, int(1), first_model_ts_day_error])
                # elif best_mse_model_ts_day_error <= first_model_ts_day_error \
                #         and best_mse_model_ts_day_error <= second_model_ts_day_error \
                #         and best_mse_model_ts_day_error <= third_model_ts_day_error \
                #         and best_mse_model_ts_day_error <= fourth_model_ts_day_error\
                #         and best_mse_model_ts_day_error <= fifth_model_ts_day_error \
                #         and best_mse_model_ts_day_error <= sixth_model_ts_day_error\
                #         and best_mse_model_ts_day_error <= seventh_model_ts_day_error \
                #         and best_mse_model_ts_day_error <= eighth_model_ts_day_error\
                #         and best_mse_model_ts_day_error <= ninth_model_ts_day_error:
                #     best_lower_error_ts_day_by_day_y_pred[time_serie_index, day_index] = best_mse_model_ts_day
                #     count_best_mse_model += 1
                #     ts_model_mse.append([time_serie_index, int(10), best_mse_model_ts_day_error])
                elif second_model_ts_day_error <= first_model_ts_day_error \
                        and second_model_ts_day_error <= third_model_ts_day_error \
                        and second_model_ts_day_error <= fourth_model_ts_day_error \
                        and second_model_ts_day_error <= fifth_model_ts_day_error\
                        and second_model_ts_day_error <= sixth_model_ts_day_error \
                        and second_model_ts_day_error <= seventh_model_ts_day_error\
                        and second_model_ts_day_error <= eighth_model_ts_day_error \
                        and second_model_ts_day_error <= ninth_model_ts_day_error\
                        and second_model_ts_day_error <= best_mse_model_ts_day_error:
                    best_lower_error_ts_day_by_day_y_pred[time_serie_index, day_index] = second_model_ts_day
                    count_best_second_model += 1
                    ts_model_mse.append([time_serie_index, int(2), second_model_ts_day_error])
                elif third_model_ts_day_error <= first_model_ts_day_error \
                        and third_model_ts_day_error <= second_model_ts_day_error \
                        and third_model_ts_day_error <= fourth_model_ts_day_error \
                        and third_model_ts_day_error <= fifth_model_ts_day_error\
                        and third_model_ts_day_error <= sixth_model_ts_day_error \
                        and third_model_ts_day_error <= seventh_model_ts_day_error\
                        and third_model_ts_day_error <= eighth_model_ts_day_error \
                        and third_model_ts_day_error <= ninth_model_ts_day_error\
                        and third_model_ts_day_error <= best_mse_model_ts_day_error:
                    best_lower_error_ts_day_by_day_y_pred[time_serie_index, day_index] = third_model_ts_day
                    count_best_third_model += 1
                    ts_model_mse.append([time_serie_index, int(3), third_model_ts_day_error])
                # elif fourth_model_ts_day_error <= first_model_ts_day_error \
                #         and fourth_model_ts_day_error <= second_model_ts_day_error \
                #         and fourth_model_ts_day_error <= third_model_ts_day_error \
                #         and fourth_model_ts_day_error <= fifth_model_ts_day_error\
                #         and fourth_model_ts_day_error <= sixth_model_ts_day_error \
                #         and fourth_model_ts_day_error <= seventh_model_ts_day_error\
                #         and fourth_model_ts_day_error <= eighth_model_ts_day_error \
                #         and fourth_model_ts_day_error <= ninth_model_ts_day_error\
                #         and fourth_model_ts_day_error <= best_mse_model_ts_day_error:
                #     best_lower_error_ts_day_by_day_y_pred[time_serie_index, day_index] = fourth_model_ts_day
                #     count_best_fourth_model += 1
                #     ts_model_mse.append([time_serie_index, int(4), fourth_model_ts_day_error])
                elif fifth_model_ts_day_error <= first_model_ts_day_error \
                        and fifth_model_ts_day_error <= second_model_ts_day_error \
                        and fifth_model_ts_day_error <= third_model_ts_day_error \
                        and fifth_model_ts_day_error <= fourth_model_ts_day_error\
                        and fifth_model_ts_day_error <= sixth_model_ts_day_error \
                        and fifth_model_ts_day_error <= seventh_model_ts_day_error\
                        and fifth_model_ts_day_error <= eighth_model_ts_day_error \
                        and fifth_model_ts_day_error <= ninth_model_ts_day_error\
                        and fifth_model_ts_day_error <= best_mse_model_ts_day_error:
                    best_lower_error_ts_day_by_day_y_pred[time_serie_index, day_index] = fifth_model_ts_day
                    count_best_fifth_model += 1
                    ts_model_mse.append([time_serie_index, int(5), fifth_model_ts_day_error])
                elif sixth_model_ts_day_error <= first_model_ts_day_error \
                        and sixth_model_ts_day_error <= second_model_ts_day_error \
                        and sixth_model_ts_day_error <= third_model_ts_day_error \
                        and sixth_model_ts_day_error <= fourth_model_ts_day_error\
                        and sixth_model_ts_day_error <= fifth_model_ts_day_error \
                        and sixth_model_ts_day_error <= seventh_model_ts_day_error\
                        and sixth_model_ts_day_error <= eighth_model_ts_day_error \
                        and sixth_model_ts_day_error <= ninth_model_ts_day_error\
                        and sixth_model_ts_day_error <= best_mse_model_ts_day_error:
                    best_lower_error_ts_day_by_day_y_pred[time_serie_index, day_index] = sixth_model_ts_day
                    count_best_sixth_model += 1
                    ts_model_mse.append([time_serie_index, int(6), sixth_model_ts_day_error])
                elif seventh_model_ts_day_error <= first_model_ts_day_error \
                        and seventh_model_ts_day_error <= second_model_ts_day_error \
                        and seventh_model_ts_day_error <= third_model_ts_day_error \
                        and seventh_model_ts_day_error <= fourth_model_ts_day_error\
                        and seventh_model_ts_day_error <= fifth_model_ts_day_error \
                        and seventh_model_ts_day_error <= sixth_model_ts_day_error\
                        and seventh_model_ts_day_error <= eighth_model_ts_day_error \
                        and seventh_model_ts_day_error <= ninth_model_ts_day_error\
                        and seventh_model_ts_day_error <= best_mse_model_ts_day_error:
                    best_lower_error_ts_day_by_day_y_pred[time_serie_index, day_index] = seventh_model_ts_day
                    count_best_seventh_model += 1
                    ts_model_mse.append([time_serie_index, int(7), seventh_model_ts_day_error])
                elif ninth_model_ts_day_error <= first_model_ts_day_error \
                        and ninth_model_ts_day_error <= second_model_ts_day_error \
                        and ninth_model_ts_day_error <= third_model_ts_day_error \
                        and ninth_model_ts_day_error <= fourth_model_ts_day_error\
                        and ninth_model_ts_day_error <= fifth_model_ts_day_error \
                        and ninth_model_ts_day_error <= sixth_model_ts_day_error\
                        and ninth_model_ts_day_error <= seventh_model_ts_day_error \
                        and ninth_model_ts_day_error <= eighth_model_ts_day_error\
                        and ninth_model_ts_day_error <= best_mse_model_ts_day_error:
                    best_lower_error_ts_day_by_day_y_pred[time_serie_index, day_index] = ninth_model_ts_day
                    count_best_ninth_model += 1
                    ts_model_mse.append([time_serie_index, int(9), ninth_model_ts_day_error])
                else:
                    best_lower_error_ts_day_by_day_y_pred[time_serie_index, day_index] = eighth_model_ts_day
                    count_best_eighth_model += 1
                    ts_model_mse.append([time_serie_index, int(8), eighth_model_ts_day_error])

            # finally reporting the results
            print('it was used ', count_best_first_model, ' ts day_by_day forecasts from first model')
            print('it was used ', count_best_second_model, ' ts day_by_day forecasts from second model')
            print('it was used ', count_best_third_model, ' ts day_by_day forecasts from third model')
            print('it was used ', count_best_fourth_model, ' ts day_by_day forecasts from fourth model')
            print('it was used ', count_best_fifth_model, ' ts day_by_day forecasts from fifth model')
            print('it was used ', count_best_sixth_model, ' ts day_by_day forecasts from sixth model')
            print('it was used ', count_best_seventh_model, ' ts day_by_day forecasts from seventh model')
            print('it was used ', count_best_eighth_model, ' ts day_by_day forecasts from eighth model')
            print('it was used ', count_best_ninth_model, ' ts day_by_day forecasts from ninth model')
            print('it was used ', count_best_mse_model, ' ts day_by_day forecasts from best_mse (tenth) model')

            # saving best mse_based between different models forecast and submission
            store_and_submit_best_model_forecast = save_forecast_and_submission()
            point_error_based_best_model_save_review = \
                store_and_submit_best_model_forecast.store_and_submit(submission_name, local_ergs_settings,
                                                                      best_lower_error_ts_day_by_day_y_pred)
            if point_error_based_best_model_save_review:
                print('best low point forecast error and generate_submission data and submission done')
            else:
                print('error at storing best_low_point_forecast_error data and generate_submission or submission')

            # evaluating the best_lower_error criteria granular_model forecast
            local_ergs_forecasts_name = 'day_by_day_best_low_error_criteria_model_forecast'
            zeros_as_forecast = stochastic_simulation_results_analysis()
            zeros_as_forecast_review = \
                zeros_as_forecast.evaluate_stochastic_simulation(local_ergs_settings,
                                                                 local_model_ergs_hyperparameters,
                                                                 local_ergs_raw_unit_sales,
                                                                 local_ergs_raw_unit_sales_ground_truth,
                                                                 local_ergs_forecasts_name)

            # saving errors by time_serie and storing the estimated best model
            ts_model_mse = np.array(ts_model_mse)
            np.save(''.join([local_ergs_settings['models_evaluation_path'],
                             'best_low_point_forecast_error_ts_model_mse']), ts_model_mse)
            np.savetxt(''.join([local_ergs_settings['models_evaluation_path'],
                                'best_low_point_forecast_error_ts_model_mse.csv']),
                       ts_model_mse, fmt='%10.15f', delimiter=',', newline='\n')
        except Exception as submodule_error:
            print('best low point forecast error and generate_submission submodule_error: ', submodule_error)
            logger.info('error in best low point forecast error and generate_submission submodule')
            logger.error(str(submodule_error), exc_info=True)
            return False
        return True
