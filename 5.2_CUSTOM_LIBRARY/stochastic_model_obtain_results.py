# Model architecture analyzer
import os
import logging
import logging.handlers as handlers
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

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


class stochastic_simulation_results_analysis:

    def evaluate_stochastic_simulation(self, local_settings, local_model_hyperparameters, local_raw_unit_sales,
                                       local_raw_unit_sales_ground_truth, local_forecasts_name):
        try:
            # evaluating model and for comparing with threshold defined in settings
            # (organic_in_block_time_serie_based_model_hyperparameters.json)
            if local_forecasts_name == 'first_model_forecast':
                print('\nevaluating the first model (stochastic_simulation) trained..')
                local_forecasts = np.load(''.join([local_settings['train_data_path'],
                                                   'first_model_forecast_data.npy']))
            elif local_forecasts_name == 'second_model_forecast':
                print('\nevaluating the second model trained..')
                local_forecasts = np.load(''.join([local_settings['train_data_path'],
                                                  'second_model_forecast_data.npy']))
            elif local_forecasts_name == 'third_model_forecast':
                print('\nevaluating the accumulated_frequencies_approach (third model) trained..')
                local_forecasts = np.load(''.join([local_settings['train_data_path'],
                                                  'third_model_forecast_data.npy']))
            elif local_forecasts_name == 'fourth_model_forecast':
                print('\nevaluating the accumulated_frequencies_approach_in_block_NN (fourth model) trained..')
                local_forecasts = np.load(''.join([local_settings['train_data_path'],
                                                  'fourth_model_forecast_data.npy']))
            elif local_forecasts_name == 'sixth_model_forecast':
                print('\nevaluating the combination forecast (sixth model) ....')
                local_forecasts = np.load(''.join([local_settings['train_data_path'],
                                                  'sixth_model_forecast_data.npy']))
            elif local_forecasts_name == 'seventh_model_forecast':
                print('\nevaluating the acc_freq_NN individual time_series forecast (seventh model) ....')
                local_forecasts = np.load(''.join([local_settings['train_data_path'],
                                                   'seventh_model_forecast_data.npy']))
            elif local_forecasts_name == 'final_best_mse_criteria_model_forecast':
                print('\nevaluating the final_best_mse_criteria_model_forecast (final days_in_block model) ....')
                local_forecasts = np.load(''.join([local_settings['train_data_path'],
                                                   'mse_based_best_ts_forecast.npy']))
            elif local_forecasts_name == 'ninth_model_acc_freq_and_nearest_neighbor_regression_forecast':
                print('\nevaluating the acc_freq and nearest_neighbor model_forecast (ninth model) ....')
                local_forecasts = \
                    np.load(''.join([local_settings['train_data_path'],
                                     'ninth_model_acc_freq_and_nearest_neighbor_regression_forecast_data.npy']))
            elif local_forecasts_name == 'day_by_day_best_low_error_criteria_model_forecast':
                print('\nevaluating the day_by_day_best_lower_error_criteria model_forecast (the granular model) ....')
                local_forecasts = \
                    np.load(''.join([local_settings['train_data_path'],
                                     'day_by_day_based_best_lower_error_ts_forecast.npy']))
            elif local_forecasts_name == 'best_mse_and_distribution_model_forecast':
                print('\nevaluating the best_mse_and_distribution_model_forecast model_forecast (smart model) ....')
                local_forecasts = \
                    np.load(''.join([local_settings['train_data_path'],
                                     'best_mse_and_distribution_model_forecast.npy']))
            elif local_forecasts_name == 'eleventh_model_forecast':
                print('\nevaluating the individual ts unit_sales neural_network model_forecast '
                      '(NN individual unit_sales model) ....')
                local_forecasts = \
                    np.load(''.join([local_settings['train_data_path'],
                                     'eleventh_model_forecast_data.npy']))
            elif local_forecasts_name == 'final_evaluation_stage_forecast_best_mse_model':
                print('\nevaluating the final_forecast_best_mse_model '
                      '(Evaluation stage, validation stage is filled with the real data) ....')
                local_forecasts = \
                    np.load(''.join([local_settings['train_data_path'],
                                     'final_evaluation_stage_forecast.npy']))
            elif local_forecasts_name == 'best_mse_and_smart_reshift_model_forecast':
                print('\nevaluating the smart_reshift model'
                      '(this is a guess adjustment stage, in evaluation stage mae helps to fill bad mse forecast) ....')
                print('in validation stage, this model pass exactly the same forecast that '
                      'final_best_mse_criteria_model_forecast')
                local_forecasts = \
                    np.load(''.join([local_settings['train_data_path'],
                                     'best_mse_and_select_smartReshift_model_forecast.npy']))
            else:
                print('\nmodel_name not expected, please review the last argument')
                print('in stochastic_model_obtain_results submodule\n')
                return False
            print(local_forecasts)
            time_serie_iterator = 0
            local_improved_time_series_forecast = []
            local_time_series_not_improved = []
            local_improved_mse = []
            local_not_improved_mse = []
            local_time_series_treated = []
            local_forecast_horizon_days = local_settings['forecast_horizon_days']
            local_stochastic_simulation_poor_result_threshold = \
                local_model_hyperparameters['stochastic_simulation_poor_result_threshold']
            print('evaluating model error by time_serie')
            nof_time_series = local_raw_unit_sales.shape[0]
            for time_serie in range(nof_time_series):
                # for time_serie in range(local_normalized_scaled_unit_sales.shape[0]):
                y_truth = local_raw_unit_sales_ground_truth[time_serie: time_serie + 1, -local_forecast_horizon_days:]
                local_point_forecast = local_forecasts[time_serie_iterator:time_serie_iterator + 1, :]
                # calculating error (MSE)
                local_error_metric_mse = mean_squared_error(y_truth, local_point_forecast)
                local_time_series_treated.append([int(time_serie), local_stochastic_simulation_poor_result_threshold,
                                                  local_error_metric_mse])
                if local_error_metric_mse < local_stochastic_simulation_poor_result_threshold:
                    # better results with time_serie specific model training
                    # print(time_serie, 'MSE improved from inf to ', local_error_metric_mse)
                    local_improved_time_series_forecast.append(int(time_serie))
                    local_improved_mse.append(local_error_metric_mse)
                else:
                    # no better results with time serie specific model training
                    # print('MSE not improved from: ', previous_result, '\t current mse: ', local_error_metric_mse)
                    local_time_series_not_improved.append(int(time_serie))
                    local_not_improved_mse.append(local_error_metric_mse)
                time_serie_iterator += 1
            local_time_series_treated = np.array(local_time_series_treated)
            local_improved_mse = np.array(local_improved_mse)
            local_not_improved_mse = np.array(local_not_improved_mse)
            average_mse_not_improved_ts = np.mean(local_not_improved_mse)
            average_mse_in_block_forecast = np.mean(local_time_series_treated[:, 2])
            average_mse_improved_ts = np.mean(local_improved_mse)
            print('mean_mse for forecast:', average_mse_in_block_forecast)
            print('number of time series with better (lower than threshold) results with this forecast: ',
                  len(local_improved_time_series_forecast))
            print('mean_mse of time series with better results with this forecast: ', average_mse_improved_ts)
            print('mean_mse of time series with worse results with this forecast: ', average_mse_not_improved_ts)
            print('not improved time series =', len(local_time_series_not_improved))
            local_improved_time_series_forecast = np.array(local_improved_time_series_forecast)
            local_time_series_not_improved = np.array(local_time_series_not_improved)

            # store data of (individual-approach) time_series forecast successfully improved and those that not
            if local_forecasts_name == 'first_model_forecast':
                np.savetxt(''.join([local_settings['models_evaluation_path'], 'first_model_mse.csv']),
                           local_time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_results_first_model_mse']),
                        local_time_series_treated)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'improved_time_series_first_model']),
                        local_improved_time_series_forecast)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_not_improved_first_model']),
                        local_time_series_not_improved)
            elif local_forecasts_name == 'second_model_forecast':
                np.savetxt(''.join([local_settings['models_evaluation_path'], 'second_model_mse.csv']),
                           local_time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_results_second_model_mse']),
                        local_time_series_treated)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'improved_time_series_second_model']),
                        local_improved_time_series_forecast)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_not_improved_second_model']),
                        local_time_series_not_improved)
            elif local_forecasts_name == 'third_model_forecast':
                np.savetxt(''.join([local_settings['models_evaluation_path'], 'third_model_mse.csv']),
                           local_time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_results_third_model_mse']),
                        local_time_series_treated)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'improved_time_series_third_model']),
                        local_improved_time_series_forecast)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_not_improved_third_model']),
                        local_time_series_not_improved)
            elif local_forecasts_name == 'fourth_model_forecast':
                np.savetxt(''.join([local_settings['models_evaluation_path'], 'fourth_model_mse.csv']),
                           local_time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_results_fourth_model_mse']),
                        local_time_series_treated)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'improved_time_series_fourth_model']),
                        local_improved_time_series_forecast)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_not_improved_fourth_model']),
                        local_time_series_not_improved)
            elif local_forecasts_name == 'sixth_model_forecast':
                np.savetxt(''.join([local_settings['models_evaluation_path'], 'sixth_model_mse.csv']),
                           local_time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_results_sixth_model_mse']),
                        local_time_series_treated)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'improved_time_series_sixth_model']),
                        local_improved_time_series_forecast)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_not_improved_sixth_model']),
                        local_time_series_not_improved)
            elif local_forecasts_name == 'seventh_model_forecast':
                np.savetxt(''.join([local_settings['models_evaluation_path'], 'seventh_model_mse.csv']),
                           local_time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_results_seventh_model_mse']),
                        local_time_series_treated)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'improved_time_series_seventh_model']),
                        local_improved_time_series_forecast)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_not_improved_seventh_model']),
                        local_time_series_not_improved)
            elif local_forecasts_name == 'eighth_model_forecast':
                np.savetxt(''.join([local_settings['models_evaluation_path'], 'eighth_model_mse.csv']),
                           local_time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_results_eighth_model_mse']),
                        local_time_series_treated)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'improved_time_series_eighth_model']),
                        local_improved_time_series_forecast)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_not_improved_eighth_model']),
                        local_time_series_not_improved)
            elif local_forecasts_name == 'ninth_model_acc_freq_and_nearest_neighbor_regression_forecast':
                np.savetxt(''.join([local_settings['models_evaluation_path'], 'ninth_model_mse.csv']),
                           local_time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_results_ninth_model_mse']),
                        local_time_series_treated)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'improved_time_series_ninth_model']),
                        local_improved_time_series_forecast)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_not_improved_ninth_model']),
                        local_time_series_not_improved)
            elif local_forecasts_name == 'final_best_mse_criteria_model_forecast':
                np.savetxt(''.join([local_settings['models_evaluation_path'], 'final_days_in_block_model_mse.csv']),
                           local_time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_results_final_days_in_block__model_mse']),
                        local_time_series_treated)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'improved_time_series_final_days_in_block_model']),
                        local_improved_time_series_forecast)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_not_improved_final_days_in_block_model']),
                        local_time_series_not_improved)
            elif local_forecasts_name == 'day_by_day_best_low_error_criteria_model_forecast':
                np.savetxt(''.join([local_settings['models_evaluation_path'], 'day_by_day_granular_model_mse.csv']),
                           local_time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_results_day_by_day_granular_model_mse']),
                        local_time_series_treated)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'improved_time_series_day_by_day_granular_model']),
                        local_improved_time_series_forecast)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_not_improved_day_by_day_granular_model']),
                        local_time_series_not_improved)
            elif local_forecasts_name == 'best_mse_and_distribution_model_forecast':
                np.savetxt(''.join([local_settings['models_evaluation_path'], 'best_mse_and_distribution_model_mse.csv']),
                           local_time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_results_best_mse_and_distribution_model_mse']),
                        local_time_series_treated)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'improved_time_series_best_mse_and_distribution_model']),
                        local_improved_time_series_forecast)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_not_improved_best_mse_and_distribution_model']),
                        local_time_series_not_improved)
            elif local_forecasts_name == 'eleventh_model_forecast':
                np.savetxt(''.join([local_settings['models_evaluation_path'], 'eleventh_model_mse.csv']),
                           local_time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_results_eleventh_model_mse']),
                        local_time_series_treated)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'improved_time_series_eleventh_model']),
                        local_improved_time_series_forecast)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_not_improved_eleventh_model']),
                        local_time_series_not_improved)
            elif local_forecasts_name == 'final_evaluation_stage_forecast_best_mse_model':
                np.savetxt(''.join([local_settings['models_evaluation_path'],
                                    'final_evaluation_stage_forecast_best_mse_model_mse.csv']),
                           local_time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_results_final_evaluation_stage_forecast_best_mse_model_mse']),
                        local_time_series_treated)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'improved_time_series_final_evaluation_stage_forecast_best_mse_model']),
                        local_improved_time_series_forecast)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_not_improved_final_evaluation_stage_forecast_best_mse_model']),
                        local_time_series_not_improved)
            elif local_forecasts_name == 'best_mse_and_select_smartReshift_model_forecast':
                np.savetxt(''.join([local_settings['models_evaluation_path'],
                                    'best_mse_and_select_smartReshift_model_mse.csv']),
                           local_time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_results_best_mse_and_select_smartReshift_model_mse']),
                        local_time_series_treated)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'improved_time_series_best_mse_and_select_smartReshift_model']),
                        local_improved_time_series_forecast)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_not_improved_best_mse_and_select_smartReshift_model']),
                        local_time_series_not_improved)
            print('specific model evaluation saved')
            print('metadata (results, time_series) saved')
            print('specific_model_obtain_result submodule has finished')

        except Exception as e1:
            print('Error in model_obtain_results submodule')
            print(e1)
            logger.info('error in model_obtain_results submodule')
            logger.error(str(e1), exc_info=True)
            return False
        print('Success at executing model_obtain_results submodule')
        return local_time_series_not_improved
