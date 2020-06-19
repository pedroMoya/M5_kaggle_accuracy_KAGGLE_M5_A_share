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
            if local_forecasts_name == 'stochastic_simulation':
                print('evaluating the first model (stochastic_simulation) trained..')
                local_forecasts = np.load(''.join([local_settings['train_data_path'],
                                                   'stochastic_simulation_forecasts.npy']))
            elif local_forecasts_name == 'diff_trends_based_stochastic_model':
                print('evaluating the second model (diff_trend based stochastic model) trained..')
                local_forecasts = np.load(''.join([local_settings['train_data_path'],
                                                  'diff_pattern_based_forecasts.npy']))
            elif local_forecasts_name == 'combination_stochastic_model':
                print('evaluating the combination (first-second model) trained..')
                local_forecasts = np.load(''.join([local_settings['train_data_path'],
                                                  'first_second_model_forecasts.npy']))
            elif local_forecasts_name == 'diff_previous_based_stochastic_model':
                print('evaluating the previous_diff (first-second model) trained..')
                local_forecasts = np.load(''.join([local_settings['train_data_path'],
                                                  'diff_previous_based_stochastic_model.npy']))
            else:
                print('model_name not expected, please review the last argument')
                print('in stochastic_model_obtain_results submodule')
                return False
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
            print('mean_mse for in-block forecast:', average_mse_in_block_forecast)
            print('number of time series with better results with this forecast: ',
                  len(local_improved_time_series_forecast))
            print('mean_mse of time series with better results with this forecast: ', average_mse_improved_ts)
            print('mean_mse of time series with worse results with this forecast: ', average_mse_not_improved_ts)
            print('not improved time series =', len(local_time_series_not_improved))
            local_improved_time_series_forecast = np.array(local_improved_time_series_forecast)
            local_time_series_not_improved = np.array(local_time_series_not_improved)

            # store data of (individual-approach) time_series forecast successfully improved and those that not
            if local_forecasts_name == 'stochastic_simulation':
                np.savetxt(''.join([local_settings['models_evaluation_path'], 'stochastic_simulation_mse.csv']),
                           local_time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_forecast_results_stochastic_simulation']),
                        local_time_series_treated)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'improved_time_series_forecast_stochastic_simulation']),
                        local_improved_time_series_forecast)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_not_improved_stochastic_simulation']),
                        local_time_series_not_improved)
            elif local_forecasts_name == 'diff_trends_based_stochastic_model':
                np.savetxt(''.join([local_settings['models_evaluation_path'],
                                    'diff_trends_based_stochastic_model_mse.csv']),
                           local_time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_forecast_results_diff_trends_based']),
                        local_time_series_treated)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'improved_time_series_forecast_diff_trends_based']),
                        local_improved_time_series_forecast)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_not_improved_diff_trends_based']),
                        local_time_series_not_improved)
            elif local_forecasts_name == 'combination_stochastic_model':
                np.savetxt(''.join([local_settings['models_evaluation_path'], 'combination_stochastic_mse.csv']),
                           local_time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_forecast_results_combination_stochastic']),
                        local_time_series_treated)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'improved_time_series_forecast_combination_stochastic']),
                        local_improved_time_series_forecast)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_not_improved_combination_stochastic']),
                        local_time_series_not_improved)
            elif local_forecasts_name == 'diff_previous_based_stochastic_model':
                np.savetxt(''.join([local_settings['models_evaluation_path'], 'diff_previous_stochastic_mse.csv']),
                           local_time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_forecast_results_diff_previous_stochastic']),
                        local_time_series_treated)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'improved_time_series_forecast_diff_previous_stochastic']),
                        local_improved_time_series_forecast)
                np.save(''.join([local_settings['models_evaluation_path'],
                                 'time_series_not_improved_cdiff_previous_stochastic']),
                        local_time_series_not_improved)
            print('specific model evaluation saved')
            print('metadata (results, time_series) saved')
            print('specific_model_obtain_result submodule has finished')

        except Exception as e1:
            print('Error in model_obtain_results submodule (called by _2_train module)')
            print(e1)
            return False
        print('Success at executing model_obtain_results submodule (called by _2_train module)')
        return local_time_series_not_improved
