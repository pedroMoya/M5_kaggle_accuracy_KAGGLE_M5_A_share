# calculate and register the MSE for fixed forecasts
import os
import sys
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

# load custom libraries
sys.path.insert(1, local_submodule_settings['custom_library_path'])
from mini_module_submission_generator import save_submission


# functions definitions


def calculate_mse(local_cm_settings, local_fixed_forecast, local_cm_forecasts, local_cm_ground_truth):
    local_cm_forecast_horizon_days = local_cm_settings['forecast_horizon_days']
    with open(''.join([local_cm_settings['hyperparameters_path'],
                       'organic_in_block_time_serie_based_model_hyperparameters.json'])) \
            as local_r_json_file:
        local_cm_hyperparameters = json.loads(local_r_json_file.read())
        local_r_json_file.close()
    local_stochastic_simulation_poor_result_threshold = \
        local_cm_hyperparameters['stochastic_simulation_poor_result_threshold']
    time_series_treated, improved_time_series_forecast, improved_mse, time_series_not_improved, not_improved_mse   \
        = [], [], [], [], []
    nof_cm_time_series = local_cm_forecasts.shape[0]
    for local_cm_time_serie in range(nof_cm_time_series):
        # for time_serie in range(local_normalized_scaled_unit_sales.shape[0]):
        y_truth = local_cm_ground_truth[local_cm_time_serie: local_cm_time_serie + 1, -local_cm_forecast_horizon_days:]
        local_point_forecast = local_cm_forecasts[local_cm_time_serie: local_cm_time_serie + 1, :]
        # calculating error (MSE)
        local_error_metric_mse = mean_squared_error(y_truth, local_point_forecast)
        # this is different to others codes elsewhere (others modules), the second column store the fixed_forecast
        time_series_treated.append([int(local_cm_time_serie), int(local_fixed_forecast),
                                    local_error_metric_mse])
        if local_error_metric_mse < local_stochastic_simulation_poor_result_threshold:
            # better results with time_serie specific model training
            # print(time_serie, 'MSE improved from inf to ', local_error_metric_mse)
            improved_time_series_forecast.append(int(local_cm_time_serie))
            improved_mse.append(local_error_metric_mse)
        else:
            # no better results with time serie specific model training
            # print('MSE not improved from: ', previous_result, '\t current mse: ', local_error_metric_mse)
            time_series_not_improved.append(int(local_cm_time_serie))
            not_improved_mse.append(local_error_metric_mse)
    time_series_treated = np.array(time_series_treated)
    improved_mse = np.array(improved_mse)
    not_improved_mse = np.array(not_improved_mse)
    average_mse_not_improved_ts = np.mean(not_improved_mse)
    average_mse_in_block_forecast = np.mean(time_series_treated[:, 2])
    average_mse_improved_ts = np.mean(improved_mse)
    print('mean_mse for forecast:', average_mse_in_block_forecast)
    print('number of time series with better (lower than threshold) results with this forecast: ',
          len(improved_time_series_forecast))
    print('mean_mse of time series with better results with this forecast: ', average_mse_improved_ts)
    print('mean_mse of time series with worse results with this forecast: ', average_mse_not_improved_ts)
    print('not improved time series =', len(time_series_not_improved))

    # analyzing in witch time_series the MSE is lower and storing values
    if os.path.isfile(''.join([local_cm_settings['models_evaluation_path'],
                               'fixed_time_series_results_model_mse.npy'])):
        time_series_treated_previous = np.load(''.join([local_cm_settings['models_evaluation_path'],
                                                        'fixed_time_series_results_model_mse.npy']))
        exist_previous_results = True
    else:
        exist_previous_results = False
    for local_cm_time_serie in range(nof_cm_time_series):
        if exist_previous_results:
            previous_mse = time_series_treated_previous[local_cm_time_serie, 2]
            current_mse = time_series_treated[local_cm_time_serie, 2]
            if current_mse > previous_mse:
                time_series_treated[local_cm_time_serie, [1, 2]] = \
                    time_series_treated_previous[local_cm_time_serie, [1, 2]]

    # store data of (individual-approach) time_series forecast successfully improved and those that not
    np.savetxt(''.join([local_cm_settings['models_evaluation_path'], 'fixed_model_mse.csv']),
               time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
    np.save(''.join([local_cm_settings['models_evaluation_path'], 'fixed_time_series_results_model_mse']),
            time_series_treated)
    return True


def make_forecast_structure(local_mfs_settings):
    local_fixed_forecast_mse = np.load(''.join([local_mfs_settings['models_evaluation_path'],
                                                'fixed_time_series_results_model_mse.npy']))
    local_nof_time_series = local_mfs_settings['number_of_time_series']
    local_mfs_forecast_horizon_days = local_mfs_settings['forecast_horizon_days']
    local_best_mse_y_pred = np.zeros(shape=(local_nof_time_series, local_mfs_forecast_horizon_days),
                                     dtype=np.dtype('float32'))
    for local_mfs_time_serie in range(local_nof_time_series):
        local_best_mse_y_pred[local_mfs_time_serie, :] = local_fixed_forecast_mse[local_mfs_time_serie, 1]
    return local_best_mse_y_pred

# classes definitions


class generate_fixed_forecast_and_calculate_mse:

    def execute(self, local_settings, local_raw_unit_sales):
        try:
            nof_time_series = local_settings['number_of_time_series']
            local_forecast_horizon_days = local_settings['forecast_horizon_days']
            local_forecasts = np.zeros(shape=(nof_time_series * 2, local_forecast_horizon_days),
                                       dtype=np.dtype('float32'))
            local_y_pred = np.zeros(shape=(nof_time_series, local_forecast_horizon_days),
                                    dtype=np.dtype('float32'))
            local_min_sales = np.amin(local_raw_unit_sales, axis=(0, 1))
            local_max_sales = np.amax(local_raw_unit_sales, axis=(0, 1))
            print('min unit sales: ', local_min_sales)
            print('max unit sales: ', local_max_sales)
            for fixed_forecast in range(local_min_sales, int(local_max_sales / 2)):
                local_y_pred.fill(fixed_forecast)
                local_forecast_name = ''.join(['fixed_', str(fixed_forecast), '_forecast'])

                # calculate mse and save results
                calculate_mse_review = calculate_mse(local_settings, fixed_forecast, local_y_pred, local_raw_unit_sales)
                if calculate_mse_review:
                    print('mse calculation for this fixed forecast done')
                else:
                    print('error at calculation for this fixed forecast')

            # generating correct best fixed forecasts
            local_y_pred_final = make_forecast_structure(local_settings)
            local_forecasts[:nof_time_series, :] = local_y_pred_final
            # dealing with Validation stage or Evaluation stage
            if local_settings['competition_stage'] == 'submitting_after_June_1th_using_1941days':
                local_forecasts[:nof_time_series, :] = local_raw_unit_sales[:, -local_forecast_horizon_days:]
                local_forecasts[nof_time_series:, :] = local_y_pred_final

            # saving forecast data
            local_forecast_name = 'best_mse_fixed_forecast_data'
            np.save(''.join([local_settings['train_data_path'], local_forecast_name]), local_forecasts)

            # saving submission as (local_name).csv
            save_submission_stochastic_simulation = save_submission()
            save_submission_review = \
                save_submission_stochastic_simulation.save(''.join([local_forecast_name, '_submission.csv']),
                                                           local_forecasts,
                                                           local_settings)
            if save_submission_review:
                print('best_fixed_forecast generation successfully completed')
            else:
                print('error at saving best_fixed_forecast generation')

        except Exception as submodule_error:
            print('best_fixed_forecast generation submodule_error: ', submodule_error)
            logger.info('error in best_fixed_forecast generation submodule')
            logger.error(str(submodule_error), exc_info=True)
            return False
        return True
