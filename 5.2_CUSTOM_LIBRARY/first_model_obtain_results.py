# Model architecture analyzer
import os
import logging
import logging.handlers as handlers
import json
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
tf.keras.backend.set_floatx('float32')
from tensorflow.keras import layers
from tensorflow.keras.experimental import PeepholeLSTMCell
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import losses, models
from tensorflow.keras import metrics
from tensorflow.keras import callbacks as cb
from tensorflow.keras import backend as kb
from sklearn.metrics import mean_squared_error
from tensorflow.keras.utils import plot_model, model_to_dot

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

    def evaluate_stochastic_simulation(self, local_settings, local_model_hyperparameters, local_raw_unit_sales):
        try:
            local_time_series_not_improved = []
            # evaluating model and for comparing with others models forecasts
            print('evaluating the model trained..')
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
            for time_serie in poor_result_time_serie_list:
                # for time_serie in range(local_normalized_scaled_unit_sales.shape[0]):
                y_truth = local_raw_unit_sales[time_serie: time_serie + 1, -local_forecast_horizon_days:]
                local_point_forecast = local_forecasts[time_serie_iterator:time_serie_iterator + 1,
                                       -local_forecast_horizon_days:]
                # calculating error (MSE)
                local_error_metric_mse = mean_squared_error(y_truth, local_point_forecast)
                local_time_series_treated.append([int(time_serie), local_stochastic_simulation_poor_result_threshold,
                                                  local_error_metric_mse])
                if local_error_metric_mse < local_stochastic_simulation_poor_result_threshold:
                    # better results with time_serie specific model training
                    print(time_serie, 'MSE improved from inf to ', local_error_metric_mse)
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
            improved_time_series_forecast = np.array(local_improved_time_series_forecast)
            time_series_not_improved = np.array(local_time_series_not_improved)

            # store data of (individual-approach) time_series forecast successfully improved and those that not
            np.save(''.join([local_settings['models_evaluation_path'], 'time_series_forecast_results']),
                    local_time_series_treated)
            np.save(''.join([local_settings['models_evaluation_path'], 'improved_time_series_forecast']),
                    improved_time_series_forecast)
            np.save(''.join([local_settings['models_evaluation_path'], 'time_series_not_improved']),
                    time_series_not_improved)
            np.savetxt(''.join([local_settings['models_evaluation_path'], 'time_series_forecast_results.csv']),
                       local_time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
            print('in-block stochastic_simulation model evaluation saved')
            print('metadata (results, time_series) saved')
            print('first_model_obtain_result submodule has finished')

            local_time_series_not_improved = np.array(local_time_series_not_improved)
        except Exception as e1:
            print('Error in first_model_obtain_results submodule (called by _2_train module)')
            print(e1)
            return False
        print('Success at executing first_model_obtain_results submodule (called by _2_train module)')
        return local_time_series_not_improved
