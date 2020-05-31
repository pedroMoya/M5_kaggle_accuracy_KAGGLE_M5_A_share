# High loss identified time series module
import os
import logging
import logging.handlers as handlers
import json
import itertools as it
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

# set random seed for reproducibility --> done in _2_train.py module
np.random.seed(42)
tf.random.set_seed(42)

# clear session
kb.clear_session()


class in_block_high_loss_ts_forecast:

    def forecast(self, local_settings, local_raw_unit_sales, local_mse=None):
        try:
            print('starting time_series in-block forecast submodule')
            # set training parameters
            with open(''.join([local_settings['hyperparameters_path'],
                               'organic_in_block_time_serie_based_model_hyperparameters.json'])) \
                    as local_r_json_file:
                model_hyperparameters = json.loads(local_r_json_file.read())
                local_r_json_file.close()
            local_time_series_group = np.load(''.join([local_settings['train_data_path'], 'time_serie_group.npy']),
                                              allow_pickle=True)
            forecast_horizon_days = local_settings['forecast_horizon_days']
            max_selling_time = local_settings['max_selling_time']
            poor_results_mse_threshold = local_settings['poor_results_mse_threshold']
            poor_result_time_serie_list = []
            time_series_treated = []
            nof_features_for_training = 0
            nof_poor_result_time_series = nof_features_for_training
            if local_mse is None:
                nof_features_for_training = local_raw_unit_sales.shape[0]
                time_serie_data = local_raw_unit_sales
                poor_result_time_serie_list = [time_serie for time_serie in range(nof_features_for_training)]
            else:
                for result in local_mse:
                    if result[1] > poor_results_mse_threshold:
                        nof_features_for_training += 1
                        poor_result_time_serie_list.append(int(result[0]))
                        nof_poor_result_time_series = len(poor_result_time_serie_list)
                time_serie_data = np.zeros(shape=(nof_poor_result_time_series, max_selling_time))
                time_serie_iterator = 0
                for time_serie in poor_result_time_serie_list:
                    time_serie_data[time_serie_iterator, :] = local_raw_unit_sales[time_serie, :]
                    time_serie_iterator += 1

            # computing non_zero_frequencies by time_serie, this brings the probability_of_sale = 1 - zero_frequency
            days_in_focus_frame = model_hyperparameters['days_in_focus_frame']
            x_data = time_serie_data[:, -days_in_focus_frame:]
            probability_of_sale_list = []
            for time_serie in range(nof_features_for_training):
                nof_nonzeros = np.count_nonzero(x_data[time_serie, :])
                probability_of_sale_list.append(nof_nonzeros / days_in_focus_frame)
            probability_of_sale_array = np.array(probability_of_sale_list)
            # mean with zero included (test with zero excluded, but obtains poorer results)
            mean_last_days_frame = np.mean(x_data[:, -days_in_focus_frame:], axis=1)
            # triggering random event and assign sale or not, if sale then fill with mean, if no maintain with zero
            y_pred = np.zeros(shape=(nof_features_for_training, days_in_focus_frame))
            random_event_array = np.random.rand(nof_features_for_training, days_in_focus_frame)
            for time_serie, day in it.product(range(nof_features_for_training), range(days_in_focus_frame)):
                if probability_of_sale_array[time_serie] > random_event_array[time_serie, day]:
                    y_pred[time_serie: time_serie + 1, day] = mean_last_days_frame[time_serie]
                else:
                    y_pred[time_serie: time_serie + 1, day] = random_event_array[time_serie, day]

            # evaluating model and comparing with aggregated (by-group) LSTM
            print('evaluating the model trained..')
            time_serie_iterator = 0
            improved_time_series_forecast = []
            time_series_not_improved = []
            improved_mse = []
            print('evaluating model error by time_serie')
            for time_serie in poor_result_time_serie_list:
                # for time_serie in range(local_normalized_scaled_unit_sales.shape[0]):
                y_truth = local_raw_unit_sales[time_serie: time_serie + 1, -forecast_horizon_days:]
                local_point_forecast = y_pred[time_serie_iterator:time_serie_iterator + 1, -forecast_horizon_days:]
                # calculating error (MSE)
                local_error_metric_mse = mean_squared_error(y_truth, local_point_forecast)
                if local_mse is None:
                    previous_result = 100.
                else:
                    previous_result = local_mse[:, 1][local_mse[:, 0] == time_serie].item()
                time_series_treated.append([int(time_serie), previous_result, local_error_metric_mse])
                if local_error_metric_mse < previous_result:
                    # better results with time_serie specific model training
                    if local_mse is None:
                        print(time_serie, 'MSE improved from inf to ', local_error_metric_mse)
                    else:
                        print(time_serie, 'MSE improved from ', previous_result, 'to ', local_error_metric_mse)
                    improved_time_series_forecast.append(int(time_serie))
                    improved_mse.append(local_error_metric_mse)
                else:
                    # no better results with time serie specific model training
                    # print('MSE not improved from: ', previous_result, '\t current mse: ', local_error_metric_mse)
                    time_series_not_improved.append(int(time_serie))
                time_serie_iterator += 1
            time_series_treated = np.array(time_series_treated)
            improved_mse = np.array(improved_mse)
            average_mse_in_block_forecast = np.mean(time_series_treated[:, 2])
            average_mse_improved_ts = np.mean(improved_mse)
            print('poor result time serie list len:', len(poor_result_time_serie_list))
            print('mean_mse for in-block forecast:', average_mse_in_block_forecast)
            print('number of time series with better results with this forecast: ',
                  len(improved_time_series_forecast))
            print('mean_mse of time series with better results with this forecast: ', average_mse_improved_ts)
            print('not improved time series =', len(time_series_not_improved))
            improved_time_series_forecast = np.array(improved_time_series_forecast)
            time_series_not_improved = np.array(time_series_not_improved)
            poor_result_time_serie_array = np.array(poor_result_time_serie_list)
            # store data of (individual-approach) time_series forecast successfully improved and those that not
            np.save(''.join([local_settings['models_evaluation_path'], 'poor_result_time_serie_array']),
                    poor_result_time_serie_array)
            np.save(''.join([local_settings['models_evaluation_path'], 'time_series_forecast_results']),
                    time_series_treated)
            np.save(''.join([local_settings['models_evaluation_path'], 'improved_time_series_forecast']),
                    improved_time_series_forecast)
            np.save(''.join([local_settings['models_evaluation_path'], 'time_series_not_improved']),
                    time_series_not_improved)
            np.savetxt(''.join([local_settings['models_evaluation_path'], 'time_series_forecast_results.csv']),
                       time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
            print('in-block model architecture and evaluation saved')
            print('metadata (results, time_series) saved')
            print('forecast submodule has finished')
        except Exception as submodule_error:
            print('time_series in-block forecast submodule_error: ', submodule_error)
            logger.info('error in forecast of in-block time_series')
            logger.error(str(submodule_error), exc_info=True)
            return False
        return True
