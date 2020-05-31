# kaggle submit functionality
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.experimental import PeepholeLSTMCell
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import losses, models
from tensorflow.keras import metrics
from tensorflow.keras import callbacks as cb
from sklearn.metrics import mean_squared_error
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import backend as kb


class modified_mape(losses.Loss):
    @tf.function
    def call(self, local_true, local_pred):
        local_true = tf.cast(local_true, dtype=tf.float32)
        local_pred = tf.cast(local_pred, dtype=tf.float32)
        numerator = tf.abs(tf.add(local_pred, -local_true))
        denominator = tf.add(tf.convert_to_tensor(1., dtype=tf.float32), tf.abs(local_true))
        return tf.math.divide_no_nan(numerator, denominator)


class customized_loss(losses.Loss):
    @tf.function
    def call(self, local_true, local_pred):
        local_true = tf.convert_to_tensor(local_true, dtype=tf.float32)
        local_pred = tf.convert_to_tensor(local_pred, dtype=tf.float32)
        factor_difference = tf.reduce_mean(tf.abs(tf.add(local_pred, -local_true)))
        factor_true = tf.reduce_mean(tf.add(tf.convert_to_tensor(1., dtype=tf.float32), local_true))
        return tf.math.multiply_no_nan(factor_difference, factor_true)


# functions definitions


def general_mean_scaler(local_array):
    if len(local_array) == 0:
        return "argument length 0"
    mean_local_array = np.mean(local_array, axis=1)
    mean_scaling = np.divide(local_array, 1 + mean_local_array)
    return mean_scaling, mean_local_array


def window_based_normalizer(local_window_array):
    if len(local_window_array) == 0:
        return "argument length 0"
    mean_local_array = np.mean(local_window_array, axis=1)
    window_based_normalized_array = np.add(local_window_array, -mean_local_array)
    return window_based_normalized_array, mean_local_array


def general_mean_rescaler(local_array, local_complete_array_unit_mean, local_forecast_horizon):
    if len(local_array) == 0:
        return "argument length 0"
    local_array = local_array.clip(0)
    local_complete_array_unit_mean = np.array([local_complete_array_unit_mean, ] * local_forecast_horizon).transpose()
    mean_rescaling = np.multiply(local_array, 1 + local_complete_array_unit_mean)
    return mean_rescaling


def window_based_denormalizer(local_window_array, local_last_window_mean, local_forecast_horizon):
    if len(local_window_array) == 0:
        return "argument length 0"
    local_last_window_mean = np.array([local_last_window_mean, ] * local_forecast_horizon).transpose()
    window_based_denormalized_array = np.add(local_window_array, local_last_window_mean)
    return window_based_denormalized_array


# main section
if __name__ == '__main__':
    try:
        # paths
        models_folder = '/kaggle/input/models-by-group/'
        models_in_block_folder = '/kaggle/input/models-in-block/'
        sales_folder = '/kaggle/input/m5-forecasting-accuracy/'
        groups_folder = '/kaggle/input/groups/'

        # constants
        nof_groups = 3
        forecast_horizon_days = 28
        time_steps_days = 56
        moving_window_length = 56
        amplification_factor = 1.0

        # load data
        raw_data_sales = pd.read_csv(''.join([sales_folder, 'sales_train_validation.csv']))
        raw_unit_sales = raw_data_sales.iloc[:, 6:].values
        print('raw sales data accessed')

        # load clean data (divided in groups) and metadata
        preprocessed_unit_sales_g1 = np.load(''.join([groups_folder, 'group1.npy']))
        preprocessed_unit_sales_g2 = np.load(''.join([groups_folder, 'group2.npy']))
        preprocessed_unit_sales_g3 = np.load(''.join([groups_folder, 'group3.npy']))
        groups_list = [preprocessed_unit_sales_g1, preprocessed_unit_sales_g2, preprocessed_unit_sales_g3]
        print('preprocessed data by group aggregated loaded')
        indexes_in_groups_array = np.load(''.join([groups_folder, 'indexes_in_groups.npy']))
        time_series_group = np.load(''.join([groups_folder, 'time_serie_group.npy']))
        improved_time_series_forecast_array = np.load(''.join([groups_folder, 'improved_time_series_forecast.npy']))
        nof_time_series_with_improved_forecasts = len(improved_time_series_forecast_array)
        poor_result_time_serie_array = np.load(''.join([groups_folder, 'poor_result_time_serie_array.npy']))
        nof_poor_result_ts_in_first_model = len(poor_result_time_serie_array)
        print('metadata loaded\n')

        # load by-group time_serie aggregated models
        forecasters_list = []
        for group in range(nof_groups):
            model_name = ''.join(['forecaster_group', str(group), '_.json'])
            json_file = open(''.join([models_folder, model_name]), 'r')
            model_json = json_file.read()
            json_file.close()
            model_group = models.model_from_json(model_json)
            print('model structure loaded and compiled for group ', group)
            model_group.load_weights(''.join([models_folder, 'model_group_', str(group), '_forecast_.h5']))
            print('weights loaded for model group ', group)
            # model_group.compile(optimizer='adam', loss='mse')
            model_group.build(input_shape=(None, time_steps_days, np.shape(time_series_group[group][0])))
            print('model for group', group, 'input_shape defined')
            model_group.compile(optimizer='adam', loss='mse')
            forecasters_list.append(model_group)
            # model_group.summary()
        print('by group models loaded and built\n')

        # load in-block model
        model_name = 'forecaster.json'
        json_file = open(''.join([models_in_block_folder, model_name]), 'r')
        model_json = json_file.read()
        json_file.close()
        model_in_block = models.model_from_json(model_json)
        print('in-block forecast model loaded')
        model_in_block.load_weights(''.join([models_in_block_folder, '_high_loss_time_serie_model_forecaster_.h5']))
        model_in_block.compile(optimizer='adam', loss='mse')
        model_in_block.build(input_shape=(None, time_steps_days, nof_poor_result_ts_in_first_model))
        print('weights loaded and input_shape build done for in-block model\n')
        # model_in_block.summary()

        # preprocess data
        print('preprocess was externally made, but the reverse rescaling and denormalize needs some computations')
        nof_time_series = raw_unit_sales.shape[0]
        nof_selling_days = raw_unit_sales.shape[1]
        mean_unit_complete_time_serie = []
        scaled_unit_sales = np.zeros(shape=(nof_time_series, nof_selling_days))
        for time_serie in range(nof_time_series):
            scaled_time_serie = general_mean_scaler(raw_unit_sales[time_serie: time_serie + 1, :])[0]
            mean_unit_complete_time_serie.append(general_mean_scaler(raw_unit_sales[time_serie: time_serie + 1, :])[1])
            scaled_unit_sales[time_serie: time_serie + 1, :] = scaled_time_serie
        mean_unit_complete_time_serie = np.array(mean_unit_complete_time_serie)
        print('data preparation done\n')

        # make time_series by-group forecasts
        print('starting forecasts')
        nof_groups = len(groups_list)
        forecasts = np.zeros(shape=(nof_time_series * 2, forecast_horizon_days))
        for group in range(nof_groups):
            time_series_in_group = time_series_group[:, [0]][time_series_group[:, [1]] == group]
            group_data = groups_list[group]
            x_test = group_data[:, -time_steps_days:]
            x_test = x_test.reshape(1, x_test.shape[1], x_test.shape[0])
            print('x_test input for group', group, 'array prepared and ready')
            forecaster = forecasters_list[group]
            point_forecast_normalized = forecaster.predict(x_test)
            print('group ', group, 'forecast done')
            # inverse reshape
            point_forecast_reshaped = point_forecast_normalized.reshape((point_forecast_normalized.shape[2],
                                                                         point_forecast_normalized.shape[1]))
            # take forecast_horizon days
            point_forecast_normalized = point_forecast_normalized[:, -forecast_horizon_days:] * amplification_factor
            # inverse transform (first moving_windows denormalizing and then general rescaling)
            time_serie_normalized_window_mean = np.mean(groups_list[group][:, -moving_window_length:], axis=1)
            denormalized_array = window_based_denormalizer(point_forecast_reshaped,
                                                           time_serie_normalized_window_mean,
                                                           forecast_horizon_days)
            group_time_serie_unit_sales_mean = []
            for time_serie in time_series_in_group:
                group_time_serie_unit_sales_mean.append(mean_unit_complete_time_serie[time_serie])
            point_forecast = general_mean_rescaler(denormalized_array,
                                                   np.array(group_time_serie_unit_sales_mean), forecast_horizon_days)
            point_forecast = point_forecast.reshape(np.shape(point_forecast)[1], np.shape(point_forecast)[2])
            for time_serie_iterator in range(np.shape(point_forecast)[0]):
                forecasts[time_series_in_group[time_serie_iterator], :] = point_forecast[time_serie_iterator, :]
        print('time_serie by-group forecasts done\n')

        # applies a specific trained  model for time_series with forecast high loss after using the first model
        print('starting specific (in-block) (with high loss in previous steps) time_series forecast')
        x_input = scaled_unit_sales[poor_result_time_serie_array, -time_steps_days:]
        print(np.shape(x_input))
        x_input = x_input.reshape((1, np.shape(x_input)[1], np.shape(x_input)[0]))
        print('a')
        y_pred_normalized = model_in_block.predict(x_input)
        print(y_pred_normalized.shape)
        y_pred_normalized_reshaped = y_pred_normalized.reshape((y_pred_normalized.shape[2],
                                                                y_pred_normalized.shape[1]))
        y_pred_normalized = y_pred_normalized[:, -forecast_horizon_days:]
        print(y_pred_normalized_reshaped.shape)
        window = scaled_unit_sales[poor_result_time_serie_array, -time_steps_days:]
        print(window.shape)
        time_serie_normalized_window_mean = np.mean(window, axis=1)
        print(time_serie_normalized_window_mean.shape)
        denormalized_array = window_based_denormalizer(y_pred_normalized_reshaped,
                                                       time_serie_normalized_window_mean,
                                                       forecast_horizon_days)
        print(denormalized_array.shape)
        print('d')
        mean_unit_time_serie = mean_unit_complete_time_serie[time_serie]
        print('e')
        point_forecast = general_mean_rescaler(denormalized_array, np.array(mean_unit_time_serie),
                                               forecast_horizon_days)
        print('f')
        # point_forecast = point_forecast.reshape((np.shape(point_forecast)[1], np.shape(point_forecast)[0]))
        print(point_forecast.shape)
        # add in-block model forecast to corresponding time_series
        time_serie_iterator = 0
        for time_serie in improved_time_series_forecast_array:
            forecasts[time_serie, :] = point_forecast[time_serie_iterator, :] * amplification_factor
            print('l')
            time_serie_iterator += 1
        print('in-block time_serie forecasts done')

        # submit results
        forecast_data_frame = np.genfromtxt(''.join([sales_folder, 'sample_submission.csv']), delimiter=',', dtype=None,
                                            encoding=None)
        forecast_data_frame[1:, 1:] = forecasts
        pd.DataFrame(forecast_data_frame).to_csv('submission.csv', index=False, header=None)
        print('submission done')

    except Exception as ee:
        print("Controlled error in main block___'___main_____'____")
        print(ee)
