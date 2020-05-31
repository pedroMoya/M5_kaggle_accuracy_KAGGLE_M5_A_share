# kaggle submit functionality
import numpy as np
import pandas as pd
import itertools as it
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

# set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# clear session
kb.clear_session()


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
        forecast_horizon_based = True  # if False, input_interval is time_steps_days-based
        forecast_horizon_days = 28
        time_steps_days = 28
        moving_window_length = 28
        if forecast_horizon_based:
            input_interval = forecast_horizon_days
        else:
            input_interval = time_steps_days
        days_in_focus_frame = 28

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

        # applying model: stochastic simulation, zero_counts and Bernoulli processes
        print('starting -stochastic simulation- forecast model, applying to all time_series')
        # computing non_zero_frequencies by time_serie, this brings the probability_of_sale = 1 - zero_frequency
        nof_features_for_training = nof_time_series
        x_data = raw_unit_sales[:, -days_in_focus_frame:]
        probability_of_sale_list = []
        forecasts = np.zeros(shape=(nof_time_series * 2, forecast_horizon_days))
        for time_serie in range(nof_features_for_training):
            nof_nonzeros = np.count_nonzero(x_data[time_serie, :])
            probability_of_sale_list.append(nof_nonzeros / days_in_focus_frame)
        probability_of_sale_array = np.array(probability_of_sale_list)
        # mean with zero included (test with zero excluded, but obtains poorer results)
        mean_last_days_frame = np.mean(x_data[:, -days_in_focus_frame:], axis=1)
        # standard deviation based confidence
        confidence_unit = np.std(x_data[:, -days_in_focus_frame:], axis=1)
        # triggering random event and assign sale or not, if sale then fill with mean, if no maintain with zero
        y_pred = np.zeros(shape=(nof_features_for_training, forecast_horizon_days))
        random_event_array = np.random.rand(nof_features_for_training, forecast_horizon_days)
        for time_serie, day in it.product(range(nof_features_for_training), range(forecast_horizon_days)):
            if probability_of_sale_array[time_serie] > random_event_array[time_serie, day]:
                y_pred[time_serie: time_serie + 1, day] = mean_last_days_frame[time_serie]
            else:
                y_pred[time_serie: time_serie + 1, day] = random_event_array[time_serie, day]
        # filling with the new point_forecasts obtained
        for time_serie in range(nof_time_series):
            forecasts[time_serie, -forecast_horizon_days:] = y_pred[time_serie, -forecast_horizon_days:]
        print('StochasticModel computed and forecasts done\n')

        # submit results
        forecast_data_frame = np.genfromtxt(''.join([sales_folder, 'sample_submission.csv']), delimiter=',', dtype=None,
                                            encoding=None)
        forecast_data_frame[1:, 1:] = forecasts
        pd.DataFrame(forecast_data_frame).to_csv('submission.csv', index=False, header=None)
        print('submission created')

    except Exception as ee:
        print("Controlled error in main block___'___main_____'____")
        print(ee)
