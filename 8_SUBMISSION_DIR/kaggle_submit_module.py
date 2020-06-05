# kaggle submit functionality
import sys
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


def random_event_realization(local_time_serie_data, local_days_in_focus_frame,
                             local_forecast_horizon_days, local_nof_features_for_training):
    # computing non_zero_frequencies by time_serie, this brings the probability_of_sale = 1 - zero_frequency
    x_data = local_time_serie_data[:, -local_days_in_focus_frame:]
    probability_of_sale_list = []
    for time_serie in range(local_nof_features_for_training):
        nof_nonzeros = np.count_nonzero(x_data[time_serie, :])
        probability_of_sale_list.append(nof_nonzeros / local_days_in_focus_frame)
    probability_of_sale_array = np.array(probability_of_sale_list)
    # mean with zero included (test with zero excluded, but obtains poorer results)
    mean_last_days_frame = np.mean(x_data[:, -local_days_in_focus_frame:], axis=1)
    # triggering random event and assign sale or not, if sale then fill with mean, if no maintain with zero
    local_y_pred = np.zeros(shape=(local_nof_features_for_training, local_forecast_horizon_days))
    random_event_array_normal = np.random.rand(local_nof_features_for_training, local_forecast_horizon_days)
    pareto_dist = np.random.pareto(3, (local_nof_features_for_training, local_forecast_horizon_days))
    pareto_dist_normalized = pareto_dist / np.amax(pareto_dist)
    random_event_array = np.divide(np.add(11 * pareto_dist_normalized, 2 * random_event_array_normal), 13.)
    for time_serie, day in it.product(range(local_nof_features_for_training), range(local_forecast_horizon_days)):
        if probability_of_sale_array[time_serie] > random_event_array[time_serie, day]:
            local_y_pred[time_serie: time_serie + 1, day] = mean_last_days_frame[time_serie]
        else:
            local_y_pred[time_serie: time_serie + 1, day] = random_event_array[time_serie, day]
    return local_y_pred


# main section
if __name__ == '__main__':
    try:
        print('starting submit script\n')
        # paths
        models_folder = '/kaggle/input/models-by-group/'
        models_in_block_folder = '/kaggle/input/models-in-block/'
        models_individual_ts_folder = '/kaggle/input/models-individual-ts/'
        sales_folder = '/kaggle/input/m5-forecasting-accuracy/'
        groups_folder = '/kaggle/input/groups/'

        # constants and settings
        max_sellings_days = 1913  # 1941 will indicate passing to Evaluation, 1913 means that for now we stay in Validation stage
        apply_first_model = True
        apply_second_model = False
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
        event_iterations = 50  # with 500 we observed very subtle better results, vs more time
        random_samples = 10

        # load data
        if max_sellings_days == 1913:
            raw_data_sales = pd.read_csv(''.join([sales_folder, 'sales_train_validation.csv']))
        elif max_sellings_days == 1941:
            raw_data_sales = pd.read_csv(''.join([sales_folder, 'sales_train_evaluation.csv']))
        else:
            print('max_sellings_days do not match')
        raw_unit_sales = raw_data_sales.iloc[:, 6:].values
        print('raw sales data accessed')
        print('shape of raw sales data (product_ID, days):', raw_unit_sales.shape)

        # load clean data (divided in groups) and metadata
        time_series_not_improved = np.load(''.join([groups_folder, 'time_series_not_improved.npy']))
        improved_time_series_forecast_array = np.load(''.join([groups_folder, 'improved_time_series_forecast.npy']))
        nof_time_series_with_improved_forecasts = len(improved_time_series_forecast_array)
        poor_result_time_serie_array = np.load(''.join([groups_folder, 'poor_result_time_serie_array.npy']))
        nof_poor_result_ts_in_first_model = len(poor_result_time_serie_array)
        print('metadata loaded\n')

        # preprocess data
        print('preprocess was externally made, but the reverse rescaling and denormalize needs some computations')
        nof_time_series = raw_unit_sales.shape[0]
        nof_selling_days = raw_unit_sales.shape[1]
        if nof_selling_days == 1913:
            print('this script work with data until day1913, this is VALIDATION stage')
        elif nof_selling_days == 1941:
            print('this script work with data until day1941, this is EVALUATION stage')
        else:
            print('unexpected number of sellings days (not 1913 neither 1941), please CHECK IT')
            sys.exit()
        mean_unit_complete_time_serie = []
        scaled_unit_sales = np.zeros(shape=(nof_time_series, nof_selling_days))
        for time_serie in range(nof_time_series):
            scaled_time_serie = general_mean_scaler(raw_unit_sales[time_serie: time_serie + 1, :])[0]
            mean_unit_complete_time_serie.append(general_mean_scaler(raw_unit_sales[time_serie: time_serie + 1, :])[1])
            scaled_unit_sales[time_serie: time_serie + 1, :] = scaled_time_serie
        mean_unit_complete_time_serie = np.array(mean_unit_complete_time_serie)
        nof_features_for_training = nof_time_series
        x_data = raw_unit_sales[:, -days_in_focus_frame:]
        forecasts = np.zeros(shape=(nof_time_series * 2, forecast_horizon_days))
        print('data preparation done\n')

        # load generic model high_loss (>1.0 estimated mse) time_serie aggregated models
        if apply_second_model:
            model_name = ''.join(['generic_forecaster_individual_ts.json'])
            json_file = open(''.join([models_in_block_folder, model_name]), 'r')
            model_json = json_file.read()
            json_file.close()
            model_architecture = models.model_from_json(model_json)
            print('model structure loaded')
            model_architecture.compile(optimizer='adam', loss='mse')
            model_architecture.summary()
            individual_ts_LSTM = model_architecture
            print('generic model for individual high_loss time_serie compiled\n')

        # applying model: stochastic simulation, zero_counts and Bernoulli processes
        if apply_first_model:
            print('starting -stochastic simulation- forecast model, applying to all time_series')
            # computing non_zero_frequencies by time_serie, this brings the probability_of_sale = 1 - zero_frequency
            # obtaining representative samples, and assuming a uniform Normal distribution..
            mean_stochastic_simulations = []
            median_stochastic_simulations = []
            standard_deviation_stochastic_simulations = []
            for event in range(event_iterations):
                y_pred = random_event_realization(x_data, days_in_focus_frame,
                                                  forecast_horizon_days, nof_features_for_training)
                standard_deviation_stochastic_simulations.append(np.std(y_pred, axis=1))
                mean_stochastic_simulations.append(np.mean(y_pred, axis=1))
                median_stochastic_simulations.append(np.median(y_pred, axis=1))
            # this statistical values brings confidence interval for the "uncertainty" branch of this competition
            standard_deviation_stochastic_simulations = np.array(standard_deviation_stochastic_simulations)
            mean_stochastic_simulations = np.array(mean_stochastic_simulations)
            median_stochastic_simulations = np.array(median_stochastic_simulations)
            mean_stochastic_simulations = np.mean(mean_stochastic_simulations, axis=0)
            median_stochastic_simulations = np.mean(median_stochastic_simulations, axis=0)
            y_pred_launched = np.divide(np.add(mean_stochastic_simulations, median_stochastic_simulations), 2.)
            standard_deviation_stochastic_simulations = np.mean(standard_deviation_stochastic_simulations, axis=0)
            y_pred = []
            for time_serie in range(nof_features_for_training):
                mu, sigma = median_stochastic_simulations[time_serie], \
                            standard_deviation_stochastic_simulations[time_serie]
                y_pred.append(np.random.normal(mu, sigma, forecast_horizon_days))
            y_pred = np.array(y_pred)
            forecasts[:nof_features_for_training, :] = y_pred
            print('StochasticModel computed and forecasts done\n')

        if apply_second_model:
            print('second_model starting')
            # executing second model in estimated high_loss time_series, individual time_serie schema
            # low order sequential fully-connected-LSTM model
            time_serie_unit_sales_mean = []
            for time_serie in time_series_not_improved:
                time_serie_unit_sales_mean.append(mean_unit_complete_time_serie[time_serie])
            time_serie_unit_sales_mean = np.array(time_serie_unit_sales_mean)
            point_forecast_normalized = []
            print('loading weights for generic or specific model -individual time_serie schema, and making forecasts')
            time_series_with_specific_model = []
            # check if first model was not execute, then all the time series enters to second model forecasts
            if not apply_first_model:
                time_series_not_improved = [time_serie for time_serie in range(nof_time_series)]
            for time_serie in time_series_not_improved:
                try:
                    individual_ts_LSTM.load_weights(''.join(
                        [models_individual_ts_folder, '_individual_ts_', str(time_serie), '_model_weights_.h5']))
                    time_series_with_specific_model.append(time_serie)
                except Exception as ee1:
                    # specific weights for this time_serie not found, maintain stochastic simulation, if it was done
                    if not apply_first_model:
                        # really, 2537 is the generic model and the specific one (for time_serie 2537)
                        individual_ts_LSTM.load_weights(
                            ''.join([models_individual_ts_folder, '_individual_ts_2537_model_weights_.h5']))
                        time_series_with_specific_model.append(time_serie)
                x_input = scaled_unit_sales[time_serie: time_serie + 1, -forecast_horizon_days:]
                x_input = x_input.reshape(1, x_input.shape[1], x_input.shape[0])
                y_pred = individual_ts_LSTM.predict(x_input)
                point_forecast_normalized.append(y_pred)
            point_forecast_normalized = np.array(point_forecast_normalized)
            print(point_forecast_normalized.shape, len(time_series_not_improved))
            point_forecast_reshaped = point_forecast_normalized.reshape(point_forecast_normalized.shape[0],
                                                                        point_forecast_normalized.shape[3])
            # inverse transform (first moving_windows denormalizing and then general rescaling)
            time_serie_normalized_window_mean = np.mean(
                scaled_unit_sales[time_series_not_improved, -moving_window_length:], axis=1)
            print(point_forecast_reshaped.shape)
            denormalized_array = window_based_denormalizer(point_forecast_reshaped, time_serie_normalized_window_mean,
                                                           forecast_horizon_days)
            time_serie_unit_sales_mean = []
            for time_serie in time_series_not_improved:
                time_serie_unit_sales_mean.append(mean_unit_complete_time_serie[time_serie])
            point_forecast = general_mean_rescaler(denormalized_array, np.array(time_serie_unit_sales_mean),
                                                   forecast_horizon_days)
            point_forecast = point_forecast.reshape(np.shape(point_forecast)[1],
                                                    np.shape(point_forecast)[2])
            # filling with new improved forecasts
            time_serie_iterator, count = 0, 0
            for time_serie in time_series_not_improved:
                if time_serie in time_series_with_specific_model:
                    forecasts[time_serie, :] = point_forecast[time_serie_iterator, :]
                    count += 1
                time_serie_iterator += 1
            print('time_series modified:', count)
            print('LSTMnnModel applied to estimated high_loss time_series and forecasts updated\n')

        # submit results
        forecast_data_frame = np.genfromtxt(''.join([sales_folder, 'sample_submission.csv']), delimiter=',', dtype=None,
                                            encoding=None)
        forecast_data_frame[1:, 1:] = forecasts
        pd.DataFrame(forecast_data_frame).to_csv('submission.csv', index=False, header=None)
        print('submission created')

    except Exception as ee:
        print("Controlled error in main block___'___main_____'____")
        print(ee)
