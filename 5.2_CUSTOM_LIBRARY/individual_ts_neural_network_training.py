# train and forecast time_series one by one
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
from tensorflow.keras.metrics import mean_absolute_percentage_error
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


def build_x_y_train_arrays(local_unit_sales, local_settings_arg,
                           local_hyperparameters, local_time_series_not_improved_arg):
    x_tray, y_train = [[]] * 2
    local_nof_selling_days = local_unit_sales.shape[1]
    local_nof_selling_days = local_unit_sales.shape[1]
    local_last_learning_day_in_year = np.mod(local_nof_selling_days, 365)
    local_max_selling_time = local_settings_arg['max_selling_time']
    local_days_in_focus_frame = local_hyperparameters['days_in_focus_frame']
    local_window_input_length = local_settings_arg['moving_window_input_length']
    local_window_output_length = local_settings_arg['moving_window_output_length']
    local_moving_window_length = local_window_input_length + local_window_output_length
    local_nof_years = local_settings_arg['number_of_years_ceil']
    local_time_steps_days = local_settings_arg['time_steps_days']
    # nof_moving_windows = np.int32(nof_selling_days / moving_window_length)
    local_remainder_days = np.mod(local_nof_selling_days, local_moving_window_length)
    local_window_first_days = [first_day
                               for first_day in range(0, local_nof_selling_days, local_moving_window_length)]
    local_length_window_walk = len(local_window_first_days)
    # last_window_start = window_first_days[length_window_walk - 1]
    if local_remainder_days != 0:
        local_window_first_days[local_length_window_walk - 1] = local_nof_selling_days - local_moving_window_length
    local_day_in_year = []
    [local_day_in_year.append(local_last_learning_day_in_year + year * 365) for year in range(local_nof_years)]
    local_stride_window_walk = local_hyperparameters['stride_window_walk']
    print('defining x_train')
    x_train = []
    if local_settings_arg['train_model_input_data_approach'] == "all":
        [x_train.append(local_unit_sales[:, day - local_time_steps_days: day - local_window_output_length])
         for day in range(local_unit_sales, local_max_selling_time, local_stride_window_walk)]
    elif local_settings_arg['train_model_input_data_approach'] == "focused":
        [x_train.append(local_unit_sales[:, day: day + local_time_steps_days])
         for last_day in local_day_in_year[:-1]
         for day in range(last_day + local_window_output_length,
                          last_day + local_window_output_length - local_days_in_focus_frame, -local_stride_window_walk)]
        # border condition, take care with last year, working with last data available, yeah really!!
        [x_train.append(np.concatenate(
            (local_unit_sales[:, day - local_window_output_length: day],
             np.zeros(shape=(local_time_series_not_improved_arg, local_time_steps_days - local_window_output_length))),
            axis=1))
            for last_day in local_day_in_year[-1:]
            for day in range(last_day, last_day - local_days_in_focus_frame, -local_stride_window_walk)]
    else:
        logging.info("\ntrain_model_input_data_approach is not defined")
        print('-a problem occurs with the data_approach settings')
        return False, None
    print('defining y_train')
    y_train = []
    if local_settings_arg['train_model_input_data_approach'] == "all":
        [y_train.append(local_unit_sales[:, day - local_time_steps_days: day])
         for day in range(local_time_steps_days, local_max_selling_time, local_stride_window_walk)]
    elif local_settings_arg['train_model_input_data_approach'] == "focused":
        [y_train.append(local_unit_sales[:, day: day + local_time_steps_days])
         for last_day in local_day_in_year[:-1]
         for day in range(last_day + local_window_output_length,
                          last_day + local_window_output_length - local_days_in_focus_frame, -local_stride_window_walk)]
        # border condition, take care with last year, working with last data available, yeah really!!
        [y_train.append(np.concatenate(
            (local_unit_sales[:, day - local_window_output_length: day],
             np.zeros(shape=(local_time_series_not_improved_arg, local_time_steps_days - local_window_output_length))),
            axis=1))
            for last_day in local_day_in_year[-1:]
            for day in range(last_day, last_day - local_days_in_focus_frame, -local_stride_window_walk)]
    return x_tray, y_train


# classes definitions

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


class neural_network_time_serie_schema:

    def train(self, local_settings, local_raw_unit_sales, local_model_hyperparameters, local_time_series_not_improved):
        try:
            # obtaining x_train and y_train
            local_x_train, local_y_train = build_x_y_train_arrays(local_raw_unit_sales, local_settings,
                                                                  local_model_hyperparameters,
                                                                  local_time_series_not_improved)
            local_forecast_horizon_days = local_settings['forecast_horizon_days']
            local_nof_features = local_x_train.shape[0]
            local_features_for_each_training = 1
            print('starting neural network - individual time_serie training')
            # building architecture and compiling model_template
            # set training parameters
            local_time_steps_days = int(local_settings['time_steps_days'])
            local_epochs = int(local_model_hyperparameters['epochs'])
            local_batch_size = int(local_model_hyperparameters['batch_size'])
            local_workers = int(local_model_hyperparameters['workers'])
            local_optimizer_function = local_model_hyperparameters['optimizer']
            local_optimizer_learning_rate = local_model_hyperparameters['learning_rate']
            if local_optimizer_function == 'adam':
                local_optimizer_function = optimizers.Adam(local_optimizer_learning_rate)
            elif local_optimizer_function == 'ftrl':
                local_optimizer_function = optimizers.Ftrl(local_optimizer_learning_rate)
            local_losses_list = []
            local_loss_1 = local_model_hyperparameters['loss_1']
            local_loss_2 = local_model_hyperparameters['loss_2']
            local_loss_3 = local_model_hyperparameters['loss_3']
            local_union_settings_losses = [local_loss_1, local_loss_2, local_loss_3]
            if 'mape' in local_union_settings_losses:
                local_losses_list.append(losses.MeanAbsolutePercentageError())
            if 'mse' in local_union_settings_losses:
                local_losses_list.append(losses.MeanSquaredError())
            if 'mae' in local_union_settings_losses:
                local_losses_list.append(losses.MeanAbsoluteError())
            if 'm_mape' in local_union_settings_losses:
                local_losses_list.append(modified_mape())
            if 'customized_loss_function' in local_union_settings_losses:
                local_losses_list.append(customized_loss())
            local_metrics_list = []
            local_metric1 = local_model_hyperparameters['metrics1']
            local_metric2 = local_model_hyperparameters['metrics2']
            local_union_settings_metrics = [local_metric1, local_metric2]
            if 'rmse' in local_union_settings_metrics:
                local_metrics_list.append(metrics.RootMeanSquaredError())
            if 'mse' in local_union_settings_metrics:
                local_metrics_list.append(metrics.MeanSquaredError())
            if 'mae' in local_union_settings_metrics:
                local_metrics_list.append(metrics.MeanAbsoluteError())
            if 'mape' in local_union_settings_metrics:
                local_metrics_list.append(metrics.MeanAbsolutePercentageError())
            local_l1 = local_model_hyperparameters['l1']
            local_l2 = local_model_hyperparameters['l2']
            if local_model_hyperparameters['regularizers_l1_l2'] == 'True':
                local_activation_regularizer = regularizers.l1_l2(l1=local_l1, l2=local_l2)
            else:
                local_activation_regularizer = None
            # define callbacks, checkpoints namepaths
            local_callback1 = cb.EarlyStopping(monitor='loss',
                                               patience=local_model_hyperparameters['early_stopping_patience'])
            local_callbacks = [local_callback1]
            print('building current model: Mix_Bid_PeepHole_LSTM_Dense_ANN')
            local_base_model = tf.keras.Sequential()
            # first layer (DENSE)
            if local_model_hyperparameters['units_layer_1'] > 0:
                # strictly dim 1 of input_shape is ['time_steps_days'] (dim 0 is number of batches: None)
                local_base_model.add(layers.Dense(units=local_model_hyperparameters['units_layer_1'],
                                                  activation=local_model_hyperparameters['activation_1'],
                                                  input_shape=(local_time_steps_days,
                                                               local_features_for_each_training),
                                                  activity_regularizer=local_activation_regularizer))
                local_base_model.add(layers.Dropout(rate=float(local_model_hyperparameters['dropout_layer_1'])))
            # second layer
            local_base_model.add(layers.Bidirectional(layers.RNN(
                PeepholeLSTMCell(units=local_model_hyperparameters['units_layer_2'],
                                 activation=local_model_hyperparameters['activation_2'],
                                 activity_regularizer=local_activation_regularizer,
                                 dropout=float(local_model_hyperparameters['dropout_layer_2'])),
                return_sequences=False)))
            local_base_model.add(RepeatVector(local_model_hyperparameters['repeat_vector']))
            # third layer
            if local_model_hyperparameters['units_layer_3'] > 0:
                local_base_model.add(layers.Dense(units=local_model_hyperparameters['units_layer_3'],
                                                  activation=local_model_hyperparameters['activation_3'],
                                                  activity_regularizer=local_activation_regularizer))
                local_base_model.add(layers.Dropout(rate=float(local_model_hyperparameters['dropout_layer_3'])))
            # fourth layer (DENSE)
            local_base_model.add(layers.Bidirectional(layers.RNN(
                PeepholeLSTMCell(units=local_model_hyperparameters['units_layer_4'],
                                 activation=local_model_hyperparameters['activation_4'],
                                 activity_regularizer=local_activation_regularizer,
                                 dropout=float(local_model_hyperparameters['dropout_layer_4'])),
                return_sequences=False)))
            local_base_model.add(RepeatVector(local_model_hyperparameters['repeat_vector']))
            # final layer
            local_base_model.add(layers.Dense(units=local_forecast_horizon_days))
            local_base_model.compile(optimizer=local_optimizer_function,
                                     loss=local_losses_list,
                                     metrics=local_metrics_list)

            # save architecture, train models and predict (inside loop) storing weights
            local_moving_window_length = local_settings['moving_window_input_length'] + \
                                         local_settings['moving_window_output_length']
            local_base_model_json = local_base_model.to_json()
            with open(''.join([local_settings['models_path'],
                               'generic_forecaster_individual_ts.json']), 'w') as json_file:
                json_file.write(local_base_model_json)
                json_file.close()
            # last_time_serie_trained = local_model_hyperparameters['last_time_serie_trained']
            # index_last_time_serie_trained = np.where(local_time_series_not_improved[:] == last_time_serie_trained)[0][0]
            # print(index_last_time_serie_trained)
            # local_time_series_not_improved = local_time_series_not_improved[index_last_time_serie_trained + 1:]
            # print(local_time_series_not_improved.shape)
            local_y_pred_list = []
            local_time_serie_iterator = 0  # 8000-14000
            for time_serie in local_time_series_not_improved[0: 1]:
                print('training time_serie:', time_serie)
                local_x, local_y = local_x_train[:, :, local_time_serie_iterator: local_time_serie_iterator + 1], \
                                   local_y_train[:, :, local_time_serie_iterator: local_time_serie_iterator + 1]
                local_base_model.fit(local_x, local_y, batch_size=local_batch_size, epochs=local_epochs,
                                     workers=local_workers, callbacks=local_callbacks, shuffle=False)
                local_base_model.save_weights(''.join([local_settings['models_path'], '/weights_fixed/_individual_ts_',
                                                       str(time_serie), '_model_weights_.h5']))
                local_x_input = local_x[-local_forecast_horizon_days:]
                local_y_pred = local_base_model(local_x_input)
                local_y_pred_list.append(local_y_pred)
            local_point_forecast_normalized = np.array(local_y_pred_list)
            print('forecast shape: ', np.shape(local_point_forecast_normalized))

            # inverse reshape
            local_window_normalized_scaled_unit_sales = np.load(''.join([local_settings['train_data_path'],
                                                                         'x_train_source.npy']))
            local_nof_time_series = local_raw_unit_sales.shape[0]
            local_nof_selling_days = local_raw_unit_sales.shape[1]
            local_mean_unit_complete_time_serie = []
            local_scaled_unit_sales = np.zeros(shape=(local_nof_time_series, local_nof_selling_days))
            for time_serie in range(local_nof_time_series):
                local_scaled_time_serie = general_mean_scaler(local_raw_unit_sales[time_serie: time_serie + 1, :])[0]
                local_mean_unit_complete_time_serie.append(
                    general_mean_scaler(local_raw_unit_sales[time_serie: time_serie + 1, :])[1])
                local_scaled_unit_sales[time_serie: time_serie + 1, :] = local_scaled_time_serie
            local_mean_unit_complete_time_serie = np.array(local_mean_unit_complete_time_serie)
            local_point_forecast_reshaped = local_point_forecast_normalized.reshape(
                (local_point_forecast_normalized.shape[2], local_point_forecast_normalized.shape[1]))
            # inverse transform (first moving_windows denormalizing and then general rescaling)
            local_time_serie_normalized_window_mean = np.mean(local_window_normalized_scaled_unit_sales[:,
                                                              -local_moving_window_length:], axis=1)

            local_denormalized_array = window_based_denormalizer(local_point_forecast_reshaped,
                                                                 local_time_serie_normalized_window_mean,
                                                                 local_time_steps_days)
            local_time_serie_unit_sales_mean = []
            [local_time_serie_unit_sales_mean.append(local_mean_unit_complete_time_serie[time_serie])
             for time_serie in local_time_series_not_improved]
            local_point_forecast = general_mean_rescaler(local_denormalized_array,
                                                         np.array(local_time_serie_unit_sales_mean),
                                                         local_time_steps_days)
            local_point_forecast = local_point_forecast.reshape(np.shape(local_point_forecast)[1],
                                                                np.shape(local_point_forecast)[2])
            local_point_forecast = local_point_forecast[:, -local_forecast_horizon_days:]

            # save points forecast
            np.savetxt(''.join([local_settings['others_outputs_path'], 'point_forecast_',
                                '_.csv']), local_point_forecast, fmt='%10.15f', delimiter=',', newline='\n')
            print('point forecasts saved to file')

            # evaluation of models forecasts according to day-wise comparison
            # forecaster(x_test) <=> y_pred
            print('\nmodels evaluation\nusing MEAN SQUARED ERROR, '
                  'MODIFIED-MEAN ABSOLUTE PERCENTAGE ERROR and MEAN ABSOLUTE ERROR')
            print('{:^19s}{:^19s}{:^19s}{:^19s}'.format('time_serie', 'error_metric_MSE',
                                                        'error_metric_Mod_MAPE', 'error_metric_MAPE'))
            local_time_series_error_mse = []
            local_time_series_error_mod_mape = []
            local_time_series_error_mape = []
            local_y_ground_truth_array = []
            local_y_pred_array = []
            local_customized_mod_mape = modified_mape()
            local_time_series_in_group = local_time_series_not_improved
            local_time_serie_iterator = 0
            for time_serie in local_time_series_in_group:
                local_y_ground_truth = local_raw_unit_sales[time_serie, -local_forecast_horizon_days:]
                local_y_pred = local_point_forecast[local_time_serie_iterator, -local_forecast_horizon_days:].flatten()
                local_error_metric_mse = mean_squared_error(local_y_ground_truth, local_y_pred)
                local_error_metric_mod_mape = 100 * local_customized_mod_mape(local_y_ground_truth, local_y_pred)
                local_error_metric_mape = mean_absolute_percentage_error(local_y_ground_truth, local_y_pred)
                # print('{:^19d}{:^19f}{:^19f}{:^19f}'.format(time_serie, error_metric_mse,
                #                                             error_metric_mod_mape, error_metric_mape))
                local_time_series_error_mse.append([time_serie, local_error_metric_mse])
                local_time_series_error_mod_mape.append(local_error_metric_mod_mape)
                local_time_series_error_mape.append(local_error_metric_mape)
                local_y_ground_truth_array.append(local_y_ground_truth)
                local_y_pred_array.append(local_y_pred)
                local_time_serie_iterator += 1
            print('individual time_serie models evaluation ended successfully')
            mean_mse = np.mean([loss_mse[1] for loss_mse in local_time_series_error_mse])
            print('time_serie mean mse: ', mean_mse)
            print('submodule for build, train and forecast time_serie individually finished successfully')
            return True
        except Exception as submodule_error:
            print('train model and forecast individual time_series submodule_error: ', submodule_error)
            logger.info('error in training and forecast-individual time_serie schema')
            logger.error(str(submodule_error), exc_info=True)
            return False
