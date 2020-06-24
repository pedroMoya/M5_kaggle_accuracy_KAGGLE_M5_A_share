# train and forecast time_series one by one
import os
import sys
import logging
import logging.handlers as handlers
import json
import itertools as it
import pandas as pd
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

# opening customized sub_modules
sys.path.insert(1, local_submodule_settings['custom_library_path'])
from mini_evaluator_submodule import mini_evaluator_submodule

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
    # replacing all zeros with the non_zero min of each time_serie
    local_nof_series = local_unit_sales.shape[0]
    # pretty sure that another better and pythonic way exist
    # (change the zeros for the lesser value greater than zero row-wise)
    print('dealing with zeros....')
    for time_serie in range(local_nof_series):
        row_with_control_of_zeros = cof_zeros(local_unit_sales[time_serie, :], local_settings_arg)
        local_unit_sales[time_serie, :] = row_with_control_of_zeros

    # creating x_train and y_train arrays
    local_nof_selling_days = local_unit_sales.shape[1]
    local_last_learning_day_in_year = np.mod(local_nof_selling_days, 365)
    local_max_selling_time = local_settings_arg['max_selling_time']
    local_days_in_focus_frame = local_hyperparameters['days_in_focus_frame']
    local_window_input_length = local_settings_arg['moving_window_input_length']
    local_window_output_length = local_settings_arg['moving_window_output_length']
    local_moving_window_length = local_window_input_length + local_window_output_length
    local_time_steps_days = local_settings_arg['time_steps_days']
    # nof_moving_windows = np.int32(nof_selling_days / moving_window_length)  # not used but useful (:-/)

    # checking consistence within time_step_days and moving_window
    if local_time_steps_days != local_moving_window_length:
        print('time_steps_days and moving_window_length are not equals, that is not consistent with the preprocessing')
        print('please check it: reconcile values or rewrite the code in x_train and y_train generation for this change')
        return False  # a controlled error will occur
    local_nof_years = local_settings_arg['number_of_years_ceil']
    local_stride_window_walk = local_hyperparameters['stride_window_walk']

    # ensure that each time_step has the same length (shape of source data) with no data loss at border condition..
    # simply concatenate at the start of the array the mean of the last 2 time_steps, to complete the shape needed
    print('dealing with last time_step_days')
    rest = np.ceil(local_nof_selling_days / local_time_steps_days) * local_time_steps_days - local_nof_selling_days
    if rest != 0:
        local_mean_block = np.zeros(shape=(local_nof_series, int(rest)))
        for time_serie in range(local_nof_series):
            local_mean_block[time_serie, :] = np.mean(local_unit_sales[time_serie, - 2 * local_time_steps_days:])
        local_unit_sales = np.concatenate((local_mean_block, local_unit_sales), axis=1)
        print(local_unit_sales.shape)

    # checking that adjustments was done well
    local_nof_selling_days = local_unit_sales.shape[1]  # necessary as local_unit_sales was concatenated
    local_remainder_days = np.mod(local_nof_selling_days, local_moving_window_length)
    if local_remainder_days != 0:
        print('an error in reshaping input data at creating x_train and y_train has occurred')
        print('please recheck before continue')
        return False  # this will return a controlled error,
    local_day_in_year = []
    [local_day_in_year.append(local_last_learning_day_in_year + year * 365) for year in range(local_nof_years)]

    # building this mild no-triviality 3D arrays for training
    print('defining x_train')
    x_train = []
    if local_settings_arg['train_model_input_data_approach'] == "all":
        [x_train.append(local_unit_sales[:, day: day + local_time_steps_days])
         for day in range(local_unit_sales, local_max_selling_time, local_stride_window_walk)]
    elif local_settings_arg['train_model_input_data_approach'] == "focused":
        [x_train.append(local_unit_sales[:, day: day + local_moving_window_length])
         for last_day in local_day_in_year[:-1]
         for day in range(last_day + local_window_output_length,
                          last_day + local_window_output_length - local_days_in_focus_frame, -local_stride_window_walk)]
        # border condition, take care with last year, working with last data available
        [x_train.append(local_unit_sales[:, day - local_moving_window_length: day])
         for last_day in local_day_in_year[-1:]
         for day in range(last_day, last_day - local_days_in_focus_frame, -local_stride_window_walk)]
    else:
        logging.info("\ntrain_model_input_data_approach is not defined")
        print('-a problem occurs with the data_approach settings')
        return False, None
    print('defining y_train')
    y_train = []
    if local_settings_arg['train_model_input_data_approach'] == "all":
        [y_train.append(local_unit_sales[:,
                        day + local_stride_window_walk: day + local_time_steps_days + local_stride_window_walk])
         for day in range(local_time_steps_days, local_max_selling_time, local_stride_window_walk)]
    elif local_settings_arg['train_model_input_data_approach'] == "focused":
        [y_train.append(local_unit_sales[:,
                        day - local_stride_window_walk: day + local_moving_window_length - local_stride_window_walk])
         for last_day in local_day_in_year[:-1]
         for day in range(last_day + local_window_output_length,
                          last_day + local_window_output_length - local_days_in_focus_frame, -local_stride_window_walk)]
        # border condition, take care with last year, working with last data available
        [y_train.append(local_unit_sales[:,
                        day - local_moving_window_length - local_stride_window_walk: day - local_stride_window_walk])
         for last_day in local_day_in_year[-1:]
         for day in range(last_day, last_day - local_days_in_focus_frame, -local_stride_window_walk)]
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # saving for human-eye review and reassurance; x_train and y_train are 3Ds arrays so...taking the last element
    last_time_step_x_train = x_train[-1:, :, :]
    last_time_step_x_train = last_time_step_x_train.reshape(last_time_step_x_train.shape[1],
                                                            last_time_step_x_train.shape[2])
    last_time_step_y_train = y_train[-1:, :, :]
    last_time_step_y_train = last_time_step_y_train.reshape(last_time_step_y_train.shape[1],
                                                            last_time_step_y_train.shape[2])
    np.savetxt(''.join([local_settings_arg['train_data_path'], 'x_train.csv']),
               last_time_step_x_train, fmt='%10.15f', delimiter=',', newline='\n')
    np.savetxt(''.join([local_settings_arg['train_data_path'], 'y_train.csv']),
               last_time_step_y_train, fmt='%10.15f', delimiter=',', newline='\n')
    return x_train, y_train


def simple_normalization(local_array, local_max):
    normalized_array = np.divide(local_array, local_max.clip(0.0001))
    return normalized_array


def cof_zeros(array, local_cof_settings):
    if local_cof_settings['zeros_control'] == "True":
        local_max = np.amax(array) + 1
        array[array <= 0] = local_max
        local_min = np.amin(array)
        array[array == local_max] = local_min
    return array


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


class pinball_function_loss(losses.Loss):
    @tf.function
    def call(self, y_true, y_pred, tau=0.1):
        error = y_true - y_pred
        return kb.mean(kb.maximum(tau * error, (tau - 1) * error), axis=-1)


class neural_network_time_serie_schema:

    def train(self, local_settings, local_raw_unit_sales, local_model_hyperparameters, local_time_series_not_improved,
              raw_unit_sales_ground_truth):
        try:
            # data normalization
            local_forecast_horizon_days = local_settings['forecast_horizon_days']
            local_x_train, local_y_train = build_x_y_train_arrays(local_raw_unit_sales, local_settings,
                                                                  local_model_hyperparameters,
                                                                  local_time_series_not_improved)
            local_forecast_horizon_days = local_settings['forecast_horizon_days']
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
            if 'pinball_loss_function' in local_union_settings_losses:
                local_losses_list.append(pinball_function_loss())
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
            if local_model_hyperparameters['units_layer_2']:
                if local_model_hyperparameters['units_layer_1'] == 0:
                    local_base_model.add(layers.RNN(
                        PeepholeLSTMCell(units=local_model_hyperparameters['units_layer_2'],
                                         activation=local_model_hyperparameters['activation_2'],
                                         input_shape=(local_time_steps_days,
                                                      local_features_for_each_training),
                                         dropout=float(local_model_hyperparameters['dropout_layer_2']))))
                else:
                    local_base_model.add(layers.RNN(
                        PeepholeLSTMCell(units=local_model_hyperparameters['units_layer_2'],
                                         activation=local_model_hyperparameters['activation_2'],
                                         dropout=float(local_model_hyperparameters['dropout_layer_2']))))
                # local_base_model.add(RepeatVector(local_model_hyperparameters['repeat_vector']))
            # third layer
            if local_model_hyperparameters['units_layer_3'] > 0:
                local_base_model.add(layers.Dense(units=local_model_hyperparameters['units_layer_3'],
                                                  activation=local_model_hyperparameters['activation_3'],
                                                  activity_regularizer=local_activation_regularizer))
                local_base_model.add(layers.Dropout(rate=float(local_model_hyperparameters['dropout_layer_3'])))
            # fourth layer
            if local_model_hyperparameters['units_layer_4'] > 0:
                local_base_model.add(layers.RNN(
                    PeepholeLSTMCell(units=local_model_hyperparameters['units_layer_4'],
                                     activation=local_model_hyperparameters['activation_4'],
                                     dropout=float(local_model_hyperparameters['dropout_layer_4']))))
            local_base_model.add(layers.Dense(units=local_forecast_horizon_days))

            # build and compile model
            local_base_model.build(input_shape=(1, local_time_steps_days, local_features_for_each_training))
            local_base_model.compile(optimizer=local_optimizer_function,
                                     loss=local_losses_list,
                                     metrics=local_metrics_list)

            # save model architecture (template for specific models)
            local_base_model.save(''.join([local_settings['models_path'],
                                           'generic_forecaster_template_individual_ts.h5']))
            local_base_model_json = local_base_model.to_json()
            with open(''.join([local_settings['models_path'],
                               'generic_forecaster_template_individual_ts.json']), 'w') as json_file:
                json_file.write(local_base_model_json)
                json_file.close()
            local_base_model.summary()

            # training model
            local_moving_window_length = local_settings['moving_window_input_length'] + \
                                         local_settings['moving_window_output_length']
            # all input data in the correct type
            local_x_train = np.array(local_x_train, dtype=np.dtype('float32'))
            local_y_train = np.array(local_y_train, dtype=np.dtype('float32'))
            local_raw_unit_sales = np.array(local_raw_unit_sales, dtype=np.dtype('float32'))
            # specific time_serie models training loop
            local_y_pred_list = []
            local_nof_time_series = local_settings['number_of_time_series']
            remainder = np.array([time_serie for time_serie in range(local_nof_time_series)
                                  if time_serie not in local_time_series_not_improved])
            for time_serie in remainder:
                # ----------------------key_point---------------------------------------------------------------------
                # take note that each loop the weights and internal last states of previous training are conserved
                # that's probably save times and (in aggregated or ordered) connected time series will improve results
                # ----------------------key_point---------------------------------------------------------------------
                print('training time_serie:', time_serie)
                local_x, local_y = local_x_train[:, time_serie: time_serie + 1, :], \
                                   local_y_train[:, time_serie: time_serie + 1, :]
                local_x = local_x.reshape(local_x.shape[0], local_x.shape[2], 1)
                local_y = local_y.reshape(local_y.shape[0], local_y.shape[2], 1)
                # training, saving model and storing forecasts
                local_base_model.fit(local_x, local_y, batch_size=local_batch_size, epochs=local_epochs,
                                     workers=local_workers, callbacks=local_callbacks, shuffle=False)
                local_base_model.save_weights(''.join([local_settings['models_path'],
                                                       '/weights_last_year/_individual_ts_',
                                                       str(time_serie), '_model_weights_.h5']))
                local_x_input = local_raw_unit_sales[time_serie: time_serie + 1, -local_forecast_horizon_days:]
                local_x_input = cof_zeros(local_x_input, local_settings)
                local_x_input = local_x_input.reshape(1, local_x_input.shape[1], 1)
                print('x_input shape:', local_x_input.shape)
                local_y_pred = local_base_model.predict(local_x_input)
                print('x_input:\n', local_x_input)
                print('y_pred shape:', local_y_pred.shape)
                local_y_pred = local_y_pred.reshape(local_y_pred.shape[1])
                local_y_pred = cof_zeros(local_y_pred, local_settings)
                if local_settings['mini_ts_evaluator'] == "True" and \
                        local_settings['competition_stage'] != 'submitting_after_June_1th_using_1941days':
                    mini_evaluator = mini_evaluator_submodule()
                    evaluation = mini_evaluator.evaluate_ts_forecast(
                            raw_unit_sales_ground_truth[time_serie, -local_forecast_horizon_days:], local_y_pred)
                    print('ts:', time_serie, 'with cof_zeros ts mse:', evaluation)
                else:
                    print('ts:', time_serie)
                print(local_y_pred)
                local_y_pred_list.append(local_y_pred)
            local_point_forecast_array = np.array(local_y_pred_list)
            local_point_forecast_normalized = local_point_forecast_array.reshape(
                (local_point_forecast_array.shape[0], local_point_forecast_array.shape[1]))
            local_point_forecast = local_point_forecast_normalized

            # save points forecast
            np.savetxt(''.join([local_settings['others_outputs_path'], 'point_forecast_NN_LSTM_simulation.csv']),
                       local_point_forecast, fmt='%10.15f', delimiter=',', newline='\n')
            print('point forecasts saved to file')
            print('submodule for build, train and forecast time_serie individually finished successfully')
            return True
        except Exception as submodule_error:
            print('train model and forecast individual time_series submodule_error: ', submodule_error)
            logger.info('error in training and forecast-individual time_serie schema')
            logger.error(str(submodule_error), exc_info=True)
            return False
