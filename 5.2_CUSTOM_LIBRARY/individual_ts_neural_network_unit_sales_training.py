# train and forecast time_series one by one_approach here is raw unit_sales
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
from build_local_x_y_train import local_bxy_x_y_builder

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


def simple_normalization(local_array, local_max):
    normalized_array = np.divide(local_array, local_max.clip(0.0001))
    return normalized_array


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


class neural_network_time_serie_unit_sales_schema:

    def train_model(self, local_settings, local_raw_unit_sales, local_model_hyperparameters):
        try:
            # loading hyperparameters
            local_days_in_focus = local_model_hyperparameters['days_in_focus_frame']
            local_raw_unit_sales_data = local_raw_unit_sales[:, -local_days_in_focus:]
            local_nof_ts = local_raw_unit_sales.shape[0]
            local_forecast_horizon_days = local_settings['forecast_horizon_days']
            local_features_for_each_training = 1
            print('starting neural network - individual time_serie training unit_sale_approach')

            # building architecture and compiling model_template
            # set training parameters
            local_time_steps_days = int(local_settings['time_steps_days'])
            local_epochs = int(local_model_hyperparameters['epochs'])
            local_batch_size = int(local_model_hyperparameters['batch_size'])
            local_workers = int(local_model_hyperparameters['workers'])
            local_optimizer_function = local_model_hyperparameters['optimizer']
            local_optimizer_learning_rate = local_model_hyperparameters['learning_rate']
            local_validation_split = local_model_hyperparameters['validation_split']
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
            print('building current model: individual_time_serie_acc_freq_LSTM_Dense_ANN')
            local_base_model = tf.keras.Sequential()
            # first layer (LSTM)
            if local_model_hyperparameters['units_layer_1'] > 0:
                local_base_model.add(
                    layers.LSTM(units=local_model_hyperparameters['units_layer_1'],
                                activation=local_model_hyperparameters['activation_1'],
                                input_shape=(local_model_hyperparameters['time_steps_days'],
                                             local_features_for_each_training),
                                dropout=float(local_model_hyperparameters['dropout_layer_1']),
                                activity_regularizer=local_activation_regularizer,
                                return_sequences=True))
            # second LSTM layer
            if local_model_hyperparameters['units_layer_2'] > 0:
                local_base_model.add(layers.Bidirectional(
                    layers.LSTM(units=local_model_hyperparameters['units_layer_2'],
                                activation=local_model_hyperparameters['activation_2'],
                                activity_regularizer=local_activation_regularizer,
                                dropout=float(local_model_hyperparameters['dropout_layer_2']),
                    return_sequences=False)))
                local_base_model.add(RepeatVector(local_model_hyperparameters['repeat_vector']))
            # third LSTM layer
            if local_model_hyperparameters['units_layer_3'] > 0:
                local_base_model.add(layers.Bidirectional(
                    layers.RNN(PeepholeLSTMCell(units=local_model_hyperparameters['units_layer_3'],
                                                dropout=float(local_model_hyperparameters['dropout_layer_3'])),
                               activity_regularizer=local_activation_regularizer,
                               return_sequences=False)))
                local_base_model.add(RepeatVector(local_model_hyperparameters['repeat_vector']))
            # fourth layer (DENSE)
            if local_model_hyperparameters['units_layer_4'] > 0:
                local_base_model.add(layers.Dense(units=local_model_hyperparameters['units_layer_4'],
                                                  activation=local_model_hyperparameters['activation_4'],
                                                  activity_regularizer=local_activation_regularizer))
                local_base_model.add(layers.Dropout(rate=float(local_model_hyperparameters['dropout_layer_4'])))
            # final layer
            local_base_model.add(layers.Dense(units=local_model_hyperparameters['units_final_layer']))

            # build and compile model
            local_base_model.build(input_shape=(1, local_time_steps_days, local_features_for_each_training))
            local_base_model.compile(optimizer=local_optimizer_function,
                                     loss=local_losses_list,
                                     metrics=local_metrics_list)

            # save model architecture (template for specific models)
            local_base_model.save(''.join([local_settings['models_path'],
                                           '_unit_sales_forecaster_template_individual_ts.h5']))
            local_base_model_json = local_base_model.to_json()
            with open(''.join([local_settings['models_path'],
                               '_unit_sales_forecaster_forecaster_template_individual_ts.json']), 'w') \
                    as json_file:
                json_file.write(local_base_model_json)
                json_file.close()
            local_base_model.summary()

            # training model
            local_moving_window_length = local_settings['moving_window_input_length'] + \
                                         local_settings['moving_window_output_length']

            # loading x_train and y_train, previously done for third and fourth models trainings
            local_builder = local_bxy_x_y_builder()
            local_x_train, local_y_train = local_builder.build_x_y_train_arrays(local_raw_unit_sales,
                                                                                local_settings,
                                                                                local_model_hyperparameters)
            local_x_train = local_x_train.reshape(local_x_train.shape[0], local_x_train.shape[2],
                                                  local_x_train.shape[1])
            local_y_train = local_x_train.reshape(local_y_train.shape[0], local_y_train.shape[2],
                                                  local_y_train.shape[1])

            # star training time_serie by time_serie
            local_y_pred_array = np.zeros(shape=(local_raw_unit_sales.shape[0], local_forecast_horizon_days),
                                          dtype=np.dtype('float32'))
            for time_serie in range(local_nof_ts):
                print('training time_serie:', time_serie)
                local_x, local_y = local_x_train[:, :, time_serie: time_serie + 1], \
                                   local_y_train[:, :, time_serie: time_serie + 1]
                # training, saving model and storing forecasts
                local_base_model.fit(local_x, local_y, batch_size=local_batch_size, epochs=local_epochs,
                                     workers=local_workers, callbacks=local_callbacks, shuffle=False,
                                     validation_split=local_validation_split)
                local_base_model.save_weights(''.join([local_settings['models_path'],
                                                       '/_weights_unit_sales_NN_35_days/_individual_ts_',
                                                       str(time_serie), '_model_weights_.h5']))
                local_x_input = local_raw_unit_sales[time_serie: time_serie + 1, -local_forecast_horizon_days:]
                local_x_input = local_x_input.reshape(1, local_x_input.shape[1], 1)
                # print('x_input shape:', local_x_input.shape)
                local_y_pred = local_base_model.predict(local_x_input)
                # print('x_input:\n', local_x_input)
                # print('y_pred shape:', local_y_pred.shape)
                local_y_pred = local_y_pred.reshape(local_y_pred.shape[1])
                # print('ts:', time_serie)
                # print(local_y_pred)
                local_y_pred_array[time_serie: time_serie + 1, :] = local_y_pred
            local_point_forecast_normalized = local_y_pred_array.reshape(
                (local_y_pred_array.shape[0], local_y_pred_array.shape[1]))
            local_point_forecast = local_point_forecast_normalized.clip(0)

            # save points forecast
            np.save(''.join([local_settings['train_data_path'], 'point_forecast_NN_from_unit_sales_training']),
                    local_point_forecast)
            np.save(''.join([local_settings['train_data_path'], 'eleventh_model_NN_unit_sales_forecast_data']),
                    local_point_forecast)
            np.savetxt(''.join([local_settings['others_outputs_path'], 'point_forecast_NN_from_unit_sales_training.csv']),
                       local_point_forecast, fmt='%10.15f', delimiter=',', newline='\n')
            print('point forecasts saved to file')
            print('submodule for build, train and forecast time_serie unit_sales individually finished successfully')
            return True, local_point_forecast
        except Exception as submodule_error:
            print('train model and forecast individual time_series units_sales_ submodule_error: ', submodule_error)
            logger.info('error in training and forecast-individual time_serie unit_sales_ schema')
            logger.error(str(submodule_error), exc_info=True)
            return False, []
