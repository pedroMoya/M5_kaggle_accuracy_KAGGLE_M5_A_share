# training of neural_network, in_block time_series, preprocess include: mute to sorted accumulated frequencies by ts
import os
import sys
import logging
import logging.handlers as handlers
import json
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

# functions definitions


def build_x_y_train_arrays(local_unit_sales, local_settings_arg, local_hyperparameters):
    # creating x_train and y_train arrays
    local_nof_series = local_unit_sales.shape[0]
    local_nof_selling_days = local_unit_sales.shape[1]
    local_last_learning_day_in_year = np.mod(local_nof_selling_days, 365)
    local_max_selling_time = local_settings_arg['max_selling_time']
    local_days_in_focus_frame = local_hyperparameters['days_in_focus_frame']
    local_window_input_length = local_hyperparameters['moving_window_input_length']
    local_window_output_length = local_hyperparameters['moving_window_output_length']
    local_moving_window_length = local_window_input_length + local_window_output_length
    local_time_steps_days = local_hyperparameters['time_steps_days']
    print('time_steps_days', local_time_steps_days)
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

    # amplification if settings is on, factor 1.2 was previously tuned
    # broadcasts lists to np arrays and applies the last pre-training preprocessing (amplification)
    print('x_train_shape:  ', x_train.shape)
    if local_settings_arg['amplification'] == 'True':
        factor = local_settings_arg['amplification_factor']  # factor tuning was done previously
        for time_serie_iterator in range(np.shape(x_train)[1]):
            max_time_serie = np.amax(x_train[:, time_serie_iterator, :])
            x_train[:, time_serie_iterator, :][x_train[:, time_serie_iterator, :] > 0] = \
                max_time_serie * factor
            max_time_serie = np.amax(y_train[:, time_serie_iterator, :])
            y_train[:, time_serie_iterator, :][y_train[:, time_serie_iterator, :] > 0] = \
                max_time_serie * factor

    # saving for human-eye review and reassurance; x_train and y_train are 3Ds arrays so...taking the last element
    last_time_step_x_train = x_train[-1:, :, :]
    last_time_step_x_train = last_time_step_x_train.reshape(last_time_step_x_train.shape[1],
                                                            last_time_step_x_train.shape[2])
    last_time_step_y_train = y_train[-1:, :, :]
    last_time_step_y_train = last_time_step_y_train.reshape(last_time_step_y_train.shape[1],
                                                            last_time_step_y_train.shape[2])
    np.savetxt(''.join([local_settings_arg['train_data_path'], '_from_fifth_model_x_train.csv']),
               last_time_step_x_train, fmt='%10.15f', delimiter=',', newline='\n')
    np.savetxt(''.join([local_settings_arg['train_data_path'], '_from_fifth_model_y_train.csv']),
               last_time_step_y_train, fmt='%10.15f', delimiter=',', newline='\n')
    return x_train, y_train


def build_model(local_bm_hyperparameters, local_bm_settings):
    model_built = 0
    time_steps_days = int(local_bm_hyperparameters['time_steps_days'])
    epochs = int(local_bm_hyperparameters['epochs'])
    batch_size = int(local_bm_hyperparameters['batch_size'])
    workers = int(local_bm_hyperparameters['workers'])
    optimizer_function = local_bm_hyperparameters['optimizer']
    optimizer_learning_rate = local_bm_hyperparameters['learning_rate']
    if optimizer_function == 'adam':
        optimizer_function = optimizers.Adam(optimizer_learning_rate)
    elif optimizer_function == 'ftrl':
        optimizer_function = optimizers.Ftrl(optimizer_learning_rate)
    losses_list = []
    loss_1 = local_bm_hyperparameters['loss_1']
    loss_2 = local_bm_hyperparameters['loss_2']
    loss_3 = local_bm_hyperparameters['loss_3']
    union_settings_losses = [loss_1, loss_2, loss_3]
    if 'mape' in union_settings_losses:
        losses_list.append(losses.MeanAbsolutePercentageError())
    if 'mse' in union_settings_losses:
        losses_list.append(losses.MeanSquaredError())
    if 'mae' in union_settings_losses:
        losses_list.append(losses.MeanAbsoluteError())
    if 'm_mape' in union_settings_losses:
        losses_list.append(modified_mape())
    if 'customized_loss_function' in union_settings_losses:
        losses_list.append(customized_loss())
    metrics_list = []
    metric1 = local_bm_hyperparameters['metrics1']
    metric2 = local_bm_hyperparameters['metrics2']
    union_settings_metrics = [metric1, metric2]
    if 'rmse' in union_settings_metrics:
        metrics_list.append(metrics.RootMeanSquaredError())
    if 'mse' in union_settings_metrics:
        metrics_list.append(metrics.MeanSquaredError())
    if 'mae' in union_settings_metrics:
        metrics_list.append(metrics.MeanAbsoluteError())
    if 'mape' in union_settings_metrics:
        metrics_list.append(metrics.MeanAbsolutePercentageError())
    l1 = local_bm_hyperparameters['l1']
    l2 = local_bm_hyperparameters['l2']
    if local_bm_hyperparameters['regularizers_l1_l2'] == 'True':
        activation_regularizer = regularizers.l1_l2(l1=l1, l2=l2)
    else:
        activation_regularizer = None
    nof_features_for_training = local_bm_hyperparameters['nof_features_for_training']
    # creating model
    forecaster_in_block = tf.keras.Sequential()
    print('creating the ANN model...')
    # first layer (DENSE)
    if local_bm_hyperparameters['units_layer_1'] > 0:
        forecaster_in_block.add(layers.Dense(units=local_bm_hyperparameters['units_layer_1'],
                                             activation=local_bm_hyperparameters['activation_1'],
                                             input_shape=(local_bm_hyperparameters['time_steps_days'],
                                                          nof_features_for_training),
                                             activity_regularizer=activation_regularizer))
        forecaster_in_block.add(layers.Dropout(rate=float(local_bm_hyperparameters['dropout_layer_1'])))
    # second LSTM layer
    if local_bm_hyperparameters['units_layer_2'] > 0 and local_bm_hyperparameters['units_layer_1'] > 0:
        forecaster_in_block.add(layers.Bidirectional(
            layers.LSTM(units=local_bm_hyperparameters['units_layer_2'],
                        activation=local_bm_hyperparameters['activation_2'],
                        activity_regularizer=activation_regularizer,
                        dropout=float(local_bm_hyperparameters['dropout_layer_2']),
                        return_sequences=False)))
        forecaster_in_block.add(RepeatVector(local_bm_hyperparameters['repeat_vector']))
    # third LSTM layer
    if local_bm_hyperparameters['units_layer_3'] > 0:
        forecaster_in_block.add(layers.Bidirectional(
            layers.LSTM(units=local_bm_hyperparameters['units_layer_2'],
                        activation=local_bm_hyperparameters['activation_2'],
                        activity_regularizer=activation_regularizer,
                        dropout=float(local_bm_hyperparameters['dropout_layer_2']),
                        return_sequences=True)))
        if local_bm_hyperparameters['units_layer_4'] == 0:
            forecaster_in_block.add(RepeatVector(local_bm_hyperparameters['repeat_vector']))
    # fourth layer (DENSE)
    if local_bm_hyperparameters['units_layer_4'] > 0:
        forecaster_in_block.add(layers.Dense(units=local_bm_hyperparameters['units_layer_4'],
                                             activation=local_bm_hyperparameters['activation_4'],
                                             activity_regularizer=activation_regularizer))
        forecaster_in_block.add(layers.Dropout(rate=float(local_bm_hyperparameters['dropout_layer_4'])))
    # final layer
    forecaster_in_block.add(TimeDistributed(layers.Dense(units=nof_features_for_training)))
    forecaster_in_block.save(''.join([local_bm_settings['models_path'], 'in_block_NN_model_structure_']),
                             save_format='tf')
    forecast_horizon_days = local_bm_settings['forecast_horizon_days']
    forecaster_in_block.build(input_shape=(1, forecast_horizon_days + 1, nof_features_for_training))
    forecaster_in_block.compile(optimizer=optimizer_function,
                                loss=losses_list,
                                metrics=metrics_list)
    forecaster_in_block_json = forecaster_in_block.to_json()
    with open(''.join([local_bm_settings['models_path'], 'freq_acc_forecaster_in_block.json']), 'w') as json_file:
        json_file.write(forecaster_in_block_json)
        json_file.close()
    print('build_model function finish (model structure saved in json and ts formats)')
    return True, model_built


def train_model(freq_acc_data, local_tm_hyperparameters, local_tm_settings):
    # building structure of data for training
    local_x_train, local_y_train = build_x_y_train_arrays(freq_acc_data, local_tm_settings, local_tm_hyperparameters)

    # define callbacks, checkpoints namepaths
    local_model_weights = ''.join([local_tm_settings['checkpoints_path'],
                                   'check_point_acc_freq_in_block_model',
                                   local_tm_hyperparameters['current_model_name'],
                                   "_loss_-{loss:.4f}-.hdf5"])
    local_callback1 = cb.EarlyStopping(monitor='loss', patience=local_tm_hyperparameters['early_stopping_patience'])
    local_callback2 = cb.ModelCheckpoint(local_model_weights, monitor='loss', verbose=1,
                                         save_best_only=True, mode='min')
    local_callbacks = [local_callback1, local_callback2]
    local_x_train = local_x_train.reshape((np.shape(local_x_train)[0], np.shape(local_x_train)[2],
                                           np.shape(local_x_train)[1]))
    local_y_train = local_y_train.reshape((np.shape(local_y_train)[0], np.shape(local_y_train)[2],
                                           np.shape(local_y_train)[1]))
    print('input_shape: ', np.shape(local_x_train))

    # train in block all time_series
    local_batch_size = local_tm_hyperparameters['batch_size']
    local_epochs = local_tm_hyperparameters['epochs']
    local_workers = local_tm_hyperparameters['workers']
    json_file = open(''.join([local_tm_settings['models_path'], 'freq_acc_forecaster_in_block.json']))
    local_forecaster_in_block_json = json_file.read()
    json_file.close()
    local_forecaster_in_block = models.model_from_json(local_forecaster_in_block_json)
    print('model structure loaded')
    local_forecaster_in_block.compile(optimizer='adam', loss='mse')
    local_forecaster_in_block.summary()
    print('model_compiled')
    # finally all is ready for training now
    local_forecaster_in_block.fit(local_x_train, local_y_train, batch_size=local_batch_size, epochs=local_epochs,
                                  workers=local_workers, callbacks=local_callbacks,
                                  shuffle=False)
    # print summary (informative; but if says "shape = multiple", probably useless)
    local_forecaster_in_block.summary()
    local_forecaster_in_block.save(''.join([local_tm_settings['models_path'],
                                            '_acc_freq_in_block_nn_model_.h5']))
    local_forecaster_in_block.save_weights(''.join([local_tm_settings['models_path'],
                                                    '_weights_acc_freq_in_block_nn_model_.h5']))
    print('in_block acc_frequencies NN model trained and saved in hdf5 format .h5')
    print('train_model function ended correctly')
    return True


def predict_accumulated_frequencies(local_acc_freq_data, local_nof_acc_frequencies, local_paf_settings):
    model_name = 'freq_acc_forecaster_in_block.json'
    json_file = open(''.join([local_paf_settings['models_path'], model_name]))
    local_forecaster_json = json_file.read()
    json_file.close()
    model_architecture = models.model_from_json(local_forecaster_json)
    print('model structure loaded')
    model_architecture.compile(optimizer='adam', loss='mse')
    model_architecture.summary()
    local_in_block_forecaster = model_architecture
    print('data_presorted_acc_freq in_block model compiled\n')
    # running model to make predictions
    print('making predictions of acc_freq with the model trained..')
    # reshaping to correct input_shape
    local_acc_freq_data = local_acc_freq_data.reshape((1, local_acc_freq_data.shape[1], local_acc_freq_data.shape[0]))
    local_x_input = local_acc_freq_data[:, -local_nof_acc_frequencies:, :]
    # making the predictions
    local_y_pred_normalized = local_in_block_forecaster.predict(local_x_input)
    # reshaping output
    local_y_pred = local_y_pred_normalized.reshape((local_y_pred_normalized.shape[2], local_y_pred_normalized.shape[1]))
    print('prediction of accumulated_frequencies function finish with success')
    return True, local_y_pred


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


class in_block_neural_network:

    @staticmethod
    def train_nn_model(local_preprocess_structure):
        try:
            print(local_preprocess_structure.shape)
            print(local_preprocess_structure)
            # open hyperparameters
            with open('./settings.json') as local_js_file:
                local_nn_settings = json.loads(local_js_file.read())
                local_js_file.close()
            with open(''.join([local_nn_settings['hyperparameters_path'], 
                               'freq_acc_in_block_model_hyperparameters.json'])) as local_js_file:
                local_nn_hyperparameters = json.loads(local_js_file.read())
                local_js_file.close()          
            # a final preprocess step (very simple normalization) and checking consistency
            # additionally this automatically transform the acc_absolute_freq to acc_relative_freq
            local_max_preprocess_structure = np.amax(local_preprocess_structure, axis=1)
            if local_max_preprocess_structure.shape != local_preprocess_structure[:, -1].shape:
                print('check of consistency in preprocess data not passed (in_block_neural_network training submodule)')
                return False
            else:
                print('consistency between preprocessed data and check_calculations was passed well')
            local_max_preprocess_structure[local_max_preprocess_structure == 0] = 1
            local_max_preprocess_structure = \
                local_max_preprocess_structure.reshape(local_max_preprocess_structure.shape[0], 1)
            local_preprocess_structure = np.divide(local_preprocess_structure, local_max_preprocess_structure)
            # build model
            build_model_review, in_block_model = build_model(local_nn_hyperparameters, local_nn_settings)
            if build_model_review:
                print('success at building model')
            else:
                print('error at building model')
                return False
            # training model
            train_model_review = train_model(local_preprocess_structure, local_nn_hyperparameters, local_nn_settings)
            if train_model_review:
                print('success at training model')
            else:
                print('error at training model')
                return False
            # making predictions
            local_forecast_horizon_days = local_nn_settings['forecast_horizon_days']
            nof_acc_frequencies = local_forecast_horizon_days + 1
            predict_acc_frequencies_review, predict_freq_array = predict_accumulated_frequencies(
                local_preprocess_structure, nof_acc_frequencies, local_nn_settings)
            if predict_acc_frequencies_review:
                print('success at making predictions of accumulated frequencies')
            else:
                print('error at making predictions of accumulated frequencies')
                return False
            # denormalize data
            predict_freq_array = np.multiply(predict_freq_array, local_max_preprocess_structure)
            print('freq_accumulated based neural_network training and predictions has end')
        except Exception as submodule_error:
            print('in_block neural_network training submodule_error: ', submodule_error)
            logger.info('error in in_block neural_network training')
            logger.error(str(submodule_error), exc_info=True)
            return False
        return predict_freq_array
