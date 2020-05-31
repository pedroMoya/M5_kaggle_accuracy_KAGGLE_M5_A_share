# High loss identified time series module
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
# np.random.seed(3)


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


class individual_high_loss_ts_forecast:

    def forecast(self, local_mse, local_normalized_scaled_unit_sales, 
                 local_mean_unit_complete_time_serie, local_raw_unit_sales, local_settings):
        try:
            print('starting high loss (mse in aggregated LSTM) specific time_serie forecast submodule')
            # set training parameters
            with open(''.join([local_settings['hyperparameters_path'],
                               'individual_time_serie_based_model_hyperparameters.json'])) \
                    as local_r_json_file:
                model_hyperparameters = json.loads(local_r_json_file.read())
                local_r_json_file.close()
            time_steps_days = int(local_settings['time_steps_days'])
            epochs = int(model_hyperparameters['epochs'])
            batch_size = int(model_hyperparameters['batch_size'])
            workers = int(model_hyperparameters['workers'])
            optimizer_function = model_hyperparameters['optimizer']
            optimizer_learning_rate = model_hyperparameters['learning_rate']
            if optimizer_function == 'adam':
                optimizer_function = optimizers.Adam(optimizer_learning_rate)
            elif optimizer_function == 'ftrl':
                optimizer_function = optimizers.Ftrl(optimizer_learning_rate)
            losses_list = []
            loss_1 = model_hyperparameters['loss_1']
            loss_2 = model_hyperparameters['loss_2']
            loss_3 = model_hyperparameters['loss_3']
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
            metric1 = model_hyperparameters['metrics1']
            metric2 = model_hyperparameters['metrics2']
            union_settings_metrics = [metric1, metric2]
            if 'rmse' in union_settings_metrics:
                metrics_list.append(metrics.RootMeanSquaredError())
            if 'mse' in union_settings_metrics:
                metrics_list.append(metrics.MeanSquaredError())
            if 'mae' in union_settings_metrics:
                metrics_list.append(metrics.MeanAbsoluteError())
            if 'mape' in union_settings_metrics:
                metrics_list.append(metrics.MeanAbsolutePercentageError())
            l1 = model_hyperparameters['l1']
            l2 = model_hyperparameters['l2']
            if model_hyperparameters['regularizers_l1_l2'] == 'True':
                activation_regularizer = regularizers.l1_l2(l1=l1, l2=l2)
            else:
                activation_regularizer = None
            nof_features_by_training = 1
            forecaster = tf.keras.Sequential()
            print('current model for specific high loss time_series: Mix_Bid_PeepHole_LSTM_Dense_ANN')
            # first layer (DENSE)
            if model_hyperparameters['units_layer_1'] > 0:
                forecaster.add(layers.Dense(units=model_hyperparameters['units_layer_1'],
                                            activation=model_hyperparameters['activation_1'],
                                            activity_regularizer=activation_regularizer))
                forecaster.add(layers.Dropout(rate=float(model_hyperparameters['dropout_layer_1'])))
            # second LSTM layer
            if model_hyperparameters['units_layer_2'] > 0:
                forecaster.add(layers.Bidirectional(layers.RNN(
                    PeepholeLSTMCell(units=model_hyperparameters['units_layer_2'],
                                     activation=model_hyperparameters['activation_2'],
                                     activity_regularizer=activation_regularizer,
                                     dropout=float(model_hyperparameters['dropout_layer_2'])),
                    return_sequences=False)))
                forecaster.add(RepeatVector(model_hyperparameters['repeat_vector']))
            # third LSTM layer
            if model_hyperparameters['units_layer_3'] > 0:
                forecaster.add(layers.Bidirectional(layers.RNN(
                    PeepholeLSTMCell(units=model_hyperparameters['units_layer_3'],
                                     activation=model_hyperparameters['activation_3'],
                                     activity_regularizer=activation_regularizer,
                                     dropout=float(model_hyperparameters['dropout_layer_3'])),
                    return_sequences=False)))
                forecaster.add(RepeatVector(model_hyperparameters['repeat_vector']))
            # fourth layer (DENSE)
            if model_hyperparameters['units_layer_4'] > 0:
                forecaster.add(layers.Dense(units=model_hyperparameters['units_layer_4'],
                                            activation=model_hyperparameters['activation_4'],
                                            activity_regularizer=activation_regularizer))
                forecaster.add(layers.Dropout(rate=float(model_hyperparameters['dropout_layer_4'])))
            # final layer
            forecaster.add(layers.Dense(units=nof_features_by_training))
            forecaster.compile(optimizer=optimizer_function,
                               loss=losses_list,
                               metrics=metrics_list)
            # forecaster.saves(''.join([local_settings['models_path'], '_model_structure_']),
            #                 save_format='tf')
            forecaster.build(input_shape=(1, local_settings['forecast_horizon_days'], 1))
            forecaster_yaml = forecaster.to_yaml()
            with open(''.join([local_settings['models_path'], 'forecaster.yaml']), 'w') as yaml_file:
                yaml_file.write(forecaster_yaml)
            forecaster_untrained = forecaster
            print('specific time_serie model initialized and compiled')
            poor_results_mse_threshold = local_settings['poor_results_mse_threshold']
            nof_selling_days = local_normalized_scaled_unit_sales.shape[1]
            last_learning_day_in_year = np.mod(nof_selling_days, 365)
            max_selling_time = local_settings['max_selling_time']
            days_in_focus_frame = model_hyperparameters['days_in_focus_frame']
            window_input_length = local_settings['moving_window_input_length']
            window_output_length = local_settings['moving_window_output_length']
            moving_window_length = window_input_length + window_output_length
            nof_years = local_settings['number_of_years_ceil']
            time_series_individually_treated = []
            time_series_not_improved = []
            dirname = os.path.dirname(__file__)
            for result in local_mse:
                time_serie = int(result[0])
                file_path = os.path.join(dirname,
                                         ''.join(['.', local_settings['models_path'], 'specific_time_serie_',
                                                  str(time_serie), 'model_forecast_.h5']))
                if os.path.isfile(file_path) or result[1] <= poor_results_mse_threshold:
                    continue
                # training
                print('\ntime_serie: ', time_serie)
                time_serie_data = local_normalized_scaled_unit_sales[time_serie, :]
                time_serie_data = time_serie_data.reshape(time_serie_data.shape[0])
                nof_selling_days = time_serie_data.shape[0]
                # nof_moving_windows = np.int32(nof_selling_days / moving_window_length)
                remainder_days = np.mod(nof_selling_days, moving_window_length)
                window_first_days = [first_day
                                     for first_day in range(0, nof_selling_days, moving_window_length)]
                length_window_walk = len(window_first_days)
                # last_window_start = window_first_days[length_window_walk - 1]
                if remainder_days != 0:
                    window_first_days[length_window_walk - 1] = nof_selling_days - moving_window_length
                day_in_year = []
                [day_in_year.append(last_learning_day_in_year + year * 365) for year in range(nof_years)]
                stride_window_walk = model_hyperparameters['stride_window_walk']
                print('defining x_train')
                x_train = []
                if local_settings['train_model_input_data_approach'] == "all":
                    [x_train.append(time_serie_data[day - time_steps_days: day - window_output_length])
                     for day in range(time_steps_days, max_selling_time, stride_window_walk)]
                elif local_settings['train_model_input_data_approach'] == "focused":
                    [x_train.append(time_serie_data[day: day + window_input_length])
                     for last_day in day_in_year[:-1]
                     for day in range(last_day + window_output_length,
                                      last_day + window_output_length - days_in_focus_frame, -stride_window_walk)]
                    # border condition, take care with last year, working with last data available
                    [x_train.append(time_serie_data[day - window_input_length: day])
                     for last_day in day_in_year[-1:]
                     for day in range(last_day, last_day - days_in_focus_frame, -stride_window_walk)]
                    x_train = np.array(x_train)
                    print('x_train_shape:  ', x_train.shape)
                else:
                    logging.info("\ntrain_model_input_data_approach is not defined")
                    print('-a problem occurs with the data_approach settings')
                    return False, None
                print('defining y_train')
                y_train = []
                if local_settings['train_model_input_data_approach'] == "all":
                    [y_train.append(time_serie_data[day - window_output_length: day])
                     for day in range(time_steps_days, max_selling_time, stride_window_walk)]
                elif local_settings['train_model_input_data_approach'] == "focused":
                    [y_train.append(time_serie_data[day: day + window_output_length])
                     for last_day in day_in_year[:-1]
                     for day in range(last_day + window_output_length,
                                      last_day + window_output_length - days_in_focus_frame, -stride_window_walk)]
                    # border condition, take care with last year, working with last data available
                    [y_train.append(time_serie_data[day - window_output_length: day])
                     for last_day in day_in_year[-1:]
                     for day in range(last_day, last_day - days_in_focus_frame, -stride_window_walk)]
                y_train = np.array(y_train)
                factor = local_settings['amplification_factor']
                max_time_serie = np.amax(x_train)
                x_train[x_train > 0] = max_time_serie * factor
                max_time_serie = np.amax(y_train)
                y_train[y_train > 0] = max_time_serie * factor
                print('x_train and y_train built done')

                # define callbacks, checkpoints namepaths
                model_weights = ''.join([local_settings['checkpoints_path'], 'model_for_specific_time_serie_',
                                         str(time_serie), model_hyperparameters['current_model_name'],
                                         "_loss_-{loss:.4f}-.hdf5"])
                callback1 = cb.EarlyStopping(monitor='loss', patience=model_hyperparameters['early_stopping_patience'])
                callback2 = cb.ModelCheckpoint(model_weights, monitor='loss', verbose=1,
                                               save_best_only=True, mode='min')
                callbacks = [callback1, callback2]
                x_train = x_train.reshape((np.shape(x_train)[0], np.shape(x_train)[1], 1))
                y_train = y_train.reshape((np.shape(y_train)[0], np.shape(y_train)[1], 1))
                print('input_shape: ', np.shape(x_train))

                # train for each time_serie
                # check settings for repeat or not the training
                need_store_time_serie = True
                # load model
                time_series_individually_treated = np.load(''.join([local_settings['models_evaluation_path'],
                                                                    'improved_time_series_forecast.npy']))
                time_series_individually_treated = time_series_individually_treated.tolist()
                model_name = ''.join(['specific_time_serie_', str(time_serie), 'model_forecast_.h5'])
                model_path = ''.join([local_settings['models_path'], model_name])
                if os.path.isfile(model_path) and model_hyperparameters['repeat_one_by_one_training'] == "False":
                    forecaster = models.load_model(model_path, custom_objects={'modified_mape': modified_mape,
                                                                               'customized_loss': customized_loss})
                    need_store_time_serie = False
                elif model_hyperparameters['one_by_one_feature_training_done'] == "False"\
                        or model_hyperparameters['repeat_one_by_one_training'] == "True":
                    forecaster = forecaster_untrained
                    forecaster.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, workers=workers,
                                   callbacks=callbacks, shuffle=False)
                    # print summary (informative; but if says "shape = multiple", probably useless)
                    forecaster.summary()

                # compile model and make forecast
                forecaster.compile(optimizer='adam', loss='mse')

                # evaluating model and comparing with aggregated (in-block) LSTM
                print('evaluating the model trained..')
                forecast_horizon_days = local_settings['forecast_horizon_days']
                time_serie_data = time_serie_data.reshape((1, time_serie_data.shape[0], 1))
                x_input = time_serie_data[:, -forecast_horizon_days:, :]
                y_pred_normalized = forecaster.predict(x_input)
                print('output shape: ', y_pred_normalized.shape)
                y_truth = local_raw_unit_sales[time_serie, -forecast_horizon_days:]
                y_truth = y_truth.reshape(1, np.shape(y_truth)[0])
                print('y_truth shape:', y_truth.shape)

                # reversing preprocess: rescale, denormalize, reshape
                # inverse reshape
                y_pred_reshaped = y_pred_normalized.reshape((y_pred_normalized.shape[2],
                                                             y_pred_normalized.shape[1]))
                print('y_pred_reshaped shape:', y_pred_reshaped.shape)

                # inverse transform (first moving_windows denormalizing and then general rescaling)
                time_serie_data = time_serie_data.reshape(np.shape(time_serie_data)[1], 1)
                print('time_serie data shape: ', np.shape(time_serie_data))
                time_serie_normalized_window_mean = np.mean(time_serie_data[-moving_window_length:])
                print('mean of this time serie (normalized values): ', time_serie_normalized_window_mean)
                local_denormalized_array = window_based_denormalizer(y_pred_reshaped,
                                                                     time_serie_normalized_window_mean,
                                                                     forecast_horizon_days)
                local_point_forecast = general_mean_rescaler(local_denormalized_array,
                                                             local_mean_unit_complete_time_serie[time_serie],
                                                             forecast_horizon_days)
                print('rescaled denormalized forecasts array shape: ', local_point_forecast.shape)

                # calculating MSE
                local_error_metric_mse = mean_squared_error(y_truth, local_point_forecast)
                print('time_serie: ', time_serie, '\tMean_Squared_Error: ', local_error_metric_mse)
                if local_error_metric_mse < result[1]:
                    print('better results with time_serie specific model training')
                    print('MSE improved from ', result[1], 'to ', local_error_metric_mse)
                    # save models for this time serie
                    forecaster.save(''.join([local_settings['models_path'],
                                             'specific_time_serie_', str(time_serie), 'model_forecast_.h5']))
                    print('model for time_serie ', str(time_serie), " saved")
                    if need_store_time_serie:
                        time_series_individually_treated.append(int(time_serie))
                else:
                    print('no better results with time serie specific model training')
                    time_series_not_improved.append(int(time_serie))
            time_series_individually_treated = np.array(time_series_individually_treated)
            time_series_not_improved = np.array(time_series_not_improved)
            # store data of (individual-approach) time_series forecast successfully improved and those that not
            np.save(''.join([local_settings['models_evaluation_path'], 'improved_time_series_forecast']),
                    time_series_individually_treated)
            np.save(''.join([local_settings['models_evaluation_path'], 'time_series_not_improved']),
                    time_series_not_improved)
            print('forecast improvement done. (specific time_serie focused) submodule has finished')
        except Exception as submodule_error:
            print('time_series individual forecast submodule_error: ', submodule_error)
            logger.info('error in forecast of individual (high_loss_identified_ts_forecast submodule)')
            logger.error(str(submodule_error), exc_info=True)
            return False
        return True
